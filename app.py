import torch
import torch.nn.functional as F
from PIL import Image
import json
import numpy as np
import io
import base64
import time
import os
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import colorsys

# Utilities
import utils_config
import utils_cv
import utils_ai

app = FastAPI(title="Scryglass Phase 1")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IdentifyRequest(BaseModel):
    image: str # base64 full frame
    click_x: int
    click_y: int
    model: str = "dino"

@app.on_event("startup")
async def startup_event():
    utils_ai.load_models_global()

@app.get("/status")
async def get_status():
    return {"ready": len(utils_ai.models) > 0}

@app.post("/identify")
async def identify_card(req: IdentifyRequest):
    debug_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "click": {"x": req.click_x, "y": req.click_y},
        "success": False,
        "score": 0.0,
        "match": None,
        "best_guess": None
    }
    
    # Create debug dir
    ts_name = time.strftime("%Y%m%d_%H%M%S")
    debug_dir = f"debug/{ts_name}"
    os.makedirs(debug_dir, exist_ok=True)
    
    try:
        # Decode base64
        if "," in req.image:
            header, encoded = req.image.split(",", 1)
        else:
            encoded = req.image
            
        image_data = base64.b64decode(encoded)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        cv_image = np.array(pil_image) 
        cv_image = cv_image[:, :, ::-1].copy() # RGB to BGR

        # DEBUG: Save full
        debug_full = cv_image.copy()
        cv2.circle(debug_full, (req.click_x, req.click_y), 10, (0, 0, 255), -1)
        cv2.imwrite(f"{debug_dir}/full.jpg", debug_full)

        # --- 1. COMPUTER VISION PHASE ---
        click_point = (req.click_x, req.click_y)
        
        # Try to find contour
        # utils_cv.find_card_contour handles the multi-stream tournament
        contour = utils_cv.find_card_contour(cv_image, click_point, debug_dir, debug_info)
        
        warped = None
        
        if contour is not None:
             # REFINE: approxPolyDP
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Store polygon for UI
            ui_polygon = approx
            if len(approx) != 4:
                rect = cv2.minAreaRect(contour)
                ui_polygon = np.int32(cv2.boxPoints(rect))
            debug_info["detected_polygon"] = ui_polygon.reshape(-1, 2).tolist()
            
            # 1. Try Precise 4-Point Perspective
            if len(approx) == 4:
                try:
                    pts = approx.reshape(4, 2).astype("float32")
                    temp_warped = utils_cv.four_point_transform(cv_image, pts)
                    th, tw = temp_warped.shape[:2]
                    if th > 200 and tw > 200:
                        warped = temp_warped
                        print(f"Precise Warp Success: {tw}x{th}")
                except Exception as e:
                    print(f"Precise Warp Failed: {e}")

            # 2. Robust Fallback (MinAreaRect)
            if warped is None:
                print("Using Fallback: MinAreaRect")
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                warped = utils_cv.four_point_transform(cv_image, box)
            
            # 3. Minimum Size Check
            h, w = warped.shape[:2]
            if h < 100 or w < 100:
                 print(f"WARNING: Crop too small ({w}x{h}). Using bounding rect fallback.")
                 x, y, bw, bh = cv2.boundingRect(contour)
                 padding = int(0.1 * max(bw, bh))
                 x = max(0, x - padding)
                 y = max(0, y - padding)
                 bw = min(cv_image.shape[1] - x, bw + 2*padding)
                 bh = min(cv_image.shape[0] - y, bh + 2*padding)
                 warped = cv_image[y:y+bh, x:x+bw]
                 
            debug_info["crop_res"] = f"{warped.shape[1]}x{warped.shape[0]}"

        # Click-Centered Fallback if CV failed
        if warped is None:
            print("Smart detection failed. Using click-centered fallback.")
            h, w = cv_image.shape[:2]
            cx, cy = req.click_x, req.click_y
            crop_w, crop_h = 400, 560
            x1 = max(0, cx - crop_w//2)
            y1 = max(0, cy - crop_h//2)
            x2 = min(w, cx + crop_w//2)
            y2 = min(h, cy + crop_h//2)
            crop = pil_image.crop((x1, y1, x2, y2))
            warped = np.array(crop)[:, :, ::-1]
            debug_info["detected_polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            debug_info["fallback_used"] = True

        # --- PRE-PROCESSING PHASE ---
        # The Enforcer: Ensure Vertical 630x880
        h_curr, w_curr = warped.shape[:2]
        if w_curr > h_curr:
             warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        
        start_w, start_h = 630, 880
        warped = cv2.resize(warped, (start_w, start_h), interpolation=cv2.INTER_CUBIC)
        
        # Save Raw for AI
        cv2.imwrite(f"{debug_dir}/crop_raw.jpg", warped)
        warped_raw = warped.copy()
        
        # Enhance for user debug (optional, not used for AI currently per instruction)
        warped_enhanced = utils_cv.enhance_image(warped)
        cv2.imwrite(f"{debug_dir}/crop_enhanced.jpg", warped_enhanced)

        # --- 2. AI IDENTIFICATION PHASE ---
        
        # Select Model
        selected_model = req.model if req.model in utils_ai.models else "dino"
        model = utils_ai.models[selected_model]
        preprocess = utils_ai.preprocessors[selected_model]
        index = utils_ai.indices.get(selected_model)
        index_map = utils_ai.index_maps.get(selected_model)
        
        if not index or not index_map:
            raise HTTPException(status_code=500, detail="Models not loaded")

        # Encode (Multi-Rotation Global Search)
        raw_rgb = cv2.cvtColor(warped_raw, cv2.COLOR_BGR2RGB)
        raw_pil = Image.fromarray(raw_rgb)
        
        rotations = [
            raw_pil,
            raw_pil.rotate(90, expand=True),
            raw_pil.rotate(180, expand=True),
            raw_pil.rotate(270, expand=True)
        ]
        
        input_tensors = torch.cat([preprocess(img).unsqueeze(0) for img in rotations]).to(utils_ai.device)
        
        # Dual-Zone Spatial Patches
        # 1. Art Zone
        w_pil, h_pil = raw_pil.size
        art_x1, art_y1 = int(w_pil * 0.08), int(h_pil * 0.11)
        art_x2, art_y2 = int(w_pil * 0.92), int(h_pil * 0.55)
        
        art_pil = raw_pil.crop((art_x1, art_y1, art_x2, art_y2))
        art_pil_upside = raw_pil.rotate(180).crop((art_x1, art_y1, art_x2, art_y2))
        
        art_pil.save(f"{debug_dir}/crop_art.jpg") # Debug
        
        with torch.no_grad():
             # Global Embeddings
            global_embs = model(input_tensors)
            global_embs = F.normalize(global_embs, p=2, dim=-1)
            
            # Patch Embeddings (Robust Mode: Art 0/180, Full 0/180)
            full_pil_0 = rotations[0]
            full_pil_180 = rotations[2]
            
            img_batch = torch.cat([
                preprocess(art_pil).unsqueeze(0),
                preprocess(art_pil_upside).unsqueeze(0),
                preprocess(full_pil_0).unsqueeze(0),
                preprocess(full_pil_180).unsqueeze(0)
            ]).to(utils_ai.device)
            
            layers = model.get_intermediate_layers(img_batch, n=1, reshape=True)
            all_patches = F.normalize(layers[0], p=2, dim=1)
            
            # Split patches
            crop_art_0 = all_patches[0:1]
            crop_art_180 = all_patches[1:2]
            crop_full_0 = all_patches[2:3]
            crop_full_180 = all_patches[3:4]
            
        vectors = global_embs.cpu().numpy().astype('float32')
        search_depth = utils_config.SPATIAL_CANDIDATES # 200
        
        D, I = index.search(vectors, search_depth)
        print(f"Global Search Depth: {search_depth}")
        
        initial_candidates = []
        # Collect candidates from all 4 rotations
        for r_idx in range(4):
            for k in range(search_depth):
                idx = int(I[r_idx][k])
                if idx < 0 or idx >= len(index_map): continue
                card = index_map[idx]
                if not any(c['name'] == card['name'] for c in initial_candidates):
                    initial_candidates.append({
                        "name": card['name'],
                        "set": card['set'],
                        "global_score": float(D[r_idx][k]),
                        "scryfall_id": card.get('scryfall_id'),
                        "color_identity": card.get('color_identity', []),
                        "image": card['high_res_url'] or card['image_path']
                    })

        # --- 3. VERIFICATION PHASE ---
        print(f"Verifying {len(initial_candidates)} candidates...")
        all_candidates = []
        
        for cand in initial_candidates:
            sid = cand['scryfall_id']
            if sid in utils_ai.art_patches:
                data = utils_ai.art_patches[sid]
                ref_art = data["art"].to(utils_ai.device)
                ref_full = data["full"].to(utils_ai.device)
                
                # Spatial Check 0
                score_0 = utils_ai.spatial_verification(crop_art_0, crop_full_0, ref_art, ref_full)
                # Spatial Check 180
                score_180 = utils_ai.spatial_verification(crop_art_180, crop_full_180, ref_art, ref_full)
                
                spatial_score = max(score_0, score_180)
                is_upside = score_180 > score_0
                
                # Color Logic
                color_weight = 0.0
                try:
                    active_art = art_pil_upside if is_upside else art_pil
                    webcam_color = utils_ai.get_dominant_color_type(active_art)
                    card_colors = cand.get('color_identity', [])
                    
                    if webcam_color != "C":
                        # Simplification of color logic for readability
                        if webcam_color in card_colors:
                            color_weight = 0.05 # Bonus
                        elif len(card_colors) > 0:
                            color_weight = -0.03 # Penalty
                except:
                    pass
                
                # Blending
                g_score = cand["global_score"]
                final_score = (0.3 * g_score) + (0.7 * spatial_score) + color_weight
                
                all_candidates.append({
                    **cand,
                    "score": final_score,
                    "spatial_verification": spatial_score
                })
            else:
                 all_candidates.append({**cand, "score": cand["global_score"], "spatial_verification": 0.0})

        # Sort and Return
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_5 = all_candidates[:5]
        debug_info["candidates"] = top_5
        
        best = top_5[0] if top_5 else None
        if best:
             debug_info["score"] = best['score']
             debug_info["best_guess"] = best['name']
             if best['score'] >= utils_config.CONFIDENCE_THRESHOLD:
                 debug_info["success"] = True
                 debug_info["match"] = best
                 
        debug_info["threshold"] = utils_config.CONFIDENCE_THRESHOLD
        
        with open(f"{debug_dir}/info.json", "w") as f:
            json.dump(debug_info, f, indent=2)
            
        return {
            "detected_polygon": debug_info.get("detected_polygon"),
            "fallback_used": debug_info.get("fallback_used", False),
            "candidates": top_5,
            "match": best if debug_info["success"] else None,
            "score": best['score'] if best else 0.0,
            "best_guess": best
        }
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    return {"status": "ok", "service": "Scryglass Phase 1"}
