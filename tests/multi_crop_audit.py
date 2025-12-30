import torch
import torch.hub
from PIL import Image
import numpy as np
import faiss
import json
import os
from torchvision import transforms as T
import cv2
import torch.nn.functional as F

def run_multi_crop_audit():
    print("=== DINOv2 MULTI-CROP + SPATIAL AUDIT ===")
    
    # 1. Setup Environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading DINOv2 vitb14...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = model.to(device)
    model.eval()

    # 2. Load Index and Maps
    index_file = 'scryglass_dino.index'
    map_file = 'index_map_dino.json'
    
    if not os.path.exists(index_file) or not os.path.exists(map_file):
        print("Error: DINOv2 index or map files missing.")
        return

    index = faiss.read_index(index_file)
    with open(map_file, 'r', encoding='utf-8') as f:
        index_map = json.load(f)

    # 3. Define Preprocessing
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_features(img):
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            # Get global embedding
            emb = model(tensor)
            # Get patch features (last layer)
            # get_intermediate_layers returns a list of tensors [B, N, D]
            patch_layers = model.get_intermediate_layers(tensor, n=1, reshape=True)
            patches = patch_layers[0] # [1, 768, 16, 16] for vitb14
            
        return emb.cpu().numpy(), patches.cpu()

    def get_top_candidates(img):
        emb, _ = get_features(img)
        faiss.normalize_L2(emb)
        D, I = index.search(emb, 15)
        
        candidates = {}
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            name = index_map[idx]['name']
            candidates[name] = {
                "name": name,
                "score": float(dist),
                "rank": i + 1,
                "scryfall_id": index_map[idx].get('scryfall_id')
            }
        return candidates

    # 4. Scan Debug Folders
    if not os.path.exists('debug'):
        print("No debug folder found.")
        return
        
    debug_dirs = sorted([d for d in os.listdir('debug') if os.path.isdir(os.path.join('debug', d))], reverse=True)
    
    for dname in debug_dirs[:2]: # Audit latest 2 failures
        folder = os.path.join('debug', dname)
        raw_path = os.path.join(folder, "crop_raw.jpg")
        
        if not os.path.exists(raw_path):
            continue
            
        print(f"\n--- AUDITING: {folder} ---")
        
        # Load Raw Image
        img_raw_cv = cv2.imread(raw_path)
        img_raw_rgb = cv2.cvtColor(img_raw_cv, cv2.COLOR_BGR2RGB)
        pil_raw = Image.fromarray(img_raw_rgb)
        
        # Generate Global Candidates
        results_raw = get_top_candidates(pil_raw)
        
        # Get Patch Features for the Raw Crop (to use in spatial verification)
        _, crop_patches = get_features(pil_raw)
        # Normalize patches
        crop_patches = F.normalize(crop_patches, p=2, dim=1) # [1, 768, 16, 16]
        
        # Aggregate Voting (using RAW as base)
        candidates_to_verify = list(results_raw.values())[:5]
        
        print(f"\nPerforming Spatial Verification on Top 5...")
        
        spatial_results = []
        for cand in candidates_to_verify:
            name = cand['name']
            card_id = cand['scryfall_id']
            ref_path = f"images/{card_id}.jpg"
            
            if not os.path.exists(ref_path):
                print(f"Ref missing for {name}: {ref_path}")
                spatial_results.append({**cand, "spatial_score": 0.0})
                continue
                
            # Load Ref
            ref_img = Image.open(ref_path).convert('RGB')
            _, ref_patches = get_features(ref_img)
            ref_patches = F.normalize(ref_patches, p=2, dim=1) # [1, 768, 16, 16]
            
            # Compute Spatial Similarity (Patch-to-Patch)
            # Simple element-wise dot product and average
            similarity_map = (crop_patches * ref_patches).sum(dim=1) # [1, 16, 16]
            spatial_score = similarity_map.mean().item()
            
            spatial_results.append({
                **cand,
                "spatial_score": spatial_score
            })
            
        # Re-Rank by Spatial Score
        spatial_results.sort(key=lambda x: x['spatial_score'], reverse=True)
        
        print(f"{'Rank':<5} | {'Global':<8} | {'Spatial':<8} | {'Name'}")
        print("-" * 65)
        for i, res in enumerate(spatial_results):
            rarity_mark = " [TARGET]" if "Rarity" in res['name'] else ""
            print(f"#{i+1:<4} | {res['score']:<8.4f} | {res['spatial_score']:<8.4f} | {res['name']}{rarity_mark}")

if __name__ == "__main__":
    run_multi_crop_audit()
