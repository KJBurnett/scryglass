import torch
import clip
import faiss
import json
import os
import torch.nn.functional as F
import numpy as np
import colorsys
from typing import Dict, Any
from utils_config import load_config

# Globals
models = {}
preprocessors = {}
indices = {}
index_maps = {}
art_patches = {}
device = "cpu"

config = load_config()

def load_models_global():
    global models, preprocessors, indices, index_maps, device, art_patches
    
    # Device logic from config
    target_device = config.get("device", "auto")
    if target_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = target_device
        
    print(f"Initializing Scryglass on device: {device}")
    
    # --- Load CLIP ---
    try:
        print("Loading CLIP ViT-B/32...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        models["clip"] = clip_model
        preprocessors["clip"] = clip_preprocess
        
        if os.path.exists("scryglass.index"):
            indices["clip"] = faiss.read_index("scryglass.index")
            try:
                with open("index_map.json", 'r', encoding='utf-8') as f:
                    index_maps["clip"] = json.load(f)
                print(f"CLIP loaded: {indices['clip'].ntotal} vectors")
            except:
                print("CLIP index_map.json not found or invalid.")
    except Exception as e:
        print(f"CLIP load failed: {e}")
    
    # --- Load DINOv2 ---
    try:
        print("Loading DINOv2 vitb14...")
        from torchvision import transforms as T
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dino_model = dino_model.to(device)
        dino_model.eval()
        models["dino"] = dino_model
        
        # DINOv2 preprocessing
        preprocessors["dino"] = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if os.path.exists("scryglass_dino.index"):
            indices["dino"] = faiss.read_index("scryglass_dino.index")
            try:
                with open("index_map_dino.json", 'r', encoding='utf-8') as f:
                    index_maps["dino"] = json.load(f)
                print(f"DINOv2 loaded: {indices['dino'].ntotal} vectors")
            except:
                 print("DINO index_map_dino.json not found or invalid.")
            
        # --- Load Art Patches ---
        if os.path.exists("scryglass_art_patches.pt"):
            art_patches = torch.load("scryglass_art_patches.pt", map_location=device)
            print(f"Loaded {len(art_patches)} Art Patches for spatial verification.")
            
    except Exception as e:
        print(f"DINOv2 load failed: {e}")
        import traceback
        traceback.print_exc()

def get_dominant_color_type(pil_img):
    # Median-based HLS analysis
    img_small = pil_img.resize((15, 15)) 
    arr = np.array(img_small)
    med = np.median(arr, axis=(0, 1)) 
    
    r, g, b = med / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    hue = h * 360 
    
    # --- LIGHTING RESILIENT BUCKETS ---
    # White: Must be desaturated AND light.
    if s < 0.15 and l > 0.50: return "W" 
    # Black: Must be very dark.
    if l < 0.20: return "B" 
    
    # Hue Mapping (Adjusted for warm indoor light)
    if 40 <= hue < 75: return "W" # Map Yellow/Warm-light to White
    if 75 <= hue < 165: return "G" # Green
    if 165 <= hue < 270: return "U" # Blue / Purple
    if 270 <= hue < 330: return "U" # Violet
    if 330 <= hue or hue < 40: return "R" # Red / Orange
    
    return "C" # Unknown

def get_top_k_score(sim_map, k=50):
    flat = sim_map.view(-1)
    top_vals, _ = torch.topk(flat, k=min(k, flat.numel()))
    return top_vals.mean().item()

def spatial_verification(crop_art, crop_full, ref_art, ref_full):
    # Pooled references
    ref_art_p = F.max_pool2d(ref_art, kernel_size=3, stride=1, padding=1)
    ref_full_p = F.max_pool2d(ref_full, kernel_size=3, stride=1, padding=1)
    
    # Cosine Sim
    sim_art = (crop_art * ref_art_p).sum(dim=1)
    sim_full = (crop_full * ref_full_p).sum(dim=1)
    
    return max(get_top_k_score(sim_art), get_top_k_score(sim_full))
