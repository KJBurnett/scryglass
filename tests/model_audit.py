import torch
import torch.hub
from PIL import Image
import numpy as np
import faiss
import json
import os
from torchvision import transforms as T

def run_audit():
    print("=== DINOv2 MATCH AUDIT ===")
    
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

    # 3. Define Preprocessing (matching app.py)
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_matches(img_path):
        if not os.path.exists(img_path):
            return None
        img = Image.open(img_path).convert('RGB')
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor).cpu().numpy()
        faiss.normalize_L2(emb)
        D, I = index.search(emb, 10)
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            results.append({
                "name": index_map[idx]['name'],
                "score": float(dist)
            })
        return results

    # 4. Scan Debug Folders
    if not os.path.exists('debug'):
        print("No debug folder found.")
        return
        
    debug_dirs = sorted([d for d in os.listdir('debug') if os.path.isdir(os.path.join('debug', d))], reverse=True)
    # Check top 3 latest
    for dname in debug_dirs[:3]:
        folder = os.path.join('debug', dname)
        print(f"\n--- AUDIT: {folder} ---")
        
        targets = {
            "RAW CROP": os.path.join(folder, "crop_raw.jpg"),
            "ENHANCED CROP": os.path.join(folder, "crop_enhanced.jpg"),
            "PRE-WARP CROP": os.path.join(folder, "crop_prewarp_debug.jpg")
        }
        
        for label, path in targets.items():
            matches = get_matches(path)
            if matches:
                print(f"[{label}]")
                for i, m in enumerate(matches[:5]):
                    rank_mark = ">>> " if i == 0 else "    "
                    rarity_mark = " [RARITY]" if "Rarity" in m['name'] else ""
                    kami_mark = " [KAMI]" if "Kami of Whispered Hopes" in m['name'] else ""
                    print(f"{rank_mark}#{i+1}: {m['score']:.4f} - {m['name']}{rarity_mark}{kami_mark}")
            else:
                print(f"[{label}] File missing: {path}")

if __name__ == "__main__":
    run_audit()
