import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np
import json
import os
from torchvision import transforms as T

# Load Index Map
index_path = "c:/Users/kyler/Workspace/scryglass/index_map_dino.json"
with open(index_path, "r") as f:
    index_map = json.load(f)

# Hardcoded Index Logic (Mocking app.py)
# We need to find the specific card entry to compare against
def get_card_idx_by_name(name):
    for i, c in enumerate(index_map):
        if c['name'] == name:
            return i
    return -1

# Load Model
print("Loading DINOv2...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model.eval()

preprocess = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Enhancement Logic (From app.py)
def enhance_image(pil_img):
    # Convert to CV2
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Warping Logic Enforcement (630x880) - skipped here as we assume crop is roughly correct
    # But we apply detailEnhance
    enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    
    # CLAHE
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(final)

def test_folder(folder_name, target_name):
    print(f"\n--- Testing {folder_name} for '{target_name}' ---")
    path = f"c:/Users/kyler/Workspace/scryglass/debug/{folder_name}/crop_raw.jpg"
    
    if not os.path.exists(path):
        print("Raw crop not found")
        return

    raw_pil = Image.open(path).convert("RGB")
    enhanced_pil = enhance_image(raw_pil)
    
    # Save enhanced for manual check
    enhanced_pil.save(f"c:/Users/kyler/Workspace/scryglass/debug/{folder_name}/test_enhanced.jpg")
    
    # Get Embeddings
    with torch.no_grad():
        raw_tensor = preprocess(raw_pil).unsqueeze(0).to(device)
        enh_tensor = preprocess(enhanced_pil).unsqueeze(0).to(device)
        
        raw_emb = model(raw_tensor)
        enh_emb = model(enh_tensor)
        
        raw_emb = F.normalize(raw_emb, p=2, dim=-1)
        enh_emb = F.normalize(enh_emb, p=2, dim=-1)

    # Get Index Embedding (We need to load the faiss index really, but let's just use the index map to find the ID and assuming we could compute sim)
    # Actually, simpler: Just load the index and search it.
    import faiss
    index = faiss.read_index("c:/Users/kyler/Workspace/scryglass/scryglass_dino.index")
    
    target_id = get_card_idx_by_name(target_name)
    if target_id == -1:
        print(f"Target '{target_name}' not found in map")
        return

    # Check Score for Target
    # We can't easily dot product against FAISS vectors without reconstructing them or using reconstruct
    # So we used index.reconstruct(target_id)
    
    # Note: faiss_id in map might correspond to index ID 
    # Let's verify: In build_index, we added cards sequentially. So map index == faiss index.
    
    target_vec = index.reconstruct(target_id)
    target_vec = torch.tensor(target_vec).to(device)
    
    score_raw = (raw_emb * target_vec).sum().item()
    score_enh = (enh_emb * target_vec).sum().item()
    
    print(f"RAW Score:      {score_raw:.4f}")
    print(f"ENHANCED Score: {score_enh:.4f}")
    print(f"Delta:          {score_enh - score_raw:.4f}")
    
    if score_enh > score_raw:
        print("Verdict: ENHANCED WINS")
    else:
        print("Verdict: RAW WINS")

# Test Cases
# 1. Aang (The Failure)
test_folder("20251229_233408", "Aang, Swift Savior // Aang and La, Ocean's Fury")

# 2. A Success Case (Yuna, Grand Summoner - Likely Foil/Art)
test_folder("20251229_233412", "Yuna, Grand Summoner")
