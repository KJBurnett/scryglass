import torch
import torch.nn.functional as F
from PIL import Image
import json
import os
import faiss
import numpy as np
from torchvision import transforms as T

# Load Index Map and FAISS
index_path = "c:/Users/kyler/Workspace/scryglass/scryglass_dino.index"
index_map_path = "c:/Users/kyler/Workspace/scryglass/index_map_dino.json"

index = faiss.read_index(index_path)
with open(index_map_path, "r") as f:
    index_map = json.load(f)

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to("cuda")
model.eval()

preprocess = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test Image
FAILED_FOLDER = "debug/20251229_232218"
crop_path = f"c:/Users/kyler/Workspace/scryglass/{FAILED_FOLDER}/crop_raw.jpg"

if not os.path.exists(crop_path):
    print(f"Folder not found: {FAILED_FOLDER}")
    exit()

img = Image.open(crop_path).convert('RGB')

# Benchmark Global Similarity for all 4 rotations
print(f"--- Global Audit for {FAILED_FOLDER} ---")
rotations = [img, img.rotate(90, expand=True), img.rotate(180, expand=True), img.rotate(270, expand=True)]
tensors = torch.cat([preprocess(r).unsqueeze(0) for r in rotations]).to("cuda")

with torch.no_grad():
    embs = model(tensors)
    embs = F.normalize(embs, p=2, dim=-1).cpu().numpy().astype('float32')

target_name = "Aang, Swift Savior // Aang and La, Ocean's Fury"
aang_index_id = -1
for i, card in enumerate(index_map):
    if card['name'] == target_name:
        aang_index_id = i
        break

if aang_index_id == -1:
    print(f"Error: {target_name} not found in index map!")
    exit()

# Search
D, I = index.search(embs, 1000) # Check top 1000

found = False
for r_idx in range(4):
    print(f"\nRotation {r_idx*90} degrees:")
    rank = -1
    for k in range(1000):
        if I[r_idx][k] == aang_index_id:
            rank = k + 1
            score = D[r_idx][k]
            print(f"  FOUND Aang at Rank {rank} with Score {score:.4f}")
            found = True
            break
    
    if rank == -1:
        print(f"  Aang NOT in top 100 global matches.")
        
    # Show Top 5 for this rotation
    print("  Top 5 for this rotation:")
    for k in range(5):
        idx = int(I[r_idx][k])
        name = index_map[idx]['name']
        score = D[r_idx][k]
        print(f"    {k+1}. {name} ({score:.4f})")

if not found:
    print("\nCRITICAL: Aang totally missing from top 100.")
