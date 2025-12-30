import torch
import clip
from PIL import Image
import faiss
import numpy as np
import json
import os
import sys

# Paths
IMAGE_PATH = "debug/20251229_185128/crop_enhanced.jpg"
if not os.path.exists(IMAGE_PATH):
    print(f"Error: {IMAGE_PATH} not found")

INDEX_PATH = "scryglass.index"
MAP_PATH = "index_map.json"
MODEL_NAME = "ViT-B/32"

def main():
    print(f"Loading Index: {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    
    with open(MAP_PATH, 'r') as f:
        index_map = json.load(f)
        
    print("Loading CLIP (cpu)...")
    device = "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    
    print(f"Processing Image: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    # Preprocess
    img_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.encode_image(img_input)
        features /= features.norm(dim=-1, keepdim=True)
        
    vector = features.cpu().numpy().astype('float32')
    
    # 1. Search Top 5 (Using Index)
    D, I = index.search(vector, 5)
    print("\n--- FAISS TOP 5 ---")
    for i in range(5):
        idx = int(I[0][i])
        score = float(D[0][i])
        card = index_map[idx]
        print(f"Rank {i+1}: {score:.4f} - {card['name']} ({card['set']})")
        
    # 2. Manual Brute Force (All 100)
    # Reconstruct all vectors (FAISS IndexFlatIP stores vectors directly)
    # We can just iterate and dot product if we validatd FAISS.
    # But let's trust FAISS returns generally correct top N.
    # However, I want to see the score of *specific* white cards if they aren't in Top 5.
    
    # Extract all vectors from index
    all_vectors = index.reconstruct_n(0, index.ntotal)
    
    # Dot product
    # vector is (1, 768), all_vectors is (100, 768)
    scores = np.dot(all_vectors, vector.T).flatten() # (100,)
    
    # Sort
    scores = np.dot(all_vectors, vector.T).flatten()
    sorted_indices = np.argsort(scores)[::-1]
    
    print("\n--- ASSISTED MODE (Land Ban) ---")
    BANNED_CARDS = {"Island", "Forest", "Mountain", "Swamp", "Plains"}
    
    rank = 0
    count = 0
    while count < 5 and rank < len(sorted_indices):
        idx = sorted_indices[rank]
        score = scores[idx]
        card = index_map[idx]
        name = card['name']
        
        rank += 1
        
        if name in BANNED_CARDS:
            continue
            
        print(f"#{count+1}: {score:.4f} - {name} ({card['set']})")
        count += 1

if __name__ == "__main__":
    main()
