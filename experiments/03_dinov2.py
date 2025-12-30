"""
Experiment 3: DINOv2 Model Test

DINOv2 (Meta, 2023) is a self-supervised vision model that excels at 
understanding image structure and geometry. It doesn't need text - 
it matches based purely on visual features.

This script:
1. Loads DINOv2 via torch hub
2. Extracts features from all 100 card images
3. Tests matching against the problematic crop
"""

import torch
from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm
from torchvision import transforms

# Config
IMAGE_DIR = "images"
METADATA_FILE = "tidus_cards.json"
TEST_IMAGE = "experiments/target.png"

# DINOv2 Model - base version (can upgrade to large if needed)
MODEL_NAME = "dinov2_vitb14"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print(f"Loading DINOv2 {MODEL_NAME}...")
    model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
    model = model.to(device)
    model.eval()
    
    # DINOv2 standard preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        cards = json.load(f)
        card_db = {c['id']: c for c in cards}
    
    # Get images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    print(f"Indexing {len(image_files)} images...")
    
    index_map = []
    vectors = []
    
    for filename in tqdm(image_files):
        card_id = filename.replace('.jpg', '')
        card_data = card_db.get(card_id)
        if not card_data:
            continue
            
        file_path = os.path.join(IMAGE_DIR, filename)
        
        try:
            image = preprocess(Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model(image)
                features = features / features.norm(dim=-1, keepdim=True)
                
            vectors.append(features.cpu().numpy().astype('float32'))
            index_map.append({
                "name": card_data.get('name'),
                "set": card_data.get('set_name'),
                "colors": card_data.get('color_identity', [])
            })
        except Exception as e:
            print(f"Error: {filename}: {e}")
    
    vectors_np = np.vstack(vectors)
    print(f"Index built: {vectors_np.shape}")
    
    # Test on problematic image
    print(f"\n--- Testing on {TEST_IMAGE} ---")
    if not os.path.exists(TEST_IMAGE):
        debug_dirs = sorted([d for d in os.listdir("debug") if os.path.isdir(f"debug/{d}")], reverse=True)
        if debug_dirs:
            TEST_IMAGE_ALT = f"debug/{debug_dirs[0]}/crop_enhanced.jpg"
            if os.path.exists(TEST_IMAGE_ALT):
                print(f"Using fallback: {TEST_IMAGE_ALT}")
                test_img = Image.open(TEST_IMAGE_ALT).convert("RGB")
            else:
                print("No test image found!")
                return
        else:
            print("No debug directories found!")
            return
    else:
        test_img = Image.open(TEST_IMAGE).convert("RGB")
    
    test_input = preprocess(test_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        test_features = model(test_input)
        test_features = test_features / test_features.norm(dim=-1, keepdim=True)
        
    test_vector = test_features.cpu().numpy().astype('float32')
    
    # Compute similarities
    scores = np.dot(vectors_np, test_vector.T).flatten()
    sorted_indices = np.argsort(scores)[::-1]
    
    print("\n=== DINOv2 TOP 10 RESULTS ===")
    for rank, idx in enumerate(sorted_indices[:10]):
        card = index_map[idx]
        score = scores[idx]
        colors = ''.join(card['colors']) if card['colors'] else 'C'
        print(f"#{rank+1}: {score:.4f} [{colors}] {card['name']} ({card['set']})")
    
    # Check if Rarity is in results
    print("\n--- Searching for 'Rarity' ---")
    for rank, idx in enumerate(sorted_indices):
        if "Rarity" in index_map[idx]['name']:
            print(f"FOUND: Rank #{rank+1} - Score {scores[idx]:.4f}")
            break
    else:
        print("Rarity NOT FOUND in database or results!")

if __name__ == "__main__":
    main()
