"""
Build FAISS index using DINOv2 embeddings.
Creates: scryglass_dino.index and index_map_dino.json
"""

import torch
from PIL import Image
import faiss
import os
import json
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# Constants
IMAGE_DIR = "images"
METADATA_FILE = "tidus_cards.json"
INDEX_FILE = "scryglass_dino.index"
MAP_FILE = "index_map_dino.json"
MODEL_NAME = "dinov2_vitb14"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINOv2 {MODEL_NAME} on {device}...")
    
    model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
    model = model.to(device)
    model.eval()
    
    # Standard DINOv2 preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load Metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        cards = json.load(f)
        card_db = {c['id']: c for c in cards}
    
    # Get images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images to index.")
    
    index_map = []
    vectors = []
    
    print("Generating DINOv2 embeddings...")
    for filename in tqdm(image_files):
        card_id = filename.replace('.jpg', '')
        card_data = card_db.get(card_id)
        
        if not card_data:
            print(f"Warning: No metadata for {filename}")
            continue
            
        file_path = os.path.join(IMAGE_DIR, filename)
        
        try:
            image = preprocess(Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model(image)
                features = features / features.norm(dim=-1, keepdim=True)
                
            vector = features.cpu().numpy().astype('float32')
            vectors.append(vector)
            
            index_map.append({
                "faiss_id": len(index_map),
                "scryfall_id": card_id,
                "name": card_data.get('name'),
                "set": card_data.get('set_name'),
                "image_path": file_path,
                "high_res_url": card_data.get('image_uris', {}).get('large'),
                "color_identity": card_data.get('color_identity', [])
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not vectors:
        print("No vectors generated. Exiting.")
        return

    # Build FAISS Index
    d = vectors[0].shape[1]  # DINOv2 vitb14 = 768
    print(f"Vector dimension: {d}")
    
    vectors_np = np.vstack(vectors)
    
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(d)
    index.add(vectors_np)
    
    # Save
    faiss.write_index(index, INDEX_FILE)
    with open(MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_map, f, indent=2)
        
    print(f"DINOv2 index built: {INDEX_FILE} (size={index.ntotal})")

if __name__ == "__main__":
    main()
