import torch
import clip
from PIL import Image
import faiss
import os
import json
import numpy as np
from tqdm import tqdm

IMAGE_DIR = "images"
METADATA_FILE = "tidus_cards.json"
INDEX_FILE = "scryglass.index"
MAP_FILE = "index_map.json"

# CLIP Model
MODEL_NAME = "ViT-B/32"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP {MODEL_NAME} on {device}...")
    
    # Load model
    model, preprocess = clip.load(MODEL_NAME, device=device)
    
    # Load Metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        cards = json.load(f)
        # Create dict for quick lookup
        card_db = {c['id']: c for c in cards}
    
    # Get List of Images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images to index.")
    
    index_map = []
    vectors = []
    
    print("Generating embeddings...")
    for filename in tqdm(image_files):
        card_id = filename.replace('.jpg', '')
        card_data = card_db.get(card_id)
        
        if not card_data:
            print(f"Warning: No metadata for {filename}")
            continue
            
        file_path = os.path.join(IMAGE_DIR, filename)
        
        try:
            image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image)
                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            # Move to CPU numpy
            vector = image_features.cpu().numpy().astype('float32')
            vectors.append(vector)
            
            # Add to map
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
    # CLIP dimension
    d = vectors[0].shape[1] 
    print(f"Vector dimension: {d}")
    
    # Stack vectors
    vectors_np = np.vstack(vectors)
    
    # IndexFlatIP for Cosine Similarity (vectors are normalized)
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(d)
    index.add(vectors_np)
    
    # Save
    faiss.write_index(index, INDEX_FILE)
    with open(MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_map, f, indent=2)
        
    print(f"Index built and saved to {INDEX_FILE}. size={index.ntotal}")

if __name__ == "__main__":
    main()
