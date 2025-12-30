import json
import os
import time
import requests
from tqdm import tqdm

INPUT_FILE = "tidus_cards.json"
IMAGE_DIR = "images"

def main():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    
    print(f"Found {len(cards)} cards to process.")
    
    for card in tqdm(cards):
        card_id = card['id']
        image_url = card.get('image_uris', {}).get('large')
        
        if not image_url:
            print(f"Skipping {card.get('name')} (No image URL)")
            continue
            
        file_path = os.path.join(IMAGE_DIR, f"{card_id}.jpg")
        
        if os.path.exists(file_path):
            continue
            
        try:
            # Polite download
            time.sleep(0.1) 
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            else:
                print(f"Failed to download {card.get('name')}: Status {response.status_code}")
        except Exception as e:
            print(f"Error downloading {card.get('name')}: {e}")

if __name__ == "__main__":
    main()
