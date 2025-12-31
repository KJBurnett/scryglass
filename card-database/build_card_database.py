import argparse
import json
import os
import time
import subprocess
import requests
import sys
from urllib.parse import urlparse

# Constants
MAPPINGS_FILE = os.path.join("card-mappings", "mappings.json")
PROCESSED_URLS_FILE = os.path.join("card-database", "processed_urls.json")
IMAGES_DIR = "images"
DEFAULT_CARDS_FILE = "tidus_cards.json"

# Headers for requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
}

def load_json(filepath, default=None):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {filepath} is corrupted. Starting fresh.")
    return default if default is not None else {}

def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def fetch_url_robust(url):
    """Try requests, fall back to curl if 403/401."""
    print(f"Fetching {url}...")
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code == 200:
            return resp.json()
        print(f"Requests failed with {resp.status_code}. Trying curl fallback...")
    except Exception as e:
        print(f"Requests failed with error {e}. Trying curl fallback...")
    
    # Curl fallback
    try:
        # Use curl to stdout
        result = subprocess.run(
            ['curl', '-s', '-H', f"User-Agent: {HEADERS['User-Agent']}", url],
            capture_output=True,
            text=True,
            encoding='utf-8' # Ensure utf-8 encoding
        )
        if result.returncode == 0 and result.stdout:
            return json.loads(result.stdout)
        else:
            print(f"Curl failed: {result.stderr}")
    except Exception as e:
        print(f"Curl fallback failed: {e}")
    
    return None

def parse_archidekt(deck_id):
    url = f"https://archidekt.com/api/decks/{deck_id}/"
    data = fetch_url_robust(url)
    cards = []
    if data and 'cards' in data:
        for item in data['cards']:
            card_data = item.get('card', {})
            if not card_data:
                continue
            
            # Archidekt structure
            uid = card_data.get('uid')
            if not uid: 
                continue

            # Construct minimal metadata needed
            cards.append({
                "id": uid,
                "name": card_data.get('oracleCard', {}).get('name', 'Unknown'),
                "set": card_data.get('edition', {}).get('editioncode'),
                "cn": card_data.get('collectorNumber'),
                # Construct likely image URL if not present, though we prefer fetching later or constructing scryfall URL
                # Archidekt api doesn't always give full image url, just hash. 
                # We will rely on Scryfall ID to fetch image or construct standard Scryfall URL.
                "image_uri": f"https://cards.scryfall.io/large/front/{uid[0]}/{uid[1]}/{uid}.jpg" # Standard scryfall pattern
            })
    return cards

def parse_moxfield(deck_id):
    url = f"https://api.moxfield.com/v2/decks/all/{deck_id}"
    data = fetch_url_robust(url)
    cards = []
    if data and 'mainboard' in data:
        for card_name, item in data['mainboard'].items():
            card_data = item.get('card', {})
            if not card_data:
                continue
            
            scryfall_id = card_data.get('scryfall_id')
            if not scryfall_id:
                continue
                
            cards.append({
                "id": scryfall_id,
                "name": card_data.get('name'),
                "set": card_data.get('set'),
                "cn": card_data.get('cn'),
                "image_uri": f"https://cards.scryfall.io/large/front/{scryfall_id[0]}/{scryfall_id[1]}/{scryfall_id}.jpg"
            })
    return cards

def download_image(url, save_path):
    if os.path.exists(save_path):
        return True # Already exists
        
    print(f"Downloading {url} to {save_path}")
    try:
        time.sleep(0.1) # Be polite
        resp = requests.get(url, stream=True, headers=HEADERS)
        if resp.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            print(f"Failed to download image: {resp.status_code}")
    except Exception as e:
        print(f"Error downloading image: {e}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Build Card Database from Deck URLs")
    parser.add_argument('--url', help='Single Deck URL (Archidekt or Moxfield)')
    parser.add_argument('--file', help='Text file with list of URLs')
    parser.add_argument('--clear-images', action='store_true', help='Delete all existing images')
    parser.add_argument('--clear-mappings', action='store_true', help='Clear existing mappings')
    
    args = parser.parse_args()
    
    # 0. Cleanup if requested
    if args.clear_images:
        print("Clearing images directory...")
        for f in os.listdir(IMAGES_DIR):
            if f.endswith(".jpg"):
                os.remove(os.path.join(IMAGES_DIR, f))
    
    if args.clear_mappings:
        print("Clearing mappings...")
        if os.path.exists(MAPPINGS_FILE):
            os.remove(MAPPINGS_FILE)
        if os.path.exists(PROCESSED_URLS_FILE):
             os.remove(PROCESSED_URLS_FILE)

    # 1. Gather URLs
    urls = []
    if args.url:
        urls.append(args.url)
    if args.file:
        if os.path.exists(args.file):
            with open(args.file, 'r') as f:
                urls.extend([line.strip() for line in f if line.strip()])
        else:
            print(f"File {args.file} not found.")
            return

    if not urls:
        print("No URLs provided.")
        return

    # 2. Load State
    processed_urls = load_json(PROCESSED_URLS_FILE, default=[])
    mappings = load_json(MAPPINGS_FILE, default=[])
    # Also load existing tidus_cards to check for existence (optional, but good for "duplicates" check if we wanted)
    # For now, we just ensure we append to mappings.
    
    new_urls_count = 0
    total_cards_processed = 0
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    for url in urls:
        if url in processed_urls and not args.clear_mappings:
            print(f"Skipping already processed: {url}")
            continue
            
        print(f"\nProcessing {url}...")
        cards = []
        
        # Identify Source
        if "archidekt.com" in url:
            # Extract ID: https://archidekt.com/decks/6821707/... -> 6821707
            try:
                path_parts = urlparse(url).path.split('/')
                # usually /decks/ID/...
                if 'decks' in path_parts:
                    idx = path_parts.index('decks')
                    deck_id = path_parts[idx+1]
                    cards = parse_archidekt(deck_id)
                else:
                    print(f"Could not parse Archidekt ID from {url}")
            except Exception as e:
                print(f"Error parsing Archidekt URL: {e}")
                
        elif "moxfield.com" in url:
            # Extract ID: https://moxfield.com/decks/ID
            try:
                path_parts = urlparse(url).path.split('/')
                deck_id = path_parts[-1] if path_parts[-1] else path_parts[-2]
                cards = parse_moxfield(deck_id)
            except Exception as e:
                print(f"Error parsing Moxfield URL: {e}")
        else:
            print(f"Unknown URL format: {url}")
            continue
            
        if not cards:
            print("No cards found or failed to fetch.")
            continue
            
        print(f"Found {len(cards)} cards.")
        
        # Process Cards
        for card in cards:
            card_id = card['id']
            # Add to mappings if not present
            existing = next((c for c in mappings if c['id'] == card_id), None)
            if not existing:
                mappings.append(card)
                
            # Download Image
            image_path = os.path.join(IMAGES_DIR, f"{card_id}.jpg")
            download_image(card['image_uri'], image_path)
            
        processed_urls.append(url)
        new_urls_count += 1
        total_cards_processed += len(cards)

    # 3. Save State
    save_json(MAPPINGS_FILE, mappings)
    save_json(PROCESSED_URLS_FILE, processed_urls)
    
    print(f"\nDone! Processed {new_urls_count} new decks. Total cards in mapping: {len(mappings)}.")

if __name__ == "__main__":
    main()
