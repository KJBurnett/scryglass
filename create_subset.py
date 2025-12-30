import csv
import json
import ijson
import pandas as pd
import glob
import os
import sys
from decimal import Decimal

# Configuration
CSV_PATH = "tidus_unleashed.csv"
JSON_PATTERN = "default-cards-*.json"
OUTPUT_PATH = "tidus_cards.json"

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def get_target_uuids(csv_path):
    """
    Reads the CSV and extracts the Scryfall UUIDs.
    Assumes Column 13 (0-indexed) contains the UUID logic based on inspection.
    """
    print(f"Reading {csv_path}...")
    try:
        # Use pandas for robust CSV parsing
        df = pd.read_csv(csv_path, header=None)
        
        # Column 13 seems to be the UUID based on inspection:
        uuids = set(df[13].dropna().astype(str))
        
        print(f"Found {len(uuids)} unique UUIDs in CSV.")
        return uuids
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return set()

def find_json_file():
    files = glob.glob(JSON_PATTERN)
    if not files:
        print("No default-cards JSON file found.")
        sys.exit(1)
    return files[0]

def process_file(json_path, target_uuids, output_path):
    print(f"Processing {json_path}...")
    
    matches = []
    
    with open(json_path, 'rb') as f:
        parser = ijson.items(f, 'item')
        
        count = 0
        for card in parser:
            count += 1
            if count % 10000 == 0:
                print(f"Scanned {count} cards... Found {len(matches)} matches.", end='\r')
            
            if card.get('id') in target_uuids:
                image_uris = card.get('image_uris')
                if not image_uris and 'card_faces' in card:
                    if card['card_faces'] and 'image_uris' in card['card_faces'][0]:
                        image_uris = card['card_faces'][0]['image_uris']
                        card['image_uris'] = image_uris
                
                if image_uris and 'large' in image_uris:
                    matches.append(card)
                else:
                    print(f"\nWarning: Match found for {card.get('name')} but no large image URI.")

    print(f"\nFinished scanning. Total matches: {len(matches)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2, cls=DecimalEncoder)
    
    print(f"Saved subset to {output_path}")

def main():
    target_uuids = get_target_uuids(CSV_PATH)
    if not target_uuids:
        print("No targets found. Exiting.")
        return

    json_path = find_json_file()
    process_file(json_path, target_uuids, OUTPUT_PATH)

if __name__ == "__main__":
    main()
