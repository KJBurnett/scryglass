import requests
import json
import os

# Setup headers for polite access
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://moxfield.com/',
    'Origin': 'https://moxfield.com'
}

def probe_archidekt():
    url = "https://archidekt.com/api/decks/6821707/"
    print(f"\n--- Probing Archidekt: {url} ---")
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            print("Root Keys:", list(data.keys()))
            if 'cards' in data:
                print(f"Card Count: {len(data['cards'])}")
                if data['cards']:
                    print("Sample Card Keys:", list(data['cards'][0].keys()))
                    # print("Sample Card Data:", json.dumps(data['cards'][0], indent=2))
                    if 'card' in data['cards'][0]:
                         print("Sample Inner Card Keys:", list(data['cards'][0]['card'].keys()))
            else:
                print("WARNING: 'cards' key not found in root.")
        else:
            print(f"Failed: {resp.status_code} - {resp.text[:100]}")
    except Exception as e:
        print(f"Error: {e}")

def probe_moxfield():
    url = "https://api.moxfield.com/v2/decks/all/GAnCfVPj7EGXBf4ftLgn-A"
    print(f"\n--- Probing Moxfield: {url} ---")
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            print("Root Keys:", list(data.keys()))
            
            # Moxfield structure varies, check for mainboard/boards
            if 'mainboard' in data:
                print(f"Mainboard Count: {len(data['mainboard'])}")
                # Mainboard is a dict of card_id -> card_data
                first_key = list(data['mainboard'].keys())[0]
                print(f"Sample Card Key (ID): {first_key}")
                print("Sample Card Data:", json.dumps(data['mainboard'][first_key], indent=2))
            else:
                print("WARNING: 'mainboard' key not found.")
                
            if 'boards' in data:
                 print("Boards Keys:", list(data['boards'].keys()))

        else:
            print(f"Failed: {resp.status_code} - {resp.text[:100]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    probe_archidekt()
    probe_moxfield()
