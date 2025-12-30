import json
import os

CONFIG_PATH = "config.json"

def load_config():
    default_config = {
        "device": "auto",
        "model_name": "ViT-B/32",
        "confidence_threshold": 0.32,
        "spatial_candidates": 15
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return {**default_config, **json.load(f)}
        except:
            return default_config
    return default_config

config = load_config()
CONFIDENCE_THRESHOLD = config["confidence_threshold"]
SPATIAL_CANDIDATES = config["spatial_candidates"]
