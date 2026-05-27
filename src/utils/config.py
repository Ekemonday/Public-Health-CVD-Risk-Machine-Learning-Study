"""
Config Loader
"""

import json
import os


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_config(config: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {path}")
