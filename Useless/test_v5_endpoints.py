#!/usr/bin/env python3
"""
Test script for v5 phrase model endpoints
"""
import numpy as np
import requests
import json
import time
from pathlib import Path

# Backend URL
BASE_URL = "http://localhost:8001"


def test_v5_endpoints():
    """Test the v5 phrase model endpoints with sample data"""

    # Load sample data
    test_data_path = Path(
        "Phrase Model References/Dataset_Split/test/Hello/clip_000/sequence.npy")
    if not test_data_path.exists():
        print(f"Test data not found at {test_data_path}")
        return

    # Load the sequence data
    sequence = np.load(test_data_path)
    print(f"Loaded sequence shape: {sequence.shape}")
    print(f"Sequence dtype: {sequence.dtype}")

    # Convert to the format expected by the API
    frames = sequence.tolist()
    times = [i * 0.05 for i in range(len(frames))]  # 20 FPS = 0.05s per frame
    hand_flags = [True] * len(frames)  # Assume hands are present for test data

    # Test data payload
    payload = {
        "frames": frames,
        "times": times,
        "hand_flags": hand_flags,
        "frames_per_segment": 48,
        "target_fps": 20.0,
        "add_deltas": True,
        "tta_pool": "max",
        "do_shift": True,
        "do_warp": True,
        "do_hand_focus": True,
        "conf_hi": 0.72,
        "conf_lo": 0.55,
        "margin": 0.20,
        "min_hand_frames": 10
    }

    # Test endpoints
    endpoints = [
        ("/infer/phrase-v5/tcn", "TCN"),
        ("/infer/phrase-v5/lstm", "LSTM"),
        ("/infer/phrase-v5/ensemble", "Ensemble")
    ]

    for endpoint, model_name in endpoints:
        print(f"\n--- Testing {model_name} Model ---")
        try:
            # Add ensemble weights for ensemble endpoint
            if model_name == "Ensemble":
                payload["ens_w_tcn"] = 0.8
                payload["ens_w_lstm"] = 0.2

            response = requests.post(
                f"{BASE_URL}{endpoint}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print(f"SUCCESS {model_name}!")
                print(
                    f"   Top prediction: {result['top_label']} (confidence: {result['top_conf']:.4f})")
                print(f"   Commit OK: {result['commit_ok']}")
                print(
                    f"   Hand frames: {result['hand_frames']}/{result['total_frames']}")
                print(f"   Top 3: {result['top3'][:3]}")
            else:
                print(f"FAILED {model_name}: {response.status_code}")
                print(f"   Error: {response.text}")

        except Exception as e:
            print(f"ERROR {model_name}: {e}")

    # Test meta endpoint
    print(f"\n--- Testing Meta Endpoint ---")
    try:
        response = requests.get(f"{BASE_URL}/infer/meta")
        if response.status_code == 200:
            meta = response.json()
            print(f"SUCCESS Meta!")
            print(f"   V5 labels count: {len(meta.get('v5_labels', []))}")
            print(f"   Frame dim: {meta.get('frame_dim', 'N/A')}")
            print(
                f"   V5 frames per segment: {meta.get('v5_frames_per_segment', 'N/A')}")
        else:
            print(f"FAILED Meta: {response.status_code}")
    except Exception as e:
        print(f"ERROR Meta: {e}")


if __name__ == "__main__":
    print("Testing V5 Phrase Model Endpoints")
    print("=" * 50)
    test_v5_endpoints()
