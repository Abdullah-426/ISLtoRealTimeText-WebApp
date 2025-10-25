#!/usr/bin/env python3
"""
Quick test of V5 phrase endpoints - Fixed Unicode issues
"""
import requests
import json
import numpy as np


def test_v5_endpoints():
    base_url = "http://localhost:8001"

    # Test meta endpoint
    print("Testing meta endpoint...")
    try:
        response = requests.get(f"{base_url}/infer/meta")
        if response.status_code == 200:
            meta = response.json()
            print("SUCCESS: Meta endpoint working")
            print(f"  V5 labels count: {len(meta.get('v5_labels', []))}")
            print(
                f"  V5 frames per segment: {meta.get('v5_frames_per_segment', 'N/A')}")
        else:
            print(f"FAILED: Meta endpoint failed: {response.status_code}")
            return
    except Exception as e:
        print(f"ERROR: Meta endpoint error: {e}")
        return

    # Create test data
    frames = []
    times = []
    hand_flags = []

    # Generate 48 frames of test data (1662 dimensions each)
    for i in range(48):
        frame = np.random.randn(1662).astype(np.float32).tolist()
        frames.append(frame)
        times.append(i * 0.05)  # 20 FPS
        hand_flags.append(True)  # Always hands present for test

    test_payload = {
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

    # Test TCN endpoint
    print("\nTesting TCN endpoint...")
    try:
        response = requests.post(f"{base_url}/infer/phrase-v5/tcn",
                                 json=test_payload,
                                 headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: TCN endpoint working")
            print(f"  Top prediction: {result.get('top_label', 'N/A')}")
            print(f"  Confidence: {result.get('top_conf', 0):.3f}")
            print(f"  Commit OK: {result.get('commit_ok', False)}")
        else:
            print(f"FAILED: TCN endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"ERROR: TCN endpoint error: {e}")

    # Test LSTM endpoint
    print("\nTesting LSTM endpoint...")
    try:
        response = requests.post(f"{base_url}/infer/phrase-v5/lstm",
                                 json=test_payload,
                                 headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: LSTM endpoint working")
            print(f"  Top prediction: {result.get('top_label', 'N/A')}")
            print(f"  Confidence: {result.get('top_conf', 0):.3f}")
            print(f"  Commit OK: {result.get('commit_ok', False)}")
        else:
            print(f"FAILED: LSTM endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"ERROR: LSTM endpoint error: {e}")

    # Test Ensemble endpoint
    print("\nTesting Ensemble endpoint...")
    try:
        ensemble_payload = test_payload.copy()
        ensemble_payload["ens_w_tcn"] = 0.8
        ensemble_payload["ens_w_lstm"] = 0.2

        response = requests.post(f"{base_url}/infer/phrase-v5/ensemble",
                                 json=ensemble_payload,
                                 headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Ensemble endpoint working")
            print(f"  Top prediction: {result.get('top_label', 'N/A')}")
            print(f"  Confidence: {result.get('top_conf', 0):.3f}")
            print(f"  Commit OK: {result.get('commit_ok', False)}")
        else:
            print(f"FAILED: Ensemble endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"ERROR: Ensemble endpoint error: {e}")


if __name__ == "__main__":
    test_v5_endpoints()
