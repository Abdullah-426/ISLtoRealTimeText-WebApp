#!/usr/bin/env python3
"""
Test V5 phrase endpoints with actual Table data from Dataset_Split
"""
import requests
import json
import numpy as np
import os


def test_with_table_data():
    base_url = "http://localhost:8001"

    # Load actual Table data from Dataset_Split
    table_path = "Phrase Model References/Dataset_Split/test/Table/clip_002/sequence.npy"
    if not os.path.exists(table_path):
        print(f"ERROR: Table data not found at {table_path}")
        return

    # Load the sequence data
    sequence_data = np.load(table_path)
    print(f"Loaded Table sequence: {sequence_data.shape}")
    print(f"Data type: {sequence_data.dtype}")
    print(
        f"Data range: [{np.min(sequence_data):.4f}, {np.max(sequence_data):.4f}]")

    # Convert to the format expected by the API
    frames = sequence_data.tolist()  # (48, 1662)
    times = [i * 0.05 for i in range(48)]  # 20 FPS timestamps
    hand_flags = [True] * 48  # Assume hands present for all frames

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

    print(f"\nSending payload:")
    print(f"  Frames: {len(frames)} x {len(frames[0])}")
    print(f"  Times: {len(times)}")
    print(f"  Hand flags: {len(hand_flags)}")
    print(f"  Add deltas: {test_payload['add_deltas']}")
    print(f"  TTA pool: {test_payload['tta_pool']}")

    # Test TCN endpoint
    print("\n=== Testing TCN endpoint ===")
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
            print(f"  Hand frames: {result.get('hand_frames', 0)}")
            print(f"  Top 3 predictions:")
            for i, (label, prob) in enumerate(result.get('top3', [])):
                print(f"    {i+1}. {label}: {prob:.3f}")
        else:
            print(f"FAILED: TCN endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"ERROR: TCN endpoint error: {e}")

    # Test LSTM endpoint
    print("\n=== Testing LSTM endpoint ===")
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
            print(f"  Hand frames: {result.get('hand_frames', 0)}")
            print(f"  Top 3 predictions:")
            for i, (label, prob) in enumerate(result.get('top3', [])):
                print(f"    {i+1}. {label}: {prob:.3f}")
        else:
            print(f"FAILED: LSTM endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"ERROR: LSTM endpoint error: {e}")

    # Test Ensemble endpoint
    print("\n=== Testing Ensemble endpoint ===")
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
            print(f"  Hand frames: {result.get('hand_frames', 0)}")
            print(f"  Top 3 predictions:")
            for i, (label, prob) in enumerate(result.get('top3', [])):
                print(f"    {i+1}. {label}: {prob:.3f}")
        else:
            print(f"FAILED: Ensemble endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"ERROR: Ensemble endpoint error: {e}")


if __name__ == "__main__":
    test_with_table_data()
