#!/usr/bin/env python3
"""
Comprehensive accuracy test for v5 phrase models.
Tests 5 clips from each class in Dataset_Split/test folder.
"""

import os
import json
import numpy as np
import requests
from pathlib import Path
from collections import defaultdict
import time


def load_test_data(test_dir):
    """Load test data from Dataset_Split/test directory."""
    test_data = {}

    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        clips = []
        for clip_name in os.listdir(class_path):
            clip_path = os.path.join(class_path, clip_name)
            if not os.path.isdir(clip_path):
                continue

            sequence_path = os.path.join(clip_path, "sequence.npy")
            if os.path.exists(sequence_path):
                try:
                    sequence = np.load(sequence_path)
                    if sequence.shape == (48, 1662):  # Valid sequence
                        clips.append({
                            'path': sequence_path,
                            'sequence': sequence,
                            'class': class_name,
                            'clip': clip_name
                        })
                except Exception as e:
                    print(f"Error loading {sequence_path}: {e}")
                    continue

        # Take first 5 clips from each class
        test_data[class_name] = clips[:5]
        print(
            f"Loaded {len(test_data[class_name])} clips for class '{class_name}'")

    return test_data


def test_model_endpoint(endpoint, test_data, model_name):
    """Test a specific model endpoint."""
    print(f"\n=== Testing {model_name} Model ===")

    results = {
        'correct': 0,
        'total': 0,
        'class_accuracy': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'predictions': []
    }

    for class_name, clips in test_data.items():
        print(f"\nTesting class: {class_name}")

        for clip_data in clips:
            sequence = clip_data['sequence']
            true_class = clip_data['class']

            # Prepare test data
            frames = sequence.tolist()
            times = [i * 0.05 for i in range(len(frames))]  # 20 FPS
            hand_flags = [True] * len(frames)

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

            try:
                response = requests.post(
                    f"http://localhost:8001{endpoint}", json=payload, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    predicted_class = result['top_label']
                    confidence = result['top_conf']

                    is_correct = predicted_class == true_class
                    results['correct'] += is_correct
                    results['total'] += 1
                    results['class_accuracy'][true_class]['correct'] += is_correct
                    results['class_accuracy'][true_class]['total'] += 1

                    results['predictions'].append({
                        'true_class': true_class,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'correct': is_correct,
                        'clip': clip_data['clip']
                    })

                    status = "OK" if is_correct else "NO"
                    print(
                        f"  {status} {clip_data['clip']}: {predicted_class} ({confidence:.3f})")

                else:
                    print(
                        f"  NO {clip_data['clip']}: HTTP {response.status_code}")
                    results['total'] += 1
                    results['class_accuracy'][true_class]['total'] += 1

            except Exception as e:
                print(f"  NO {clip_data['clip']}: Error - {e}")
                results['total'] += 1
                results['class_accuracy'][true_class]['total'] += 1

    return results


def print_accuracy_report(results, model_name):
    """Print detailed accuracy report."""
    print(f"\n{'='*60}")
    print(f"ACCURACY REPORT: {model_name}")
    print(f"{'='*60}")

    overall_accuracy = results['correct'] / \
        results['total'] * 100 if results['total'] > 0 else 0
    print(
        f"Overall Accuracy: {results['correct']}/{results['total']} ({overall_accuracy:.2f}%)")

    print(f"\nPer-Class Accuracy:")
    print(f"{'Class':<20} {'Correct':<8} {'Total':<6} {'Accuracy':<10}")
    print(f"{'-'*50}")

    class_accuracies = []
    for class_name, stats in results['class_accuracy'].items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total'] * 100
            class_accuracies.append(accuracy)
            print(
                f"{class_name:<20} {stats['correct']:<8} {stats['total']:<6} {accuracy:<10.2f}%")

    if class_accuracies:
        avg_class_accuracy = np.mean(class_accuracies)
        print(f"\nAverage Class Accuracy: {avg_class_accuracy:.2f}%")

    # Show some incorrect predictions
    incorrect = [p for p in results['predictions'] if not p['correct']]
    if incorrect:
        print(f"\nSample Incorrect Predictions:")
        for pred in incorrect[:10]:  # Show first 10
            print(
                f"  {pred['true_class']} → {pred['predicted_class']} ({pred['confidence']:.3f}) [{pred['clip']}]")


def main():
    print("Comprehensive Accuracy Test for V5 Phrase Models")
    print("=" * 60)

    # Load test data
    test_dir = "Phrase Model References/Dataset_Split/test"
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    print("Loading test data...")
    test_data = load_test_data(test_dir)

    total_classes = len(test_data)
    total_clips = sum(len(clips) for clips in test_data.values())
    print(f"\nLoaded {total_clips} clips from {total_classes} classes")

    # Test each model
    models = [
        ("/infer/phrase-v5/tcn", "TCN"),
        ("/infer/phrase-v5/lstm", "LSTM"),
        ("/infer/phrase-v5/ensemble", "Ensemble")
    ]

    all_results = {}

    for endpoint, model_name in models:
        results = test_model_endpoint(endpoint, test_data, model_name)
        all_results[model_name] = results
        print_accuracy_report(results, model_name)
        time.sleep(1)  # Brief pause between models

    # Summary comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Overall Acc':<12} {'Avg Class Acc':<15}")
    print(f"{'-'*45}")

    for model_name, results in all_results.items():
        overall_acc = results['correct'] / \
            results['total'] * 100 if results['total'] > 0 else 0

        class_accuracies = []
        for stats in results['class_accuracy'].values():
            if stats['total'] > 0:
                class_accuracies.append(
                    stats['correct'] / stats['total'] * 100)

        avg_class_acc = np.mean(class_accuracies) if class_accuracies else 0

        print(f"{model_name:<12} {overall_acc:<12.2f}% {avg_class_acc:<15.2f}%")


if __name__ == "__main__":
    main()
