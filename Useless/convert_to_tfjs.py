#!/usr/bin/env python3
"""
Convert Keras models to TensorFlow.js format
Supports custom layers and handles the ISL models
"""

from custom_layers import TemporalAttentionLayer
import os
import sys
import json
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Add current directory to path for custom layers
sys.path.append(str(Path(__file__).parent))

# Import custom layers


def convert_keras_to_tfjs(model_path, output_dir, model_name):
    """Convert a Keras model to TensorFlow.js format"""
    print(f"Converting {model_name} from {model_path} to {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load the model with custom objects
        custom_objects = {
            'TemporalAttentionLayer': TemporalAttentionLayer
        }

        model = keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False
        )

        print(f"Model loaded successfully. Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")

        # Convert to TensorFlow.js format
        tfjs.converters.save_keras_model(
            model,
            output_dir,
            quantization_dtype=None,  # No quantization for now
            skip_op_check=False,
            strip_debug_ops=True
        )

        print(f"SUCCESS: {model_name} converted successfully to {output_dir}")
        return True

    except Exception as e:
        print(f"FAILED: Failed to convert {model_name}: {str(e)}")
        return False


def main():
    """Main conversion function"""
    print("Starting Keras to TensorFlow.js conversion...")

    # Check if tensorflowjs is installed
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("tensorflowjs not installed. Installing...")
        os.system("pip install tensorflowjs")
        import tensorflowjs as tfjs

    # Model configurations
    models = [
        {
            "name": "letters",
            "path": "models/isl_wcs_raw_aug_light_v2/best.keras",
            "output": "frontend/public/models/letters"
        },
        {
            "name": "lstm_phrases",
            "path": "models/isl_phrases_v3_lstm/best.keras",
            "output": "frontend/public/models/lstm"
        },
        {
            "name": "tcn_phrases",
            "path": "models/isl_phrases_v3_tcn/best.keras",
            "output": "frontend/public/models/tcn"
        }
    ]

    # Convert each model
    success_count = 0
    for model_config in models:
        if os.path.exists(model_config["path"]):
            success = convert_keras_to_tfjs(
                model_config["path"],
                model_config["output"],
                model_config["name"]
            )
            if success:
                success_count += 1
        else:
            print(f"WARNING: Model file not found: {model_config['path']}")

    print(
        f"\nConversion complete! {success_count}/{len(models)} models converted successfully.")

    # Copy labels files
    print("\nCopying labels files...")
    label_files = [
        ("models/isl_wcs_raw_aug_light_v2/labels.json",
         "frontend/public/models/letters/labels.json"),
        ("models/isl_phrases_v3_lstm/labels.json",
         "frontend/public/models/lstm/labels.json"),
        ("models/isl_phrases_v3_tcn/labels.json",
         "frontend/public/models/tcn/labels.json")
    ]

    for src, dst in label_files:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, 'r') as f:
                labels_data = json.load(f)
            with open(dst, 'w') as f:
                json.dump(labels_data, f, indent=2)
            print(f"SUCCESS: Copied {src} -> {dst}")
        else:
            print(f"WARNING: Labels file not found: {src}")


if __name__ == "__main__":
    main()
