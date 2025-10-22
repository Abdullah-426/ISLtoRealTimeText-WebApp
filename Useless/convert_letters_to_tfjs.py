#!/usr/bin/env python3
"""
Convert the letters MLP model to TF.js format
"""
import tensorflow as tf
import tensorflowjs as tfjs
import os
import sys
from pathlib import Path


def convert_letters_model():
    # Paths
    model_path = Path("models/isl_wcs_raw_aug_light_v2/best.keras")
    output_dir = Path("frontend/public/models/letters")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load the Keras model
        print(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(str(model_path))

        # Print model info
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Model summary:")
        model.summary()

        # Convert to TF.js format
        print(f"Converting to TF.js format...")
        tfjs.converters.save_keras_model(model, str(output_dir))

        print(f"✅ Successfully converted model to {output_dir}")
        return True

    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return False


if __name__ == "__main__":
    success = convert_letters_model()
    sys.exit(0 if success else 1)
