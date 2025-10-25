import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Custom", name="wcs_fn")
def wcs_fn(t):
    """Wrist-center + scale invariance + presence flags.
    Input t: (B, 2, 21, 3) -> returns (centered, present_flags)."""
    EPS = 1e-6
    wrist = t[:, :, 0:1, :]                           # (B,2,1,3)
    centered = t - wrist                              # (B,2,21,3)
    dist = tf.norm(centered, axis=-1)                 # (B,2,21)
    span = tf.reduce_max(dist, axis=-1, keepdims=True)  # (B,2,1)
    span = tf.maximum(span, EPS)
    centered = centered / span[..., None]             # (B,2,21,3)
    present = tf.reduce_sum(tf.abs(t), axis=[2, 3])   # (B,2)
    present = tf.cast(tf.not_equal(present, 0.0), tf.float32)
    return centered, present


@tf.keras.utils.register_keras_serializable(package="Custom", name="pres_fn")
def pres_fn(t):
    """Presence flags only: (B, 2, 21, 3) -> (B, 2) float32 in {0,1}."""
    present = tf.reduce_sum(tf.abs(t), axis=[2, 3])   # (B,2)
    present = tf.cast(tf.not_equal(present, 0.0), tf.float32)
    return present


@tf.keras.utils.register_keras_serializable(package="Custom", name="lhand_fn")
def lhand_fn(z):
    """(B,2,21,3) -> (B,21,3) - left hand slice."""
    return z[:, 0, :, :]


@tf.keras.utils.register_keras_serializable(package="Custom", name="rhand_fn")
def rhand_fn(z):
    """(B,2,21,3) -> (B,21,3) - right hand slice."""
    return z[:, 1, :, :]


@tf.keras.utils.register_keras_serializable(package="Custom")
class TemporalAttentionLayer(layers.Layer):
    """Temporal attention over time. Input (B,T,D) -> Output (B,D)."""

    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.proj = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, activation=None)
        self.softmax = layers.Softmax(axis=1)

    def call(self, x, mask=None):
        s = self.proj(x)
        s = self.score(s)
        if mask is not None:
            mask_f = tf.cast(mask[:, :, None], dtype=s.dtype)
            s = s + (1.0 - mask_f) * tf.constant(-1e9, dtype=s.dtype)
        w = self.softmax(s)
        return tf.reduce_sum(x * w, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg
