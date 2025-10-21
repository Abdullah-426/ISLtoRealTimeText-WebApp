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
        self.W = layers.Dense(self.units, use_bias=True, name="attn_W")
        self.v = layers.Dense(1,     use_bias=False, name="attn_v")
        self._D = None

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected (B,T,D); got {input_shape}")
        self._D = int(input_shape[-1])
        super().build(input_shape)

    def call(self, x, training=None):
        scores = self.v(tf.nn.tanh(self.W(x)))  # (B,T,1)
        alpha = tf.nn.softmax(scores, axis=1)  # (B,T,1)
        context = tf.reduce_sum(alpha * x, axis=1)  # (B,D)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg