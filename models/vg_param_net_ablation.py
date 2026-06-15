#!/usr/bin/env python3
"""
VGParamNet Ablation Model
Predicts alpha, n, theta_s, theta_r from 14 inputs (theta_s and theta_r
excluded from the input features) to test whether measured boundary conditions
are necessary for accurate VG-parameter prediction.

Parameter bounds — same as VGParamNet for alpha and n; physically motivated
bounds for theta_s and theta_r with an enforced minimum gap to prevent
division-by-zero in the Se calculation.

    alpha   : [0.001, 1.0]  1/kPa
    n       : [1.10,  3.5]
    theta_s : [0.15,  0.65] (covers sandy loam to heavy clay)
    theta_r : [0.01,  0.15]
    gap     : theta_s - theta_r >= 0.05  (enforced after prediction)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.vg_param_net import vg_theta, ALPHA_MIN, ALPHA_MAX, N_MIN, N_MAX

# Theta bounds
THETA_S_MIN, THETA_S_MAX = 0.15, 0.65
THETA_R_MIN, THETA_R_MAX = 0.01, 0.15
MIN_THETA_GAP            = 0.05   # theta_s - theta_r must be >= this


@tf.keras.utils.register_keras_serializable(package="VGParamNetAblation")
class VGParamNetAblation(keras.Model):
    """
    Predicts alpha, n, theta_s, theta_r from 14 soil properties
    (theta_s and theta_r excluded from input).

    All four parameters use sigmoid-based bounded mapping for well-conditioned
    gradients.  A minimum gap constraint ensures theta_s - theta_r >= 0.05 so
    that Se = (theta - theta_r)/(theta_s - theta_r) is always numerically safe.
    """

    def __init__(self, soil_prop_dim=14, hidden=(128, 128),
                 name="VGParamNetAblation", **kwargs):
        super().__init__(name=name, **kwargs)
        self.soil_prop_dim = soil_prop_dim
        self.hidden        = hidden
        self.mlp = keras.Sequential(
            [
                layers.Dense(hidden[0], activation="relu", name="dense1"),
                layers.Dense(hidden[1], activation="relu", name="dense2"),
            ],
            name="mlp",
        )
        self.out = layers.Dense(4, activation=None, name="raw_out")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"soil_prop_dim": self.soil_prop_dim, "hidden": self.hidden})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(
            soil_prop_dim=config.get("soil_prop_dim", 14),
            hidden=tuple(config.get("hidden", (128, 128))),
            name=config.get("name", "VGParamNetAblation"),
        )

    def call(self, soil_props, training=None):
        """
        Args:
            soil_props : [B, 14] — 14 soil properties excluding theta_s/theta_r

        Returns:
            alpha   : [B] in [ALPHA_MIN, ALPHA_MAX]
            n       : [B] in [N_MIN, N_MAX]
            theta_s : [B] in [THETA_S_MIN, THETA_S_MAX]
            theta_r : [B] in [THETA_R_MIN, THETA_R_MAX], with theta_s - theta_r >= MIN_THETA_GAP
        """
        x   = self.mlp(soil_props, training=training)
        raw = self.out(x)

        # Sigmoid-based bounded mapping (consistent with VGParamNet)
        alpha   = ALPHA_MIN   + (ALPHA_MAX   - ALPHA_MIN)   * tf.sigmoid(raw[:, 0])
        n       = N_MIN       + (N_MAX       - N_MIN)       * tf.sigmoid(raw[:, 1])
        theta_s = THETA_S_MIN + (THETA_S_MAX - THETA_S_MIN) * tf.sigmoid(raw[:, 2])
        theta_r = THETA_R_MIN + (THETA_R_MAX - THETA_R_MIN) * tf.sigmoid(raw[:, 3])

        # Enforce minimum gap: if theta_s - theta_r < MIN_THETA_GAP, lift theta_s.
        # This is a soft correction that preserves gradients for both parameters.
        gap     = theta_s - theta_r
        deficit = tf.maximum(MIN_THETA_GAP - gap, 0.0)
        theta_s = theta_s + deficit

        return alpha, n, theta_s, theta_r
