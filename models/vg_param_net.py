#!/usr/bin/env python3
"""
Van Genuchten Parameter Network (VGParamNet)
Variant A: Predicts alpha and n parameters, then uses analytical VG formula to compute
theta(psi).  Guarantees monotone SWCC by construction.

Parameter bounds (physically motivated, Carsel & Parrish 1988; Schaap et al. 2001):
    alpha : [0.001, 1.0]  1/kPa   (typical mineral soils)
    n     : [1.10,  3.5]          (n < 1.1 not observed in mineral soils)

Parameterisation uses sigmoid instead of softplus+clamp so that gradients are
well-conditioned everywhere inside the feasible range and never saturate at
either boundary.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Physical parameter bounds ────────────────────────────────────────────────
ALPHA_MIN, ALPHA_MAX = 0.001, 1.0    # 1/kPa
N_MIN,     N_MAX     = 1.10,  3.5   # dimensionless

# Maximum realistic matric suction for psi50 clipping (prevents log explosion)
PSI50_MIN, PSI50_MAX = 0.01, 1.0e5  # kPa


@tf.function
def vg_se(psi, alpha, n):
    """
    Effective saturation Se(psi) for the van Genuchten (1980) model.

    Args:
        psi   : suction [B, P] in kPa  (must be >= 0)
        alpha : VG alpha [B] in 1/kPa
        n     : VG n [B] (> 1)

    Returns:
        Se : effective saturation [B, P] clipped to [1e-8, 1-1e-8]
    """
    alpha = tf.maximum(alpha, 1e-8)
    n     = tf.maximum(n,     1.001)
    m     = 1.0 - 1.0 / n

    alpha = tf.reshape(alpha, [-1, 1])
    n     = tf.reshape(n,     [-1, 1])
    m     = tf.reshape(m,     [-1, 1])

    psi_safe    = tf.maximum(psi, 1e-8)
    alpha_psi   = tf.clip_by_value(alpha * psi_safe, 1e-10, 1e4)
    alpha_psi_n = tf.clip_by_value(tf.pow(alpha_psi, n), 0.0, 1e6)
    Se          = tf.pow(1.0 + alpha_psi_n, -m)
    return tf.clip_by_value(Se, 1e-8, 1.0 - 1e-8)


@tf.function
def psi_at_Se(alpha, n, Se_target=0.5):
    """
    Inverse VG: suction at a given effective saturation.

    Output is clipped to [PSI50_MIN, PSI50_MAX] so that the log-space psi50
    loss cannot explode when n is near its minimum.

    Args:
        alpha     : [B] in 1/kPa
        n         : [B] (> 1)
        Se_target : scalar, e.g. 0.5 for psi50

    Returns:
        psi : [B] in kPa, clipped to [PSI50_MIN, PSI50_MAX]
    """
    alpha = tf.maximum(alpha, 1e-8)
    n     = tf.maximum(n,     1.001)
    m     = 1.0 - 1.0 / n
    Se    = tf.cast(Se_target, alpha.dtype)

    # VG inverse: psi = [(Se^(-1/m) - 1)^(1/n)] / alpha
    inner = tf.maximum(tf.pow(Se, -1.0 / m) - 1.0, 1e-10)
    psi   = tf.pow(inner, 1.0 / n) / alpha
    return tf.clip_by_value(psi, PSI50_MIN, PSI50_MAX)


@tf.function
def vg_theta(psi, theta_s, theta_r, alpha, n):
    """
    Van Genuchten theta(psi) = theta_r + (theta_s - theta_r) * Se(psi).

    Args:
        psi     : [B, P] in kPa
        theta_s : [B] saturated water content
        theta_r : [B] residual water content
        alpha   : [B] in 1/kPa
        n       : [B] dimensionless

    Returns:
        theta : [B, P] volumetric water content
    """
    Se      = vg_se(psi, alpha, n)
    theta_s = tf.reshape(theta_s, [-1, 1])
    theta_r = tf.reshape(theta_r, [-1, 1])
    return theta_r + (theta_s - theta_r) * Se


@tf.keras.utils.register_keras_serializable(package="VGParamNet")
class VGParamNet(keras.Model):
    """
    Predicts alpha and n from soil property features; theta_s and theta_r are
    supplied externally from measured data.

    Architecture:
        [B, D] -> MLP(hidden) -> Dense(2) -> sigmoid -> (alpha, n)

    Parameterisation (sigmoid-based, bounded):
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sigmoid(raw_alpha)
        n     = N_MIN     + (N_MAX     - N_MIN)     * sigmoid(raw_n)

    Sigmoid gives well-conditioned gradients everywhere in the feasible range,
    unlike softplus+clamp which saturates at the upper bound.
    """

    def __init__(self, soil_prop_dim, hidden=(128, 128), name="VGParamNet", **kwargs):
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
        self.out = layers.Dense(2, activation=None, name="raw_out")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"soil_prop_dim": self.soil_prop_dim, "hidden": self.hidden})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(
            soil_prop_dim=config.get("soil_prop_dim", 16),
            hidden=tuple(config.get("hidden", (128, 128))),
            name=config.get("name", "VGParamNet"),
        )

    def call(self, soil_props, training=None):
        """
        Args:
            soil_props : [B, D] normalised soil property features

        Returns:
            alpha : [B] in [ALPHA_MIN, ALPHA_MAX]  1/kPa
            n     : [B] in [N_MIN, N_MAX]
        """
        x   = self.mlp(soil_props, training=training)
        raw = self.out(x)

        # Sigmoid maps (-inf, +inf) -> (0, 1); then linearly scale to target range.
        # Gradient is sigmoid*(1-sigmoid) -- always > 0, no dead zones at bounds.
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * tf.sigmoid(raw[:, 0])
        n     = N_MIN     + (N_MAX     - N_MIN)     * tf.sigmoid(raw[:, 1])

        return alpha, n
