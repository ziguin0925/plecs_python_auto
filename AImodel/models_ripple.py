"""
=============================================================
  Ripple Prediction — Model Zoo
  Output: [Iripple, Vripple]  (multi-output regression)
  Input : 4 features
=============================================================
"""

import numpy as np
import tensorflow as tf
from keras import layers, Model, Input, regularizers

# ── Scikit-learn ──────────────────────────────────────────
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

# ── Gradient Boosting ──────────────────────────────────────
try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_BOOST = True
except ImportError:
    HAS_BOOST = False
    print("[Warning] XGBoost / LightGBM not installed — skipping.")


# ══════════════════════════════════════════════════════════
#  1. ORIGINAL BASELINE  (your existing model)
# ══════════════════════════════════════════════════════════
def build_original_ripple():
    """Simple 2-hidden-layer MLP (your baseline)."""
    model = tf.keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(30),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),
        layers.Dense(10),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),
        layers.Dense(2),   # Iripple, Vripple
    ], name="Original_MLP")
    return model


# ══════════════════════════════════════════════════════════
#  2. RESIDUAL MLP  (your existing residual model)
# ══════════════════════════════════════════════════════════
def build_residual_mlp(input_dim=4, output_dim=2):
    """ResNet-style skip connections."""
    inputs = Input(shape=(input_dim,))

    x = layers.Dense(30)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    for _ in range(2):          # 2 residual blocks
        residual = x
        x = layers.Dense(30)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Dense(30)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.LeakyReLU(alpha=0.01)(x)

    outputs = layers.Dense(output_dim)(x)
    return Model(inputs, outputs, name="Residual_MLP")


# ══════════════════════════════════════════════════════════
#  3. WIDE & DEEP  MLP
# ══════════════════════════════════════════════════════════
def build_wide_deep(input_dim=4, output_dim=2):
    """Wide path (raw input) + Deep path concatenated at output."""
    inputs = Input(shape=(input_dim,))

    # Wide: direct connection
    wide = layers.Dense(16, activation="relu")(inputs)

    # Deep: stacked layers
    x = layers.Dense(64)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    merged = layers.Concatenate()([wide, x])
    outputs = layers.Dense(output_dim)(merged)
    return Model(inputs, outputs, name="Wide_and_Deep")


# ══════════════════════════════════════════════════════════
#  4. DEEP MLP with Dropout (과적합 방지)
# ══════════════════════════════════════════════════════════
def build_deep_dropout(input_dim=4, output_dim=2, dropout_rate=0.3):
    """Deeper MLP + Dropout regularization."""
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),
        layers.Dropout(dropout_rate),
        layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),
        layers.Dropout(dropout_rate),
        layers.Dense(32),
        layers.LeakyReLU(alpha=0.01),
        layers.Dense(output_dim),
    ], name="Deep_Dropout_MLP")
    return model


# ══════════════════════════════════════════════════════════
#  5. 1-D CNN (feature interaction 포착)
# ══════════════════════════════════════════════════════════
def build_1d_cnn(input_dim=4, output_dim=2):
    """Conv1D treats features as a sequence to capture local patterns."""
    inputs = Input(shape=(input_dim, 1))   # (batch, 4, 1)

    x = layers.Conv1D(32, kernel_size=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(64, kernel_size=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(output_dim)(x)

    return Model(inputs, outputs, name="Conv1D_MLP")


def preprocess_for_cnn(X):
    """Expand dims for Conv1D: (N,4) → (N,4,1)."""
    return np.expand_dims(X, axis=-1)


# ══════════════════════════════════════════════════════════
#  6. SELF-ATTENTION MLP (Feature Attention)
# ══════════════════════════════════════════════════════════
def build_attention_mlp(input_dim=4, output_dim=2):
    """
    Lightweight self-attention on feature dimension.
    Learns which of the 4 inputs matter most dynamically.
    """
    inputs = Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inputs)           # (batch, 4, 1)

    # Multi-head attention across 4 features
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=4)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)

    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    outputs = layers.Dense(output_dim)(x)

    return Model(inputs, outputs, name="Attention_MLP")


# ══════════════════════════════════════════════════════════
#  7. HIGHWAY NETWORK
# ══════════════════════════════════════════════════════════
class HighwayLayer(layers.Layer):
    """Gated skip connection: y = T*H(x) + (1-T)*x."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.H = layers.Dense(units, activation="relu")
        self.T = layers.Dense(units, activation="sigmoid",
                              bias_initializer=tf.constant_initializer(-1.0))

    def call(self, x):
        h = self.H(x)
        t = self.T(x)
        return t * h + (1.0 - t) * x


def build_highway_net(input_dim=4, output_dim=2, num_highway=4):
    inputs = Input(shape=(input_dim,))
    x = layers.Dense(30, activation="relu")(inputs)
    for _ in range(num_highway):
        x = HighwayLayer(30)(x)
    outputs = layers.Dense(output_dim)(x)
    return Model(inputs, outputs, name="Highway_Net")


# ══════════════════════════════════════════════════════════
#  8. MIXTURE OF EXPERTS (MoE) - lightweight
# ══════════════════════════════════════════════════════════
def build_moe(input_dim=4, output_dim=2, num_experts=4):
    """
    Gating network selects blend of expert sub-networks.
    Different experts may specialise on different operating regimes.
    """
    inputs = Input(shape=(input_dim,))

    # Gating network
    gate_logits = layers.Dense(num_experts, activation="softmax",
                               name="gate")(inputs)  # (batch, E)

    # Expert networks
    expert_outs = []
    for i in range(num_experts):
        e = layers.Dense(30, activation="relu", name=f"expert_{i}_l1")(inputs)
        e = layers.Dense(output_dim, name=f"expert_{i}_out")(e)   # (batch, 2)
        e = layers.Reshape((1, output_dim))(e)                     # (batch,1,2)
        expert_outs.append(e)

    experts = layers.Concatenate(axis=1)(expert_outs)              # (batch,E,2)

    # Weighted sum: gate (batch,E,1) * experts (batch,E,2)
    gate_exp = layers.Reshape((num_experts, 1))(gate_logits)
    weighted = layers.Multiply()([gate_exp, experts])              # (batch,E,2)
    outputs  = tf.reduce_sum(weighted, axis=1)                     # (batch,2)

    return Model(inputs, outputs, name="MoE_Net")


# ══════════════════════════════════════════════════════════
#  9-A. SVR (Support Vector Regression)
# ══════════════════════════════════════════════════════════
def build_svr():
    """MultiOutput SVR with RBF kernel."""
    return MultiOutputRegressor(
        SVR(kernel="rbf", C=10, epsilon=0.01, gamma="scale"),
        n_jobs=-1
    )


# ══════════════════════════════════════════════════════════
#  9-B. Ridge Regression
# ══════════════════════════════════════════════════════════
def build_ridge():
    return MultiOutputRegressor(Ridge(alpha=1.0))


# ══════════════════════════════════════════════════════════
#  9-C. Bayesian Ridge
# ══════════════════════════════════════════════════════════
def build_bayesian_ridge():
    return MultiOutputRegressor(BayesianRidge())


# ══════════════════════════════════════════════════════════
#  9-D. ElasticNet (L1 + L2)
# ══════════════════════════════════════════════════════════
def build_elasticnet():
    return MultiOutputRegressor(ElasticNet(alpha=0.01, l1_ratio=0.5))


# ══════════════════════════════════════════════════════════
#  9-E. Random Forest
# ══════════════════════════════════════════════════════════
def build_random_forest():
    return RandomForestRegressor(
        n_estimators=200, max_depth=None,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )


# ══════════════════════════════════════════════════════════
#  9-F. Gaussian Process Regression
# ══════════════════════════════════════════════════════════
def build_gpr():
    kernel = Matern(nu=1.5) + WhiteKernel(noise_level=1e-3)
    return MultiOutputRegressor(
        GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                 normalize_y=True),
        n_jobs=-1
    )


# ══════════════════════════════════════════════════════════
#  9-G. XGBoost  (requires: pip install xgboost)
# ══════════════════════════════════════════════════════════
def build_xgboost():
    if not HAS_BOOST:
        raise ImportError("Install xgboost: pip install xgboost")
    return MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0
        ),
        n_jobs=-1
    )


# ══════════════════════════════════════════════════════════
#  9-H. LightGBM  (requires: pip install lightgbm)
# ══════════════════════════════════════════════════════════
def build_lightgbm():
    if not HAS_BOOST:
        raise ImportError("Install lightgbm: pip install lightgbm")
    return MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1
        ),
        n_jobs=-1
    )


# ══════════════════════════════════════════════════════════
#  MODEL REGISTRY  — used by the pipeline
# ══════════════════════════════════════════════════════════
KERAS_MODELS = {
    "Original_MLP"    : build_original_ripple,
    "Residual_MLP"    : build_residual_mlp,
    "Wide_and_Deep"   : build_wide_deep,
    "Deep_Dropout"    : build_deep_dropout,
    "Conv1D_MLP"      : build_1d_cnn,
    "Attention_MLP"   : build_attention_mlp,
    "Highway_Net"     : build_highway_net,
    "MoE_Net"         : build_moe,
}

SKLEARN_MODELS = {
    "SVR"             : build_svr,
    "Ridge"           : build_ridge,
    "BayesianRidge"   : build_bayesian_ridge,
    "ElasticNet"      : build_elasticnet,
    "RandomForest"    : build_random_forest,
    "GaussianProcess" : build_gpr,
}

if HAS_BOOST:
    SKLEARN_MODELS["XGBoost"]   = build_xgboost
    SKLEARN_MODELS["LightGBM"]  = build_lightgbm
