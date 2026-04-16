"""
=============================================================
  Ripple Prediction — Training & Comparison Pipeline
  Output: [Iripple, Vripple]
=============================================================
  Usage:
      python train_compare.py
=============================================================
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from keras import callbacks

warnings.filterwarnings("ignore")

# ── Import model zoo ──────────────────────────────────────
from models_ripple import (
    KERAS_MODELS, SKLEARN_MODELS, preprocess_for_cnn
)

# ══════════════════════════════════════════════════════════
#  CONFIG — 여기만 수정하세요
# ══════════════════════════════════════════════════════════
EPOCHS      = 100
BATCH_SIZE  = 32
LR          = 1e-3
PATIENCE    = 20          # EarlyStopping patience
TEST_SIZE   = 0.2
VAL_SIZE    = 0.1
RANDOM_SEED = 42
OUTPUT_COLS = ["Iripple", "Vripple"]

# ── 실제 데이터로 교체하세요 ──────────────────────────────

df = pd.read_csv('/home/pcsl/Documents/plecs/sepic/plecs_python_auto/out/sepic_data/ripple_results.csv')
X_all = df[['L1', 'L2', 'C1', 'F']].values.astype(np.float32)
y_all  = df[['i_L1_ripple_rate', 'V_out_ripple_rate']].values.astype(np.float32)

np.random.seed(RANDOM_SEED)
# ══════════════════════════════════════════════════════════
#  DATA PREP
# ══════════════════════════════════════════════════════════
X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_SEED)

scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)
X_test_s  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train)
y_val_s   = scaler_y.transform(y_val)

print(f"Train: {X_train_s.shape}  Val: {X_val_s.shape}  Test: {X_test_s.shape}")


# ══════════════════════════════════════════════════════════
#  HELPER — metrics
# ══════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # Per-output
    mae_i = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_v = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    return {
        "MAE"      : round(mae, 5),
        "RMSE"     : round(rmse, 5),
        "R2"       : round(r2, 4),
        "MAE_Ir"   : round(mae_i, 5),
        "MAE_Vr"   : round(mae_v, 5),
    }


# ══════════════════════════════════════════════════════════
#  KERAS TRAINING
# ══════════════════════════════════════════════════════════
def train_keras(name, build_fn):
    print(f"\n{'─'*50}")
    print(f"  Training: {name}")

    is_cnn = "Conv1D" in name
    Xtr = preprocess_for_cnn(X_train_s) if is_cnn else X_train_s
    Xvl = preprocess_for_cnn(X_val_s)   if is_cnn else X_val_s
    Xte = preprocess_for_cnn(X_test_s)  if is_cnn else X_test_s

    model = build_fn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="mse",
        metrics=["mae"]
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=10, min_lr=1e-6),
    ]

    t0 = time.time()
    history = model.fit(
        Xtr, y_train_s,
        validation_data=(Xvl, y_val_s),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=cb, verbose=0
    )
    elapsed = time.time() - t0

    y_pred_s = model.predict(Xte, verbose=0)
    y_pred   = scaler_y.inverse_transform(y_pred_s)

    metrics = compute_metrics(y_test, y_pred, name)
    metrics["Train_sec"] = round(elapsed, 1)
    metrics["Epochs"]    = len(history.history["loss"])

    print(f"  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}"
          f"  R²={metrics['R2']:.4f}  [{elapsed:.1f}s / {metrics['Epochs']} ep]")
    return metrics, model, y_pred


# ══════════════════════════════════════════════════════════
#  SKLEARN TRAINING
# ══════════════════════════════════════════════════════════
def train_sklearn(name, build_fn):
    print(f"\n{'─'*50}")
    print(f"  Training: {name}")

    model = build_fn()
    t0 = time.time()
    model.fit(X_train_s, y_train_s)
    elapsed = time.time() - t0

    y_pred_s = model.predict(X_test_s)
    y_pred   = scaler_y.inverse_transform(y_pred_s)

    metrics = compute_metrics(y_test, y_pred, name)
    metrics["Train_sec"] = round(elapsed, 1)
    metrics["Epochs"]    = "—"

    print(f"  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}"
          f"  R²={metrics['R2']:.4f}  [{elapsed:.1f}s]")
    return metrics, model, y_pred


# ══════════════════════════════════════════════════════════
#  RUN ALL MODELS
# ══════════════════════════════════════════════════════════
results  = {}
models   = {}
all_pred = {}

print("\n" + "═"*50)
print("  KERAS / DEEP LEARNING MODELS")
print("═"*50)
for name, fn in KERAS_MODELS.items():
    try:
        m, mdl, pred = train_keras(name, fn)
        results[name]  = m
        models[name]   = mdl
        all_pred[name] = pred
    except Exception as e:
        print(f"  [SKIP] {name}: {e}")

print("\n" + "═"*50)
print("  SKLEARN / CLASSICAL MODELS")
print("═"*50)
for name, fn in SKLEARN_MODELS.items():
    try:
        m, mdl, pred = train_sklearn(name, fn)
        results[name]  = m
        models[name]   = mdl
        all_pred[name] = pred
    except Exception as e:
        print(f"  [SKIP] {name}: {e}")


# ══════════════════════════════════════════════════════════
#  RESULTS TABLE
# ══════════════════════════════════════════════════════════
df = pd.DataFrame(results).T
df = df.sort_values("MAE")
df.index.name = "Model"

print("\n\n" + "═"*70)
print("  FINAL COMPARISON  (sorted by MAE ↑ = better)")
print("═"*70)
print(df[["MAE", "RMSE", "R2", "MAE_Ir", "MAE_Vr", "Train_sec", "Epochs"]].to_string())
print("═"*70)
df.to_csv("results_comparison.csv")
print("\n  Saved → results_comparison.csv")


# ══════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════
MODEL_NAMES = list(df.index)
COLORS_DL   = "#4F8EF7"
COLORS_ML   = "#F76F4F"
KERAS_SET   = set(KERAS_MODELS.keys())

bar_colors = [COLORS_DL if n in KERAS_SET else COLORS_ML for n in MODEL_NAMES]

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Ripple Prediction — Model Comparison\n(Iripple & Vripple)",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

# ── (A) MAE bar chart ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
bars = ax1.barh(MODEL_NAMES, df["MAE"].astype(float), color=bar_colors, edgecolor="none", height=0.6)
ax1.set_xlabel("MAE (lower = better)", fontsize=11)
ax1.set_title("(A) Overall MAE — all models", fontsize=12, fontweight="bold")
ax1.axvline(df["MAE"].astype(float).min(), color="green", lw=1.5, linestyle="--", label="Best")
ax1.legend()
for b, v in zip(bars, df["MAE"].astype(float)):
    ax1.text(v + 0.0002, b.get_y() + b.get_height() / 2,
             f"{v:.4f}", va="center", fontsize=8)
# legend patch
import matplotlib.patches as mpatches
dl_patch = mpatches.Patch(color=COLORS_DL, label="Deep Learning")
ml_patch = mpatches.Patch(color=COLORS_ML, label="Classical ML")
ax1.legend(handles=[dl_patch, ml_patch], loc="lower right")

# ── (B) MAE per output ───────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
x_pos = np.arange(len(MODEL_NAMES))
width = 0.4
ax2.bar(x_pos - width/2, df["MAE_Ir"].astype(float), width, label="Iripple", color="#5BC0EB")
ax2.bar(x_pos + width/2, df["MAE_Vr"].astype(float), width, label="Vripple", color="#FDE74C")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(MODEL_NAMES, rotation=45, ha="right", fontsize=7)
ax2.set_title("(B) MAE per output", fontsize=12, fontweight="bold")
ax2.legend()
ax2.set_ylabel("MAE")

# ── (C) R² score ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.barh(MODEL_NAMES, df["R2"].astype(float), color=bar_colors, edgecolor="none", height=0.6)
ax3.axvline(1.0, color="green", lw=1, linestyle="--")
ax3.set_title("(C) R² Score (higher = better)", fontsize=12, fontweight="bold")
ax3.set_xlabel("R²")

# ── (D) RMSE ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.barh(MODEL_NAMES, df["RMSE"].astype(float), color=bar_colors, edgecolor="none", height=0.6)
ax4.set_title("(D) RMSE", fontsize=12, fontweight="bold")
ax4.set_xlabel("RMSE")

# ── (E) Training time ────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
train_times = [float(df.loc[n, "Train_sec"]) for n in MODEL_NAMES]
ax5.barh(MODEL_NAMES, train_times, color=bar_colors, edgecolor="none", height=0.6)
ax5.set_title("(E) Training Time (seconds)", fontsize=12, fontweight="bold")
ax5.set_xlabel("Seconds")

plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("  Saved → model_comparison.png")
plt.show()

print("\n  Done ✓")
