import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models
from sklearn.metrics import confusion_matrix
import itertools
import random
import os

# ── Seed 고정 ──────────────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
# ───────────────────────────────────────────────────────

# ── 데이터 로드 ────────────────────────────────────────
fashion_mnist = keras.datasets.fashion_mnist
(train_images_raw, train_labels), (test_images_raw, test_labels) = fashion_mnist.load_data()
train_images_raw = train_images_raw.reshape((60000, 28, 28, 1))
test_images_raw  = test_images_raw.reshape((10000, 28, 28, 1))

# 정규화 버전
train_norm = train_images_raw / 255.0
test_norm  = test_images_raw  / 255.0

# Z-score 버전
mean = train_images_raw.mean()
std  = train_images_raw.std()
train_zscore = (train_images_raw - mean) / std
test_zscore  = (test_images_raw  - mean) / std
# ───────────────────────────────────────────────────────

def build_model(name):
    m = models.Sequential(name=name)

    if name == 'baseline':
        m.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (3,3), activation='relu'))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (3,3), activation='relu'))

    elif name == 'no_normalize':
        m.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (3,3), activation='relu'))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (3,3), activation='relu'))

    elif name == 'zscore':
        m.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (3,3), activation='relu'))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (3,3), activation='relu'))

    elif name == 'epoch10':
        m.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (3,3), activation='relu'))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (3,3), activation='relu'))

    elif name == 'filter5_padding':
        m.add(layers.Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(28,28,1)))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (5,5), activation='relu', padding='same'))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (5,5), activation='relu', padding='same'))

    elif name == 'more_filters':
        m.add(layers.Conv2D(64,  (3,3), activation='relu', padding='same', input_shape=(28,28,1)))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))

    elif name == 'batchnorm':
        m.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)))
        m.add(layers.BatchNormalization())
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        m.add(layers.BatchNormalization())
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        m.add(layers.BatchNormalization())

    elif name == 'dropout':
        m.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Dropout(0.25))
        m.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        m.add(layers.MaxPooling2D((2,2)))
        m.add(layers.Dropout(0.25))
        m.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))

    m.add(layers.Flatten())
    m.add(layers.Dense(64, activation='relu'))
    m.add(layers.Dense(10, activation='softmax'))
    return m

# ── 실험 설정 ──────────────────────────────────────────
experiments = [
    {'name': 'baseline',       'data': (train_norm,   test_norm),   'epochs': 10},
    {'name': 'no_normalize',   'data': (train_images_raw.astype('float32'), test_images_raw.astype('float32')), 'epochs': 10},
    {'name': 'zscore',         'data': (train_zscore, test_zscore), 'epochs': 10},
    {'name': 'epoch20',        'data': (train_norm,   test_norm),   'epochs': 10},
    {'name': 'filter5_padding','data': (train_norm,   test_norm),   'epochs': 10},
    {'name': 'more_filters',   'data': (train_norm,   test_norm),   'epochs': 10},
    {'name': 'batchnorm',      'data': (train_norm,   test_norm),   'epochs': 10},
    {'name': 'dropout',        'data': (train_norm,   test_norm),   'epochs': 10},
]

# ── 학습 및 결과 저장 ──────────────────────────────────
results = {}

for exp in experiments:
    print(f"\n{'='*50}")
    print(f"  실험: {exp['name']}")
    print(f"{'='*50}")

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = build_model(exp['name'])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    train_X, test_X = exp['data']
    history = model.fit(train_X, train_labels,
                        epochs=exp['epochs'],
                        validation_split=0.1,
                        verbose=1)

    test_loss, test_acc = model.evaluate(test_X, test_labels, verbose=0)

    results[exp['name']] = {
        'history'  : history,
        'test_loss': test_loss,
        'test_acc' : test_acc,
    }
    print(f"  → test_acc: {test_acc:.4f} | test_loss: {test_loss:.4f}")

# ── 결과 비교 출력 ─────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'모델':<20} {'Test Accuracy':>15} {'Test Loss':>12}")
print(f"{'='*60}")
for name, res in results.items():
    print(f"{name:<20} {res['test_acc']:>14.4f} {res['test_loss']:>12.4f}")
print(f"{'='*60}")

# ── 시각화 ─────────────────────────────────────────────
names     = list(results.keys())
accs      = [results[n]['test_acc']  for n in names]
losses    = [results[n]['test_loss'] for n in names]
colors    = plt.cm.tab10(np.linspace(0, 1, len(names)))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')

# Accuracy 막대 그래프
bars = axes[0].bar(names, accs, color=colors)
axes[0].set_title('Test Accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(min(accs) - 0.05, 1.0)
axes[0].tick_params(axis='x', rotation=30)
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=9)

# Loss 막대 그래프
bars2 = axes[1].bar(names, losses, color=colors)
axes[1].set_title('Test Loss')
axes[1].set_ylabel('Loss')
axes[1].set_ylim(0, max(losses) + 0.1)
axes[1].tick_params(axis='x', rotation=30)
for bar, loss in zip(bars2, losses):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f'{loss:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ── epoch별 val_accuracy 추이 ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Training History (val)', fontsize=14, fontweight='bold')

for i, (name, res) in enumerate(results.items()):
    h = res['history'].history
    ep = range(1, len(h['val_accuracy']) + 1)
    axes[0].plot(ep, h['val_accuracy'], label=name, color=colors[i], marker='o', markersize=3)
    axes[1].plot(ep, h['val_loss'],     label=name, color=colors[i], marker='o', markersize=3)

axes[0].set_title('Validation Accuracy per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Validation Loss per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()