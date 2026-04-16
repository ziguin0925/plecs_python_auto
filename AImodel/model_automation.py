import csv
import itertools
import numpy as np
import tensorflow as tf
from keras import layers
import keras
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix',
cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        # nlabel = '{0}(n={1})'.format(labels[k],n)
        nlabel = '{0}'.format(labels[k])
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)
    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] >thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# -------------------------------
# 1. 데이터 로드
# -------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

target = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "num_conv_blocks": 3,
    "filters": [32, 64, 128],
    "kernel_size": 5,
    "use_batchnorm": True,
    "dropout_rate": 0.3,
    "dense_units": [64, 32],
}

# -------------------------------
# 2. 하이퍼파라미터 공간 정의
# -------------------------------
param_grid = {
    "batch_size": [32, 64, 128],
    "epochs": [50, 100],
    "learning_rate": [1e-3, 5e-4],
    "num_conv_blocks": [1,2,3],
    "filters": [[32, 64, 128]],
    "kernel_size": [3, 5],
    "use_batchnorm": [True, False],
    "dropout_rate": [0.0, 0.3],
    "dense_units": [[32], [64, 32]], 
}


def build_model(config):
    # 일반 CNN모델 생성
    model = keras.Sequential()
    model.add(keras.Input(shape=(32,32,3)))

    # Conv Block
    for i in range(config["num_conv_blocks"]):
        filters = config["filters"][i]

        model.add(layers.Conv2D(filters,
                                (config["kernel_size"], config["kernel_size"]),
                                padding='same',
                                use_bias=False,
                                input_shape=None))

        if config["use_batchnorm"]:
            model.add(layers.BatchNormalization())

        model.add(layers.Activation('relu'))

        model.add(layers.MaxPooling2D()) # 기본 2,2
        model.add(layers.Dropout(config["dropout_rate"]))

    model.add(layers.GlobalAveragePooling2D())


    for units in config["dense_units"]:
        model.add(layers.Dense(units))
        model.add(layers.Dropout(config["dropout_rate"]))
        if config["use_batchnorm"]:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

    model.add(layers.Dense(10)) 

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model


def res_model_build() -> keras.Sequential :
    model = keras.Sequential()

    return model 




def run_experiment(config, x_train, y_train, x_test, y_test):
    model = build_model(config)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
    )
    early_stop = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[lr_scheduler, early_stop],
        verbose=0
    )


    best_val_acc   = max(history.history["val_accuracy"])
    best_train_acc = max(history.history["accuracy"])

    y_train_pred = model.predict(x_train, verbose=0)
    y_test_pred  = model.predict(x_test,  verbose=0)

    y_train_prob = tf.nn.softmax(y_train_pred).numpy()
    y_test_prob  = tf.nn.softmax(y_test_pred).numpy()

    y_train_onehot = np.eye(10)[y_train.flatten()]
    y_test_onehot  = np.eye(10)[y_test.flatten()]

    train_mse = float(np.mean((y_train_prob - y_train_onehot) ** 2))
    test_mse  = float(np.mean((y_test_prob  - y_test_onehot)  ** 2))

        # 예측
    predictions = model.predict(x_test)
    pred_labels = np.argmax(predictions, axis=1)

    # CIFAR-10 클래스 이름
    labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
    ]

    true_labels = y_test.flatten()

    # confusion matrix 계산
    cm = confusion_matrix(true_labels, pred_labels)

    print(cm)

    # 시각화 (정규화 포함)
    plot_confusion_matrix(cm, labels=labels, normalize=True)


    return {
        "config":           config,
        "best_val_accuracy":  best_val_acc,
        "best_train_accuracy": best_train_acc,
        "train_mse":          train_mse,
        "test_mse":           test_mse,
    }


# -------------------------------
# 6. 결과 저장 (실험 1개 단위)
# -------------------------------
def save_result(result, param_grid, csv_file="experiment_results.csv"):
    config = result["config"]

    row = {}
    for key in param_grid.keys():
        value = config[key]
        if isinstance(value, list):
            value = "_".join(map(str, value))
        row[key] = value

    row["best_val_accuracy"]   = result["best_val_accuracy"]
    row["best_train_accuracy"] = result["best_train_accuracy"]
    row["train_mse"]           = result["train_mse"]
    row["test_mse"]            = result["test_mse"]

    df_new = pd.DataFrame([row])

    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(csv_file, index=False)
    return len(df_combined)


# -------------------------------
# 7. 전체 실험 루프
# -------------------------------
def run_all_experiments(experiments, param_grid, x_train, y_train, x_test, y_test):
    results = []

    for i, config in enumerate(experiments):
        print(f"\n===== Experiment {i+1}/{len(experiments)} =====")
        print(config)

        result = run_experiment(config, x_train, y_train, x_test, y_test)
        results.append(result)

        total_rows = save_result(result, param_grid)
        print(f"[저장완료] Experiment {i+1} → experiment_results.csv (누적 {total_rows}행)")

    return results


# -------------------------------
# 8. 최고 모델 출력
# -------------------------------
def print_best_result(results):
    best_result = max(results, key=lambda x: x["best_val_accuracy"])
    print("\n===== BEST RESULT =====")
    print(best_result)

keys = param_grid.keys()
values = param_grid.values()

experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
start_idx = experiments.index(target)
experiments = experiments[start_idx:start_idx+1]

print(f"총 실험 수: {len(experiments)}")

# -------------------------------
# 실행
# -------------------------------
results = run_all_experiments(experiments, param_grid, x_train, y_train, x_test, y_test)
print_best_result(results)