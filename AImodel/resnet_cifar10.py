from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import keras
import tensorflow as tf
import pandas as pd
import os


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

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


def res_block(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def resnet_build(config, input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    if config["augmentation"] == True:  
        x = data_augmentation(inputs)

    x = layers.Conv2D(64, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for i, filters in enumerate(config["filters"]):
        for j in range(config["blocks_per_stage"]):

            # 첫 블록에서만 downsampling
            if i > 0 and j == 0:
                x = res_block(x, filters, stride=2)
            else:
                x = res_block(x, filters)

    if config["Pooling"] == True:
        x = layers.GlobalMaxPooling2D()(x)
    else : 
        x = layers.Flatten()(x)

    for units in config["dense_units"]:
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    return model


def run_experiment(config, x_train, y_train, x_test, y_test):
    model = resnet_build(config)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
    )
    early_stop = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        batch_size=64,
        callbacks=[lr_scheduler, early_stop],
        # callbacks=[lr_scheduler],
        verbose=1
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

    # print(cm)

    # 시각화 (정규화 포함)
    plot_confusion_matrix(cm, labels=labels, normalize=True)


    return {
        "config":           config,
        "best_val_accuracy":  best_val_acc,
        "best_train_accuracy": best_train_acc,
        "train_mse":          train_mse,
        "test_mse":           test_mse,
    }

config = {
    "epochs": 100,
    "learning_rate": 1e-3,
}

def save_result(result, param_grid, csv_file="resNet_results.csv"):
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

target = {
    "augmentation" : True,
    "filters": [64, 128, 256],
    "blocks_per_stage": 3,
    "dense_units": [256, 128, 64], 
    "Pooling" :True
}

param_grid = {
    "augmentation" : [True, False],
    "filters": [[64, 128, 256],[64, 128, 256, 512]],
    "blocks_per_stage": [2,3],
    "dense_units": [[128, 64], [256, 128, 64]], 
    "Pooling" :[True, False]
}


keys = param_grid.keys()
values = param_grid.values()
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
start_idx = experiments.index(target)
experiments = experiments[start_idx:start_idx+1]

results = run_all_experiments(experiments, param_grid, x_train, y_train, x_test, y_test)
