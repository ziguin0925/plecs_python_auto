from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, Input, Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


tf.random.set_seed(777) # 하이퍼파라미터 튜닝을 위해 실행시 마다 변수가 같은 초기값 가지게 하기


def compile_model(model:Model, optimizer = optimizers.Adam):

    model.compile(
        optimizer = optimizer,
        loss='mse',
        metrics=['mae']
    )

    return model

def build_original_model_ripple():
    model_ripple = tf.keras.models.Sequential([
        layers.Input(shape=(4,)),

        layers.Dense(30), # wx+b
        layers.BatchNormalization(), # 정규화
        layers.LeakyReLU(alpha=0.01), # Relu, tanh

        layers.Dense(10),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),

        layers.Dense(2),  #  Iripple, Vripple,
    ])
    return compile_model(model_ripple, optimizers.Adam(1e-3))

def build_random_forest():
    return RandomForestRegressor(
        n_estimators=200, max_depth=None,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )

def build_residual_mlp_leakage(input_dim=4, output_dim=1):
    inputs = Input(shape=(input_dim,))
    
    # Block 1
    x = layers.Dense(30)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Residual Block 1
    residual = x
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # skip connection
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Residual Block 2
    residual = x
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # skip connection
    x = layers.LeakyReLU(alpha=0.01)(x)

    # Output
    outputs = layers.Dense(output_dim)(x)
    
    model = Model(inputs, outputs, name='residual_mlp_leakage')
    return compile_model(model, optimizers.Adam(1e-3))


def build_residual_mlp_ripple(input_dim=4, output_dim=2):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Block 1
    x = layers.Dense(128)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Residual Block 1
    residual = x # 현재의 x를 저장

    # 잔차 학습 구문
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)

    # 원본과 잔차를 더함(최종 결과)
    x = layers.Add()([x, residual])  # skip connection(잔차 학습)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Residual Block 2
    residual = x
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # skip connection
    x = layers.LeakyReLU(alpha=0.01)(x)

    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    # Output
    outputs = layers.Dense(output_dim)(x)
    
    model = Model(inputs, outputs, name='residual_mlp_ripple')
    return compile_model(model, optimizers.Adam(1e-3))


def build_residual_mlp_loss(input_dim=4, output_dim=2):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Block 1
    x = layers.Dense(30)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Residual Block 1
    residual = x # 현재의 x를 저장

    # 잔차 학습 구문
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)

    # 원본과 잔차를 더함(최종 결과)
    x = layers.Add()([x, residual])  # skip connection(잔차 학습)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Residual Block 2
    residual = x
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # skip connection
    x = layers.LeakyReLU(alpha=0.01)(x)

    # Output
    outputs = layers.Dense(output_dim)(x)
    
    model = Model(inputs, outputs, name='residual_mlp_ripple')
    return compile_model(model, optimizers.Adam(1e-3))


def build_residual_mlp_temp(input_dim=4, output_dim=1):
    inputs = Input(shape=(input_dim,))
    
    # Block 1
    x = layers.Dense(30)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Residual Block 1
    residual = x
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # skip connection
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Residual Block 2
    residual = x
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # skip connection
    x = layers.LeakyReLU(alpha=0.01)(x)

    # Output
    outputs = layers.Dense(output_dim)(x)
    
    model = Model(inputs, outputs, name='residual_mlp_leakage')
    return compile_model(model, optimizers.Adam(1e-3))


def build_residual(input_dim=4, output_dim=1, model_name="residual_model"):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Block 1
    x = layers.Dense(30)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Residual Block 1
    residual = x # 현재의 x를 저장

    # 잔차 학습 구문
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)

    # 원본과 잔차를 더함(최종 결과)
    x = layers.Add()([x, residual])  # skip connection(잔차 학습)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    # Residual Block 2
    residual = x
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # skip connection
    x = layers.LeakyReLU(alpha=0.01)(x)

    # Output
    outputs = layers.Dense(output_dim)(x)
    
    model = Model(inputs, outputs, name=model_name)
    return compile_model(model, optimizers.Adam(1e-3))


def build_conduction_mode_classifier(input_dim=4):
    inputs = tf.keras.Input(shape=(input_dim,))

    x = layers.Dense(64)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    # Residual Block
    residual = x
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.LeakyReLU(alpha=0.01)(x)

    # 단일 출력 레이어 (2개 클래스)
    out_mode = layers.Dense(2, activation='sigmoid', name='mode')(x)

    model = Model(inputs, out_mode, name='conduction_mode_classifier')
    return compile_model(model, optimizers.Adam(1e-3))

def split_data(*arrays, train_ratio=0.70, val_ratio=0.15, random_state=42):
    """
    여러 배열을 동시에 train/val/test로 분할
    
    Parameters:
        *arrays: 분할할 배열들 (X, y1, y2, ...)
        train_ratio: 학습 비율 (default 0.70)
        val_ratio:   검증 비율 (default 0.15)
        random_state: 랜덤 시드 (default 42)
    
    Returns:
        각 배열의 (train, val, test) 튜플 리스트
    """
    test_ratio = 1 - train_ratio - val_ratio  # 0.15

    temp_split = train_test_split(*arrays, test_size=1 - train_ratio, random_state=random_state)

    # train_test_split 반환: [X_train, X_test, y1_train, y1_test, y2_train, y2_test]
    trains = temp_split[0::2]
    temps  = temp_split[1::2]

    # temp를 val / test로 분할 (15 / 15)
    val_test_split = train_test_split(*temps, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_state)

    vals  = val_test_split[0::2]  # val
    tests = val_test_split[1::2]  # test

    return list(zip(trains, vals, tests))


def scale_data(train, val, test, scaler=None):
    """
    train 기준으로 fit 후 val, test는 transform만 적용
    
    Parameters:
        train, val, test : 분할된 배열
        scaler : 사용할 scaler (default: StandardScaler)
                 예) MinMaxScaler(), StandardScaler(), RobustScaler()
    
    Returns:
        train, val, test (변환된 배열), fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
    
    train = scaler.fit_transform(train)
    val   = scaler.transform(val)
    test  = scaler.transform(test)
    
    return train, val, test, scaler


def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test,
                      model_name='model', epochs=100, batch_size=100):
    
    early_stop = EarlyStopping(
        monitor='val_loss',   # 검증 성능 기준
        patience=10,          # 10 epoch 동안 개선 없으면 stop
        restore_best_weights=True  # 가장 좋았던 시점으로 되돌림 (중요🔥)
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,      # lr 절반으로 감소
        patience=5,
        min_lr=1e-6
    )

    checkpoint = ModelCheckpoint(
        f'{model_name}.h5',
        monitor='val_loss',
        save_best_only=True
    )
    # =========================
    callbacks = [early_stop, reduce_lr, checkpoint]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )

    def evaluate_to_dict(model, X, y):
        results = model.evaluate(X, y, verbose=0)
        return dict(zip(model.metrics_names, results))

    train_metrics = evaluate_to_dict(model, X_train, y_train)
    val_metrics   = evaluate_to_dict(model, X_val,   y_val)
    test_metrics  = evaluate_to_dict(model, X_test,  y_test)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")

    all_keys = train_metrics.keys()

    header = f"{'Metric':15} {'Train':>10} {'Val':>10} {'Test':>10}"
    print(header)
    print('-'*50)

    for key in all_keys:
        print(f"{key:15} "
              f"{train_metrics[key]:>10.4f} "
              f"{val_metrics[key]:>10.4f} "
              f"{test_metrics[key]:>10.4f}")

    print(f"{'='*50}")


    keys = [k for k in history.history.keys() if not k.startswith('val_')]
    n = len(keys)

    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for i, key in enumerate(keys):
        axes[i].plot(history.history[key], label='train')
        axes[i].plot(history.history.get(f'val_{key}', []), label='val')
        axes[i].set_title(key)
        axes[i].set_xlabel('Epoch')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    # plt.savefig(f'{model_name}_history.png', dpi=150, bbox_inches='tight')
    plt.show()

    return history



def train_and_evaluate_sklearn(model,
                              X_train, y_train,
                              X_val, y_val,
                              X_test, y_test,
                              model_name='model'):

    # =========================
    # Train (no epochs, no batch)
    model.fit(X_train, y_train)

    # =========================
    # Evaluate

    def evaluate_to_dict(model, X, y):
        y_pred = model.predict(X)
        return {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2':  r2_score(y, y_pred)
        }

    train_metrics = evaluate_to_dict(model, X_train, y_train)
    val_metrics   = evaluate_to_dict(model, X_val,   y_val)
    test_metrics  = evaluate_to_dict(model, X_test,  y_test)

    # =========================
    # Print (Keras 스타일 유지)
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")

    all_keys = train_metrics.keys()

    header = f"{'Metric':15} {'Train':>10} {'Val':>10} {'Test':>10}"
    print(header)
    print('-'*50)

    for key in all_keys:
        print(f"{key:15} "
              f"{train_metrics[key]:>10.4f} "
              f"{val_metrics[key]:>10.4f} "
              f"{test_metrics[key]:>10.4f}")

    print(f"{'='*50}")

    # =========================
    # Fake "history" (for compatibility)
    history = {
        'mse': [train_metrics['mse']],
        'val_mse': [val_metrics['mse']],
        'mae': [train_metrics['mae']],
        'val_mae': [val_metrics['mae']]
    }

    # =========================
    # Plot (1-step history)
    keys = [k for k in history.keys() if not k.startswith('val_')]
    n = len(keys)

    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for i, key in enumerate(keys):
        axes[i].plot(history[key], label='train', marker='o')
        axes[i].plot(history.get(f'val_{key}', []), label='val', marker='o')
        axes[i].set_title(key)
        axes[i].set_xlabel('Step (pseudo)')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

    return history