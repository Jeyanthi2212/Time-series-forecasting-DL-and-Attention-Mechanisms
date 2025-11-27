# ts_forecast_transformer_functional.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # optional: disable oneDNN for numeric consistency

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input

# --- 1. Data generation: synthetic multivariate time-series ---
def generate_synthetic_multivariate(num_steps=2000, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start='2020-01-01', periods=num_steps, freq='D')
    df = pd.DataFrame(index=dates)
    trend = np.linspace(0, 1, num_steps)
    seasonality = 0.5 * np.sin(2 * np.pi * np.arange(num_steps) / 30)
    noise = 0.1 * np.random.randn(num_steps)
    df['series_main'] = trend + seasonality + noise
    df['exog'] = 0.3 * np.sin(2 * np.pi * np.arange(num_steps) / 90) + 0.05 * np.random.randn(num_steps)
    df['aux'] = 0.8 * df['series_main'].shift(5).fillna(method='bfill') + \
                0.2 * df['exog'] + 0.05 * np.random.randn(num_steps)
    return df

# --- 2. Preprocessing: scaling & windowing ---
def preprocess(df, input_len=60, output_len=14):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len, 0])  # target: series_main
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

# --- 3. Baseline LSTM model ---
def build_lstm(input_shape, output_len):
    inp = Input(shape=input_shape)
    x = layers.LSTM(64, activation='relu')(inp)
    out = layers.Dense(output_len)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 4. Transformer-based forecaster (functional API) ---
def build_transformer(input_shape, d_model=32, num_heads=4, ff_dim=64,
                      num_layers=2, dropout=0.1, output_len=14):
    inp = Input(shape=input_shape)  # shape = (input_len, num_features)
    x = layers.Dense(d_model)(inp)  # embed features into d_model-dim
    seq_len = input_shape[0]

    # Positional encoding via embedding
    pos = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=d_model)(pos)
    pos_embedding = tf.expand_dims(pos_embedding, axis=0)  # shape (1, seq_len, d_model)
    x = x + pos_embedding  # broadcast added to each batch

    for _ in range(num_layers):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn = layers.Dropout(dropout)(attn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ffn = layers.Dense(ff_dim, activation='relu')(x)
        ffn = layers.Dense(d_model)(ffn)
        ffn = layers.Dropout(dropout)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    x = layers.Flatten()(x)
    out = layers.Dense(output_len)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 5. Training + Evaluation function ---
def train_and_evaluate(X, y, model_fn, model_name="model",
                       test_split=0.2, epochs=30, batch_size=32):
    n = len(X)
    split = int(n * (1 - test_split))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = model_fn(X_train.shape[1:], y_train.shape[1]) if model_name == 'transformer' \
            else model_fn(X_train.shape[1:], y_train.shape[1])

    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.1,
              verbose=2)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    naive = np.repeat(X_test[:, -1, 0].reshape(-1, 1),
                      y_test.shape[1], axis=1)
    mase = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - naive))

    print(f"*** {model_name} results: MAE={mae:.4f}, RMSE={rmse:.4f}, MASE={mase:.4f} ***")
    return model, (y_test, y_pred)

# --- 6. Main execution ---
def main():
    df = generate_synthetic_multivariate(num_steps=2500)
    X, y, scaler = preprocess(df, input_len=60, output_len=14)
    print("Prepared data:", X.shape, y.shape)

    # baseline LSTM
    lstm_model, lstm_res = train_and_evaluate(X, y, build_lstm,
                                              model_name="lstm", epochs=30)

    # transformer
    trans_model, trans_res = train_and_evaluate(X, y, build_transformer,
                                                model_name="transformer", epochs=30)

    # sample plot for transformer's forecast vs actual
    y_true, y_pred = trans_res
    plt.figure(figsize=(10,4))
    plt.plot(y_true.flatten(), label='true')
    plt.plot(y_pred.flatten(), label='pred')
    plt.legend()
    plt.title('Transformer Forecast vs Actual (Test set)')
    plt.show()

if __name__ == '__main__':
    main()
