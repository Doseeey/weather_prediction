import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.callbacks import EarlyStopping


def make_prediction_lstm(criteria, time_span, debug_mode):
    # Wczytanie danych
    df = pd.read_csv('Wroclaw.csv', sep=';', parse_dates=['date'])

    for column in ["temp_pow", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr"]:
        df[column] = df[column].str.replace(",", ".").astype(float)

    df = df.dropna(subset=["temp_pow"])
    df = df.fillna(df.bfill())  # Zastąpienie deprecated backfill

    # Przygotowanie danych
    X = df[criteria]
    y = df["temp_pow"]

    X = X.fillna(X.bfill())  # Zastąpienie deprecated backfill

    # Zmiana podziału na trening/test (80%/20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Skalowanie danych
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)

    # Dopasowanie kształtu do LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Tworzenie modelu
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dropout(0.2),
        LSTM(64, activation='tanh'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Early stopping
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Trenowanie modelu
    model.fit(X_train_lstm, y_train, epochs=200, batch_size=64, callbacks=[early_stopping], verbose=debug_mode)

    # Przewidywanie wartości
    y_pred_all = model.predict(X_lstm).flatten()
    df["temp_pow_pred"] = y_pred_all

    # Zakres czasowy
    train_end_date = df['date'].iloc[len(X_train) - 1]
    prediction_end_date = train_end_date + pd.Timedelta(days=time_span)

    # Filtrowanie zakresu
    df_filtered = df[(df['date'] > train_end_date) & (df['date'] <= prediction_end_date)]

    # Zwracanie przewidywanych wartości
    return df_filtered['temp_pow_pred']


# Parametry funkcji
columns = ["kier_wiatr", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr", "wilg"]
time_span = 30
debug_mode = True

# Wywołanie funkcji
lstm_predictions = make_prediction_lstm(columns, time_span, debug_mode)
