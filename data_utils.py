import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_stock_data(symbol="GOOGL", start="2018-01-01", end="2025-07-21"):
    data = yf.download(symbol, start=start, end=end)[["Close"]]

    data["return"] = data['Close'].pct_change()
    data["return_1d_ago"] = data["return"].shift(1)
    data["return_2d_ago"] = data["return"].shift(2)
    data["return_3d_ago"] = data["return"].shift(3)

    data["Return_5d"] = data["Close"].shift(-1) / data["Close"] - 1
    data["Return_5d_suavizado"] = data["Return_5d"].rolling(window=3).mean()  ##Suavizamos la F objetivo
    data.dropna(inplace=True)

    print(data["Return_5d"].describe())
    print(data["Return_5d_suavizado"].describe())

    lower_threshold = np.percentile(data["Return_5d_suavizado"],33.33)
    upper_threshold = np.percentile(data["Return_5d_suavizado"],66.67)

    ##print(data)

    data["Action"] = 0

    ##print(data)

    data.loc[data["Return_5d_suavizado"] > upper_threshold, "Action"] = 1
    data.loc[data["Return_5d_suavizado"] < lower_threshold, "Action"] = -1

    ##print(data)

    return data

def prepare_sequences(data, seq_len=20, rtg_horizon=3):
    features = ["return", "return_1d_ago", "return_2d_ago", "return_3d_ago"]
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    X, A, R, RTG = [], [], [], []
    for i in range(len(data) - seq_len - rtg_horizon):
        window = data.iloc[i:i+seq_len]
        next_row = data.iloc[i+seq_len]

        # Recompensas futuras desde el punto actual hasta rtg_horizon
        future_returns = data["Return_5d"].iloc[i+seq_len : i+seq_len+rtg_horizon]
        return_to_go = future_returns.sum()

        X.append(window[features].values)
        A.append(next_row["Action"] + 1)  # Clase (0, 1, 2)
        R.append(next_row["Return_5d"])
        RTG.append(return_to_go)

    return np.array(X), np.array(A), np.array(R), np.array(RTG)
