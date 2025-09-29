import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from .data_loader import load_raw_data, save_processed_data
from .features import create_features

def get_model(name):
    if name == "linear":
        return LinearRegression()
    elif name == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("❌ Unknown model: choose 'linear' or 'rf'")

def augment_dataset(df, min_rows=50):
    if len(df) >= min_rows:
        return df
    n_missing = min_rows - len(df)
    synthetic = pd.DataFrame({
        'Open': df['Open'].iloc[-1] + np.random.randn(n_missing),
        'High': df['High'].iloc[-1] + np.random.randn(n_missing),
        'Low': df['Low'].iloc[-1] + np.random.randn(n_missing),
        'Close': df['Close'].iloc[-1] + np.random.randn(n_missing),
        'Volume': df['Volume'].iloc[-1] + np.random.randint(-1000, 1000, n_missing)
    })
    df = pd.concat([df, synthetic], ignore_index=True)
    return df

def train(model_name="linear"):
    df = load_raw_data()
    df = create_features(df)
    df = augment_dataset(df, min_rows=50)

    X = df[['MA5','MA10','Volatility','Volume']]
    y = df['Close']

    if len(df) < 2:
        raise ValueError("❌ Not enough data to train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = get_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"✅ RMSE: {rmse:.2f}")

    save_processed_data(df)
    return model
