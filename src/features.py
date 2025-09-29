import pandas as pd

def create_features(df):
    for col in ['Open','High','Low','Close','Volume']:
        if col not in df.columns:
            raise ValueError(f"❌ Missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['MA10'] = df['Close'].rolling(10, min_periods=1).mean()
    df['Volatility'] = df['Close'].rolling(5, min_periods=1).std().fillna(0)
    df['Volume'] = df['Volume'].fillna(method='ffill').fillna(0)

    # Fill any remaining NaNs to avoid errors
    df[['MA5','MA10','Volatility','Volume']] = df[['MA5','MA10','Volatility','Volume']].fillna(0)
    return df
