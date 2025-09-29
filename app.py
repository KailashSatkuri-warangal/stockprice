import streamlit as st
import joblib
import os
from src.train import train
from src.data_loader import load_raw_data
from src.features import create_features

st.title("📈 Google Stock Price Predictor")

option = st.selectbox("Choose Model", ["linear","rf"])

if st.button("Train Model"):
    model = train(option)
    st.success(f"{option} model trained successfully!")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{option}_model.pkl")

if st.button("Predict Next Day"):
    df = load_raw_data()
    df = create_features(df)
    latest_row = df[['MA5','MA10','Volatility','Volume']].iloc[-1:]

    if latest_row.empty:
        st.error("❌ Not enough data to predict.")
    else:
        model_path = f"models/{option}_model.pkl"
        if not os.path.exists(model_path):
            st.error("❌ Model not trained yet. Click 'Train Model' first.")
        else:
            model = joblib.load(model_path)
            pred = model.predict(latest_row)[0]
            st.metric("Next Day Predicted Close Price", f"${pred:.2f}")
