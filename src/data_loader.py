import os
import pandas as pd
import shutil
import kagglehub
from dotenv import load_dotenv

load_dotenv()

RAW_PATH = os.path.join("data", "raw", "GOOG.csv")
PROCESSED_PATH = os.path.join("data", "processed", "processed.csv")
DATASET_NAME = "medharawat/google-stock-price"

def ensure_dataset():
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    if os.path.exists(RAW_PATH):
        print("✅ Dataset already exists at:", RAW_PATH)
        return

    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

    downloaded_folder = kagglehub.dataset_download(DATASET_NAME)
    print("✅ Dataset downloaded at:", downloaded_folder)

    for fname in os.listdir(downloaded_folder):
        if fname.endswith(".csv"):
            shutil.move(os.path.join(downloaded_folder, fname), RAW_PATH)
    print("✅ Dataset ready at:", RAW_PATH)

def load_raw_data():
    ensure_dataset()
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError("❌ Dataset not found. Check Kaggle credentials or network.")
    df = pd.read_csv(RAW_PATH)
    if df.empty:
        raise ValueError("❌ Dataset is empty!")
    return df

def save_processed_data(df):
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Processed data saved at {PROCESSED_PATH}")
