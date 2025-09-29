import os
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

load_dotenv()

DATASET = "medharawat/google-stock-price"
RAW_PATH = "data/raw/GOOG.csv"

def set_kaggle_credentials():
	if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
		os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
		os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
	else:
		os.environ["KAGGLE_CONFIG_DIR"] = r"D:\7thsem\python_training\stockpricepredict"

def download_dataset(save_path="data/raw"):
	os.makedirs(save_path, exist_ok=True)
	set_kaggle_credentials()
	api = KaggleApi()
	api.authenticate()
	api.dataset_download_files(DATASET, path=save_path, unzip=True)

def load_or_download(file_path=RAW_PATH):
	if not os.path.exists(file_path):
		download_dataset(os.path.dirname(file_path))
	if not os.path.exists(file_path):
		raise FileNotFoundError("Download failed. Check Kaggle credentials.")
	return file_path
