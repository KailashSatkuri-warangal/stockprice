import os
import shutil
import kagglehub
from dotenv import load_dotenv

load_dotenv()

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

dataset_name = "medharawat/google-stock-price"
save_path = "data/raw"
os.makedirs(save_path, exist_ok=True)

csv_file_path = os.path.join(save_path, "GOOG.csv")

if os.path.exists(csv_file_path):
    print("✅ Dataset already exists at:", csv_file_path)
else:
    downloaded_folder = kagglehub.dataset_download(dataset_name)  # Already extracted
    print("✅ Dataset downloaded at:", downloaded_folder)

    # Move CSV to our raw folder
    for fname in os.listdir(downloaded_folder):
        if fname.endswith(".csv"):
            shutil.move(os.path.join(downloaded_folder, fname), csv_file_path)
    print("✅ Dataset ready at:", csv_file_path)
