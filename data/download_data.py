import pandas as pd
import requests
import os

DATA_URL = "https://raw.githubusercontent.com/raj1603chdry/Fake-News-Detection-System/master/datasets/train.csv"
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "fake_news.csv")

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Downloading dataset from {DATA_URL}...")
    response = requests.get(DATA_URL)
    if response.status_code == 200:
        with open(DATA_FILE, 'wb') as f:
            f.write(response.content)
        print(f"Dataset saved to {DATA_FILE}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    download_data()
