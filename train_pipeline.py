import pandas as pd
from src.preprocess import preprocess_df
from src.model import FakeNewsModel
from sklearn.model_selection import train_test_split
import os

# Define file paths
DATA_FILE = "data/fake_news.csv"

def train_pipeline():
    if not os.path.exists(DATA_FILE):
        print("Dataset not found. Please run data/download_data.py first.")
        return
    
    # Load dataset
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Basic inspection (Dataset labels: 0 for reliable, 1 for fake)
    print(f"Dataset has {len(df)} entries.")
    
    # Preprocess
    print("Pre-processing text...")
    df = preprocess_df(df, text_column='text')
    
    # Select features (label for this dataset is "label")
    X = df['cleaned']
    y = df['label'] # Assuming 0 for real, 1 for fake
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train
    model = FakeNewsModel()
    acc = model.train(X_train, X_test, y_train, y_test)
    
    print(f"Training pipeline completed. Final Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    train_pipeline()
    print("Model trained and saved to models/ directory.")
