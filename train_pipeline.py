import pandas as pd
from src.preprocess import preprocess_df
from src.model import FakeNewsModel
from sklearn.model_selection import train_test_split
import os

# Define file paths
DATA_FILE = "data/train.csv"

def train_pipeline():
    if not os.path.exists(DATA_FILE):
        print("Dataset not found. Please run data/download_data.py first.")
        return
    
    # Load dataset
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Basic inspection
    print(f"Dataset has {len(df)} entries.")
    df = df.dropna(subset=['fake'])
    
    # Preprocess
    print("Pre-processing text...")
    df = preprocess_df(df, text_column='text')
    
    # Select features (label for this dataset is "fake": 1=Fake, 0=Real)
    X = df['cleaned']
    y = df['fake']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train
    model = FakeNewsModel()
    acc = model.train(X_train, X_test, y_train, y_test)
    
    print(f"Training pipeline completed. Final Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    train_pipeline()
    print("Model trained and saved to models/ directory.")
