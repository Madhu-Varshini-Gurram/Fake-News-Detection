import os
import subprocess
import sys

def main():
    print("🚀 Initializing Fake News Detection Project...")
    
    # Check dependencies (Optional but good practice)
    # print("🔍 Checking dependencies...")
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # 1. Download Data
    print("\n--- Step 1: Downloading Dataset ---")
    subprocess.run([sys.executable, "data/download_data.py"])
    
    # 2. Train Model
    print("\n--- Step 2: Training NLP Model ---")
    subprocess.run([sys.executable, "train_pipeline.py"])
    
    print("\n🎉 Setup Complete! Accuracy and metrics saved to console above.")
    print("\n--- Step 3: Launching Dashboard ---")
    print("The Streamlit app will now open in your browser...")
    
    try:
        # Launch streamlit as a subprocess
        subprocess.run(["streamlit", "run", "app/app.py"])
    except KeyboardInterrupt:
        print("\n👋 App closed. Thank you for using Fake News Detection!")

if __name__ == "__main__":
    main()
