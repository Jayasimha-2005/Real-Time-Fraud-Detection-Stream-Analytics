"""
Project Setup Verification Script
Checks if all files and directories are in place
"""

import os
import sys

def check_file(path, required=True):
    """Check if a file exists"""
    exists = os.path.exists(path)
    status = "✓" if exists else ("✗" if required else "○")
    req_text = "(required)" if required else "(generated after execution)"
    print(f"  {status} {path} {'' if exists else req_text}")
    return exists

def check_dir(path):
    """Check if a directory exists"""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {path}/")
    return exists

def main():
    print("=" * 70)
    print("  Real-Time Fraud Detection - Project Verification")
    print("=" * 70)
    print()
    
    base_dir = "real-time-fraud-detection"
    
    # Core files
    print("📄 Core Files:")
    core_files = [
        "README.md",
        "EXECUTION_GUIDE.md",
        "PROJECT_SUMMARY.md",
        ".gitignore",
        "requirements.txt"
    ]
    core_ok = all(check_file(f) for f in core_files)
    print()
    
    # Data directories
    print("📁 Data Directories:")
    data_ok = True
    data_ok &= check_dir(f"{base_dir}/data")
    data_ok &= check_dir(f"{base_dir}/data/raw")
    data_ok &= check_dir(f"{base_dir}/data/processed")
    print()
    
    # Dataset
    print("📊 Dataset (Download from Kaggle):")
    dataset_ok = check_file(f"{base_dir}/data/raw/creditcard.csv", required=False)
    if not dataset_ok:
        print("     → Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print()
    
    # Notebooks
    print("📓 Jupyter Notebooks:")
    notebooks_ok = True
    notebooks = [
        "01_data_preprocessing.ipynb",
        "02_lstm_model.ipynb",
        "03_gru_model.ipynb",
        "04_autoencoder_model.ipynb"
    ]
    for nb in notebooks:
        notebooks_ok &= check_file(f"{base_dir}/notebooks/{nb}")
    print()
    
    # Streaming
    print("🌊 Streaming Components:")
    streaming_ok = True
    streaming_ok &= check_file(f"{base_dir}/streaming/kafka_producer.py")
    streaming_ok &= check_file(f"{base_dir}/streaming/spark_streaming.py")
    streaming_ok &= check_file(f"{base_dir}/streaming/README.md")
    print()
    
    # Models directory
    print("📦 Models Directory:")
    models_ok = check_dir(f"{base_dir}/models")
    models_ok &= check_dir(f"{base_dir}/models/saved_models")
    print()
    
    # Reports directory
    print("📊 Reports Directory:")
    reports_ok = check_dir(f"{base_dir}/reports")
    print()
    
    # Python packages
    print("🐍 Python Packages:")
    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas (run: pip install pandas)")
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy (run: pip install numpy)")
    
    try:
        import sklearn
        print("  ✓ scikit-learn")
    except ImportError:
        print("  ✗ scikit-learn (run: pip install scikit-learn)")
    
    try:
        import tensorflow
        print("  ✓ tensorflow")
    except ImportError:
        print("  ✗ tensorflow (run: pip install tensorflow)")
    
    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib (run: pip install matplotlib)")
    
    try:
        import seaborn
        print("  ✓ seaborn")
    except ImportError:
        print("  ✗ seaborn (run: pip install seaborn)")
    
    print()
    
    # Summary
    print("=" * 70)
    print("📋 Summary:")
    print("=" * 70)
    
    if core_ok and data_ok and notebooks_ok and streaming_ok and models_ok and reports_ok:
        print("✅ Project structure is complete!")
        print()
        if not dataset_ok:
            print("⚠  Next step: Download creditcard.csv from Kaggle")
            print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        else:
            print("🚀 Ready to execute! Follow EXECUTION_GUIDE.md")
    else:
        print("⚠  Some files/directories are missing.")
        print("   Check the messages above for details.")
    
    print()
    print("📖 For detailed instructions, see: EXECUTION_GUIDE.md")
    print()

if __name__ == "__main__":
    main()
