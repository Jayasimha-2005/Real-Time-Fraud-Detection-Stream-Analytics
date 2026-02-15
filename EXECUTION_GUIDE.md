# 🚀 Project Execution Guide

## Complete Step-by-Step Implementation Plan

This guide walks you through executing the entire Real-Time Fraud Detection project from start to finish.

---

## 📋 Table of Contents

1. [Setup Environment](#1-setup-environment)
2. [Download Dataset](#2-download-dataset)
3. [Phase 1: Data Preparation](#3-phase-1-data-preparation--baseline)
4. [Phase 2: Deep Learning Models](#4-phase-2-deep-learning-models)
5. [Phase 3-5: Autoencoder & Ensemble](#5-phase-3-5-autoencoder--ensemble--evaluation)
6. [Phase 4 Extension: Real-Time Streaming](#6-phase-4-extension-real-time-streaming)
7. [Results & Reporting](#7-results--reporting)

---

## 1. Setup Environment

### Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```powershell
pip install -r requirements.txt
```

**Note:** If you encounter Windows long-path errors, enable long paths:
1. Run PowerShell as Administrator
2. Execute: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`
3. Restart and retry installation

---

## 2. Download Dataset

### Get the Credit Card Fraud Dataset

1. **Go to Kaggle:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. **Download** `creditcard.csv`
3. **Place it in:** `real-time-fraud-detection/data/raw/creditcard.csv`

```
real-time-fraud-detection/
└── data/
    └── raw/
        └── creditcard.csv  ← Place here
```

---

## 3. Phase 1: Data Preparation & Baseline

### Open Notebook 01

```powershell
cd real-time-fraud-detection
jupyter notebook
# Or use VS Code: Open notebooks/01_data_preprocessing.ipynb
```

### Execute All Cells

Run the notebook from top to bottom (`Cell → Run All`).

### What This Does:

✅ Loads `creditcard.csv` (284,807 transactions)  
✅ Analyzes class imbalance (~0.17% fraud)  
✅ Creates visualizations (distributions, correlations)  
✅ Normalizes features  
✅ Train/test split (80/20)  
✅ Trains baseline models:
   - Logistic Regression
   - Random Forest  
✅ Evaluates performance  
✅ Saves preprocessed data to `data/processed/`  
✅ Saves models to `models/saved_models/`

### Expected Time: **5-10 minutes**

### Key Outputs:

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `models/saved_models/logistic_regression.pkl`
- `models/saved_models/random_forest.pkl`

---

## 4. Phase 2: Deep Learning Models

### Open Notebook 02

```powershell
jupyter notebook notebooks/02_lstm_model.ipynb
```

### Execute All Cells

Run the notebook from top to bottom.

### What This Does:

✅ Loads preprocessed data  
✅ Reshapes data for LSTM/GRU (3D tensors)  
✅ Builds LSTM architecture (2 layers, dropout)  
✅ Trains LSTM with class weights  
✅ Builds GRU architecture  
✅ Trains GRU model  
✅ Compares LSTM vs GRU performance  
✅ Generates training curves  
✅ Saves models

### Expected Time: **15-30 minutes** (depending on hardware)

### Key Outputs:

- `models/saved_models/lstm_model.keras`
- `models/saved_models/gru_model.keras`

### Notes:

- Uses GPU if available (TensorFlow will automatically detect)
- Early stopping prevents overfitting
- Class weights handle imbalanced data

---

## 5. Phase 3-5: Autoencoder + Ensemble + Evaluation

### Open Notebook 04

```powershell
jupyter notebook notebooks/04_autoencoder_model.ipynb
```

### Execute All Cells

This is the **most comprehensive notebook** covering:

### Phase 3: Autoencoder

✅ Trains autoencoder on **normal transactions only**  
✅ Calculates reconstruction error  
✅ Sets anomaly threshold (95th percentile)  
✅ Evaluates anomaly detection performance

### Phase 4: Ensemble

✅ Loads all trained models (LSTM, GRU, Autoencoder)  
✅ Generates predictions from each model  
✅ Creates weighted ensemble:
   - LSTM: 35%
   - GRU: 35%
   - Autoencoder: 30%  
✅ Optimizes ensemble threshold using F1-score  
✅ Evaluates final ensemble performance

### Phase 5: Comprehensive Evaluation

✅ Compares **all 6 models**:
   - Logistic Regression
   - Random Forest
   - LSTM
   - GRU
   - Autoencoder
   - Ensemble  
✅ Generates comparison metrics  
✅ Creates ROC curves  
✅ Creates Precision-Recall curves  
✅ Measures inference latency  
✅ Saves all results and visualizations

### Expected Time: **20-40 minutes**

### Key Outputs:

- `models/saved_models/autoencoder_model.keras`
- `reports/model_comparison.csv`
- `reports/model_comparison.png`
- `reports/roc_pr_curves.png`
- `reports/latency_comparison.png`

---

## 6. Phase 4 Extension: Real-Time Streaming

### Prerequisites

1. **Install Kafka:** Download from https://kafka.apache.org/downloads
2. **Install Java:** JDK 8 or 11 required

### Step 1: Start Kafka Infrastructure

**Terminal 1 - Zookeeper:**
```powershell
cd C:\kafka
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
```

**Terminal 2 - Kafka Server:**
```powershell
cd C:\kafka
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

**Terminal 3 - Create Topic:**
```powershell
cd C:\kafka
.\bin\windows\kafka-topics.bat --create --topic fraud-transactions --bootstrap-server localhost:9092
```

### Step 2: Start Producer

**Terminal 4:**
```powershell
cd real-time-fraud-detection\streaming
python kafka_producer.py
```

### Step 3: Start Consumer

**Terminal 5:**
```powershell
cd real-time-fraud-detection\streaming
python spark_streaming.py
```

### What This Does:

✅ Producer streams test transactions to Kafka (~100/sec)  
✅ Spark consumer processes in real-time  
✅ Loads trained models for inference  
✅ Detects fraud with ensemble scoring  
✅ Displays results with latency metrics

### Expected Time: **Continuous (Ctrl+C to stop)**

### See: `streaming/README.md` for detailed streaming guide

---

## 7. Results & Reporting

### Generated Files

After completing all phases, you'll have:

**Preprocessed Data:**
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`

**Trained Models:**
- `models/saved_models/logistic_regression.pkl`
- `models/saved_models/random_forest.pkl`
- `models/saved_models/lstm_model.keras`
- `models/saved_models/gru_model.keras`
- `models/saved_models/autoencoder_model.keras`

**Reports & Visualizations:**
- `reports/model_comparison.csv`
- `reports/model_comparison.png`
- `reports/roc_pr_curves.png`
- `reports/latency_comparison.png`

### Performance Summary

**Expected Results:**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Reg | ~0.97 | ~0.05-0.10 | ~0.90 | ~0.10-0.15 |
| Random Forest | ~0.99 | ~0.80-0.90 | ~0.75-0.85 | ~0.80-0.85 |
| LSTM | ~0.99 | ~0.85-0.92 | ~0.80-0.88 | ~0.83-0.90 |
| GRU | ~0.99 | ~0.85-0.92 | ~0.80-0.88 | ~0.83-0.90 |
| Autoencoder | ~0.95-0.98 | ~0.10-0.30 | ~0.80-0.95 | ~0.20-0.45 |
| **Ensemble** | **~0.99** | **~0.88-0.95** | **~0.85-0.92** | **~0.87-0.93** |

**Latency:**
- Logistic Reg: ~0.1-0.5 ms
- Random Forest: ~1-3 ms
- LSTM: ~8-15 ms
- GRU: ~7-13 ms
- Autoencoder: ~2-5 ms
- Ensemble: ~15-25 ms

---

## 🎯 Success Checklist

- [ ] Environment setup complete
- [ ] Dataset downloaded and placed correctly
- [ ] Phase 1 notebook executed successfully
- [ ] Baseline models trained and saved
- [ ] Phase 2 notebook executed successfully
- [ ] LSTM and GRU models trained
- [ ] Phase 3-5 notebook executed successfully
- [ ] Autoencoder and ensemble complete
- [ ] All reports and visualizations generated
- [ ] (Optional) Kafka + Spark streaming tested

---

## 🐛 Common Issues & Solutions

### Issue: ImportError for tensorflow/keras
**Solution:**
```powershell
pip install --upgrade tensorflow keras
```

### Issue: Out of Memory during training
**Solution:** Reduce batch size in notebooks:
```python
batch_size = 128  # Instead of 256
```

### Issue: Kafka connection refused
**Solution:** Ensure Kafka is running:
```powershell
# Check if Kafka process is running
netstat -an | findstr 9092
```

### Issue: Jupyter kernel crashes
**Solution:** Restart kernel and run cells individually

---

## 📊 Next Steps

1. **Create PowerPoint Presentation:**
   - Use generated visualizations from `reports/`
   - Include architecture diagram
   - Show performance comparison

2. **Write Abstract/Paper:**
   - Use results from `reports/model_comparison.csv`
   - Document methodology
   - Discuss limitations

3. **Deploy to Production:**
   - Containerize with Docker
   - Deploy to AWS/Azure/GCP
   - Set up CI/CD pipeline

4. **Add Explainability:**
   - Integrate SHAP values
   - Add LIME explanations
   - Create feature importance analysis

---

## 📌 Important Notes

- **Dataset:** 284,807 transactions, 492 frauds (0.172%)
- **Features:** V1-V28 (PCA), Time, Amount
- **Imbalance:** Use class weights and ensemble
- **Runtime:** Total ~1-2 hours for all notebooks
- **Hardware:** GPU recommended but not required

---

## 🎓 Learning Objectives Achieved

✅ Real-world imbalanced dataset handling  
✅ Baseline vs Deep Learning comparison  
✅ LSTM/GRU for sequence modeling  
✅ Autoencoder for anomaly detection  
✅ Ensemble methods for improved performance  
✅ Real-time streaming with Kafka + Spark  
✅ Production-ready evaluation metrics

---

## 📚 References

- Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- TensorFlow: https://www.tensorflow.org/
- Apache Kafka: https://kafka.apache.org/
- Apache Spark: https://spark.apache.org/

---

**Good luck with your project! 🚀**
