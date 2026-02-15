# Real-Time Streaming Setup Guide

## Phase 4: Kafka + Spark Streaming

This guide explains how to set up and run the real-time fraud detection streaming pipeline.

---

## Prerequisites

### 1. Install Apache Kafka

**Download Kafka:**
```powershell
# Download from https://kafka.apache.org/downloads
# Extract to a directory (e.g., C:\kafka)
```

**Start Zookeeper:**
```powershell
cd C:\kafka
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
```

**Start Kafka Server (new terminal):**
```powershell
cd C:\kafka
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

**Create Topic:**
```powershell
.\bin\windows\kafka-topics.bat --create --topic fraud-transactions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

### 2. Install Apache Spark

```powershell
pip install pyspark
```

---

## Running the Streaming Pipeline

### Step 1: Train Models First

Before running the streaming pipeline, ensure you've trained the models:

```powershell
# Open and run these notebooks in order:
# 1. notebooks/01_data_preprocessing.ipynb
# 2. notebooks/02_lstm_model.ipynb
# 3. notebooks/04_autoencoder_model.ipynb
```

This will generate trained models in `models/saved_models/`.

### Step 2: Start Kafka Producer

In a terminal, navigate to the streaming directory:

```powershell
cd real-time-fraud-detection\streaming
python kafka_producer.py
```

This will:
- Load test data from `data/processed/X_test.csv`
- Stream transactions to Kafka topic `fraud-transactions`
- Send ~100 transactions per second

### Step 3: Start Spark Consumer

In a **new terminal**, start the Spark streaming consumer:

```powershell
cd real-time-fraud-detection\streaming
python spark_streaming.py
```

This will:
- Connect to Kafka topic
- Load trained models (LSTM, GRU, Autoencoder)
- Process transactions in real-time
- Display fraud detection results
- Measure inference latency

---

## Expected Output

### Producer Output:
```
====================================================
  KAFKA PRODUCER - Real-Time Fraud Detection Streaming
====================================================
✓ Connected to Kafka broker

📊 Loaded 57000 transactions from ../../data/processed/X_test.csv
🎯 Streaming to topic: 'fraud-transactions'
⏱ Delay between messages: 0.01s

✓ Sent 100 transactions (0 fraud detected)
✓ Sent 200 transactions (1 fraud detected)
...
```

### Consumer Output:
```
====================================================
  SPARK STREAMING - Real-Time Fraud Detection
====================================================
✓ Spark session created successfully

Loading fraud detection models...
✓ LSTM model loaded
✓ GRU model loaded
✓ Autoencoder model loaded

🎯 Real-time fraud detection started!
   Listening to Kafka topic: 'fraud-transactions'
   Press Ctrl+C to stop

====================================================
Processing Batch #0
====================================================
  🟢 NORMAL | ID: TXN_000000 | Score: 0.0234 | Latency: 12.34ms ✓
  🟢 NORMAL | ID: TXN_000050 | Score: 0.0189 | Latency: 11.89ms ✓
  🔴 FRAUD DETECTED | ID: TXN_000123 | Score: 0.8745 | Latency: 13.21ms ✓

📊 Batch Summary:
   Total: 100 | Correct: 98/100 (98.0%)
   Fraud Detected: 2 | Avg Latency: 12.56ms
```

---

## Configuration

### Producer Settings (kafka_producer.py):

```python
DATA_PATH = '../../data/processed/X_test.csv'  # Path to test data
TOPIC = 'fraud-transactions'                    # Kafka topic name
DELAY = 0.01                                     # 10ms = 100 tx/sec
```

### Consumer Settings (spark_streaming.py):

```python
# Model weights for ensemble
LSTM_WEIGHT = 0.35
GRU_WEIGHT = 0.35
AUTOENCODER_WEIGHT = 0.30

# Fraud detection threshold
THRESHOLD = 0.5
```

---

## Troubleshooting

### Issue: "NoBrokersAvailable"
**Solution:** Make sure Kafka server is running on localhost:9092

### Issue: "Models not found"
**Solution:** Train models first by running notebooks 01, 02, and 04

### Issue: "Spark can't find Kafka package"
**Solution:** Install Spark with Kafka support:
```powershell
pip install pyspark[kafka]
```

### Issue: "Java not found"
**Solution:** Install Java JDK 8 or 11:
```powershell
# Download from: https://www.oracle.com/java/technologies/downloads/
# Set JAVA_HOME environment variable
```

---

## Performance Metrics

- **Throughput:** ~100-1000 transactions/second
- **Latency:** ~10-20ms per transaction
- **Accuracy:** ~98% (depends on trained models)

---

## Stopping the Pipeline

1. Press `Ctrl+C` in the Spark consumer terminal
2. Press `Ctrl+C` in the Kafka producer terminal
3. Stop Kafka server: `Ctrl+C` in Kafka terminal
4. Stop Zookeeper: `Ctrl+C` in Zookeeper terminal

---

## Next Steps

- Tune model weights for better performance
- Add model explainability (SHAP/LIME)
- Deploy to cloud (AWS, Azure, GCP)
- Scale with Kafka clusters
- Implement model retraining pipeline
