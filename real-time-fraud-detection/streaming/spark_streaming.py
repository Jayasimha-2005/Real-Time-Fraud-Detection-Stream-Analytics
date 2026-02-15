"""
Spark Streaming Consumer - Phase 4: Real-Time Fraud Detection
Consumes transactions from Kafka and performs real-time fraud detection
"""

import json
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, MapType
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

def create_spark_session():
    """Create Spark session with Kafka integration"""
    try:
        spark = SparkSession.builder \\
            .appName('RealTimeFraudDetection') \\
            .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \\
            .config('spark.sql.streaming.checkpointLocation', '/tmp/checkpoint') \\
            .getOrCreate()
        
        spark.sparkContext.setLogLevel('WARN')
        print("✓ Spark session created successfully")
        return spark
    except Exception as e:
        print(f"✗ Error creating Spark session: {e}")
        return None

def load_models():
    """Load trained fraud detection models"""
    models = {}
    try:
        models['lstm'] = keras.models.load_model('../../models/saved_models/lstm_model.keras')
        print("✓ LSTM model loaded")
    except:
        print("⚠ LSTM model not found")
    
    try:
        models['gru'] = keras.models.load_model('../../models/saved_models/gru_model.keras')
        print("✓ GRU model loaded")
    except:
        print("⚠ GRU model not found")
    
    try:
        models['autoencoder'] = keras.models.load_model('../../models/saved_models/autoencoder_model.keras')
        print("✓ Autoencoder model loaded")
    except:
        print("⚠ Autoencoder model not found")
    
    return models if models else None

def predict_fraud(features, models):
    """
    Perform fraud prediction using ensemble of models
    
    Args:
        features: Dictionary of transaction features
        models: Dictionary of loaded models
    
    Returns:
        tuple: (fraud_score, prediction, latency_ms)
    """
    start_time = time.time()
    
    try:
        # Convert features to numpy array
        feature_values = list(features.values())
        X = np.array(feature_values).reshape(1, -1)
        X_seq = X.reshape(1, 1, -1)  # For LSTM/GRU
        
        predictions = []
        
        # LSTM prediction
        if 'lstm' in models:
            lstm_pred = float(models['lstm'].predict(X_seq, verbose=0)[0][0])
            predictions.append(lstm_pred * 0.35)
        
        # GRU prediction
        if 'gru' in models:
            gru_pred = float(models['gru'].predict(X_seq, verbose=0)[0][0])
            predictions.append(gru_pred * 0.35)
        
        # Autoencoder prediction (reconstruction error)
        if 'autoencoder' in models:
            X_recon = models['autoencoder'].predict(X, verbose=0)
            recon_error = np.mean(np.square(X - X_recon))
            # Normalize to 0-1 range (assuming max error ~0.1)
            ae_score = min(recon_error / 0.1, 1.0)
            predictions.append(ae_score * 0.30)
        
        # Ensemble fraud score
        fraud_score = sum(predictions) if predictions else 0.0
        prediction = 1 if fraud_score > 0.5 else 0
        
        latency_ms = (time.time() - start_time) * 1000
        
        return fraud_score, prediction, latency_ms
        
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return 0.0, 0, 0.0

def process_stream(spark, models):
    """
    Process Kafka stream and perform real-time fraud detection
    
    Args:
        spark: Spark session
        models: Loaded ML models
    """
    # Define schema for incoming JSON
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("timestamp", DoubleType(), True),
        StructField("features", MapType(StringType(), DoubleType()), True),
        StructField("actual_label", IntegerType(), True)
    ])
    
    try:
        # Read from Kafka
        df = spark \\
            .readStream \\
            .format("kafka") \\
            .option("kafka.bootstrap.servers", "localhost:9092") \\
            .option("subscribe", "fraud-transactions") \\
            .option("startingOffsets", "earliest") \\
            .load()
        
        # Parse JSON
        transactions = df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
        
        # Process each batch
        def process_batch(batch_df, batch_id):
            print(f"\\n{'='*60}")
            print(f"Processing Batch #{batch_id}")
            print(f"{'='*60}")
            
            if batch_df.isEmpty():
                print("⚠ Empty batch")
                return
            
            batch_results = []
            
            for row in batch_df.collect():
                transaction_id = row['transaction_id']
                features = row['features']
                actual_label = row['actual_label']
                
                # Predict fraud
                fraud_score, prediction, latency = predict_fraud(features, models)
                
                # Determine result
                status = "🔴 FRAUD DETECTED" if prediction == 1 else "🟢 NORMAL"
                correct = "✓" if prediction == actual_label else "✗"
                
                batch_results.append({
                    'id': transaction_id,
                    'score': fraud_score,
                    'pred': prediction,
                    'actual': actual_label,
                    'correct': correct,
                    'latency': latency
                })
                
                # Print every 50th transaction
                if len(batch_results) % 50 == 0:
                    print(f"  {status} | ID: {transaction_id} | Score: {fraud_score:.4f} | "
                          f"Latency: {latency:.2f}ms {correct}")
            
            # Batch summary
            total = len(batch_results)
            correct_preds = sum(1 for r in batch_results if r['correct'] == '✓')
            fraud_detected = sum(1 for r in batch_results if r['pred'] == 1)
            avg_latency = np.mean([r['latency'] for r in batch_results])
            
            print(f"\\n📊 Batch Summary:")
            print(f"   Total: {total} | Correct: {correct_preds}/{total} ({correct_preds/total*100:.1f}%)")
            print(f"   Fraud Detected: {fraud_detected} | Avg Latency: {avg_latency:.2f}ms")
        
        # Start streaming query
        query = transactions \\
            .writeStream \\
            .foreachBatch(process_batch) \\
            .outputMode("append") \\
            .start()
        
        print("\\n🎯 Real-time fraud detection started!")
        print("   Listening to Kafka topic: 'fraud-transactions'")
        print("   Press Ctrl+C to stop\\n")
        
        query.awaitTermination()
        
    except KeyboardInterrupt:
        print("\\n⚠ Streaming stopped by user")
    except Exception as e:
        print(f"\\n✗ Streaming error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution"""
    print("=" * 60)
    print("  SPARK STREAMING - Real-Time Fraud Detection")
    print("=" * 60)
    
    # Create Spark session
    spark = create_spark_session()
    if not spark:
        sys.exit(1)
    
    # Load models
    print("\\nLoading fraud detection models...")
    models = load_models()
    if not models:
        print("✗ No models loaded. Train models first using the notebooks.")
        sys.exit(1)
    
    # Process stream
    try:
        process_stream(spark, models)
    finally:
        spark.stop()
        print("\\n✓ Spark session stopped")

if __name__ == '__main__':
    main()
