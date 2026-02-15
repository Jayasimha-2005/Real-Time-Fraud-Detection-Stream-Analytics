"""
Kafka Producer - Phase 4: Real-Time Streaming Simulation
Streams fraud detection test data row-by-row to Kafka topic
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import sys

def create_producer():
    """Create Kafka producer with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda v: v.encode('utf-8') if v else None,
                acks='all',
                retries=3
            )
            print("✓ Connected to Kafka broker")
            return producer
        except NoBrokersAvailable:
            print(f"⚠ Attempt {attempt + 1}/{max_retries}: Kafka broker not available")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("✗ Could not connect to Kafka. Make sure Kafka is running on localhost:9092")
                print("  Start Kafka with: bin/kafka-server-start.sh config/server.properties")
                return None

def stream_transactions(producer, data_path, topic='fraud-transactions', delay=0.1):
    """
    Stream transactions from CSV to Kafka topic
    
    Args:
        producer: Kafka producer instance
        data_path: Path to test data CSV
        topic: Kafka topic name
        delay: Delay between messages (seconds)
    """
    try:
        # Load test data
        df = pd.read_csv(data_path)
        print(f"\\n📊 Loaded {len(df)} transactions from {data_path}")
        print(f"🎯 Streaming to topic: '{topic}'")
        print(f"⏱ Delay between messages: {delay}s\\n")
        
        sent_count = 0
        fraud_count = 0
        
        for idx, row in df.iterrows():
            # Create transaction message
            transaction = {
                'transaction_id': f'TXN_{idx:06d}',
                'timestamp': time.time(),
                'features': row.drop('Class').to_dict() if 'Class' in row else row.to_dict(),
                'actual_label': int(row['Class']) if 'Class' in row else None
            }
            
            # Track fraud transactions
            if transaction['actual_label'] == 1:
                fraud_count += 1
            
            # Send to Kafka
            future = producer.send(
                topic,
                key=transaction['transaction_id'],
                value=transaction
            )
            
            # Wait for confirmation
            try:
                record_metadata = future.get(timeout=10)
                sent_count += 1
                
                # Print progress every 100 transactions
                if sent_count % 100 == 0:
                    print(f"✓ Sent {sent_count} transactions ({fraud_count} fraud detected)")
                    
            except Exception as e:
                print(f"✗ Failed to send transaction {transaction['transaction_id']}: {e}")
            
            # Simulate real-time delay
            time.sleep(delay)
        
        print(f"\\n🎉 Streaming completed!")
        print(f"   Total sent: {sent_count}")
        print(f"   Fraud transactions: {fraud_count}")
        print(f"   Normal transactions: {sent_count - fraud_count}")
        
    except FileNotFoundError:
        print(f"✗ Error: File not found: {data_path}")
        print("   Make sure you've run notebook 01 to generate preprocessed data")
    except KeyboardInterrupt:
        print("\\n⚠ Streaming interrupted by user")
    except Exception as e:
        print(f"✗ Error during streaming: {e}")
    finally:
        producer.flush()

def main():
    """Main execution"""
    print("=" * 60)
    print("  KAFKA PRODUCER - Real-Time Fraud Detection Streaming")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = '../../data/processed/X_test.csv'
    TOPIC = 'fraud-transactions'
    DELAY = 0.01  # 10ms delay = 100 transactions/second
    
    # Create producer
    producer = create_producer()
    if not producer:
        sys.exit(1)
    
    # Stream transactions
    try:
        stream_transactions(producer, DATA_PATH, TOPIC, DELAY)
    finally:
        producer.close()
        print("\\n✓ Producer closed")

if __name__ == '__main__':
    main()
