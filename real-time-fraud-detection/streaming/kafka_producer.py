import json
import time
from kafka import KafkaProducer

# Placeholder Kafka producer
producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

if __name__ == '__main__':
    for i in range(5):
        producer.send('transactions', {'id': i, 'amount': 100 + i})
        time.sleep(1)
