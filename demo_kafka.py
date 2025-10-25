#!/usr/bin/env python3
"""
Demo script to showcase Kafka integration
Demonstrates end-to-end message flow from producer to consumer
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import yaml

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'pipelines'))


def create_demo_data():
    """Create a small demo dataset for testing"""
    demo_data = pd.DataFrame({
        'customerID': ['DEMO-001', 'DEMO-002', 'DEMO-003', 'DEMO-004', 'DEMO-005'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No', 'Yes', 'No'],
        'tenure': [12, 48, 6, 24, 60],
        'PhoneService': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'OnlineBackup': ['No', 'Yes', 'No', 'No', 'Yes'],
        'DeviceProtection': ['No', 'No', 'Yes', 'No', 'Yes'],
        'TechSupport': ['No', 'Yes', 'No', 'No', 'Yes'],
        'StreamingTV': ['No', 'Yes', 'No', 'No', 'Yes'],
        'StreamingMovies': ['No', 'No', 'Yes', 'No', 'Yes'],
        'Contract': ['Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Credit card (automatic)', 'Mailed check', 'Bank transfer (automatic)', 'Electronic check'],
        'MonthlyCharges': [29.85, 89.10, 53.45, 42.30, 103.20],
        'TotalCharges': [358.20, 4274.80, 320.70, 1017.20, 6193.20],
        'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
    })
    
    demo_file = 'demo_data.csv'
    demo_data.to_csv(demo_file, index=False)
    print(f"✓ Created demo dataset: {demo_file} ({len(demo_data)} records)")
    return demo_file


def check_kafka_connectivity():
    """Check if Kafka is running and accessible"""
    try:
        from kafka import KafkaProducer
        from kafka.errors import NoBrokersAvailable
        
        # Try to create a producer
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            request_timeout_ms=5000
        )
        producer.close()
        print("✓ Kafka connectivity verified")
        return True
        
    except ImportError:
        print("✗ Kafka Python library not installed")
        print("  Run: pip install kafka-python")
        return False
    except NoBrokersAvailable:
        print("✗ Kafka broker not accessible")
        print("  Make sure Kafka is running on localhost:9092")
        return False
    except Exception as e:
        print(f"✗ Kafka connectivity check failed: {e}")
        return False


def run_demo_producer(demo_file, duration=30):
    """Run producer in demo mode"""
    try:
        from producer import TelcoKafkaProducer
        
        print(f"Starting demo producer for {duration} seconds...")
        
        # Create producer
        producer = TelcoKafkaProducer('config.yml')
        producer.producer = producer._create_producer()
        
        # Load demo data
        df = pd.read_csv(demo_file)
        
        # Send messages for specified duration
        start_time = time.time()
        message_count = 0
        
        while time.time() - start_time < duration:
            # Sample random row
            row = df.sample(n=1).iloc[0]
            message = producer._prepare_message(row)
            
            # Send to Kafka
            future = producer.producer.send(
                producer.topic,
                key=message['customerID'],
                value=message
            )
            
            message_count += 1
            print(f"  Sent message {message_count}: {message['customerID']}")
            
            time.sleep(2)  # Send every 2 seconds
        
        producer.producer.close()
        print(f"✓ Demo producer completed: {message_count} messages sent")
        
    except Exception as e:
        print(f"✗ Demo producer failed: {e}")


def run_demo_consumer(duration=25):
    """Run consumer in demo mode"""
    try:
        from consumer import TelcoKafkaConsumer
        
        print(f"Starting demo consumer for {duration} seconds...")
        
        # Create consumer
        consumer = TelcoKafkaConsumer('config.yml')
        consumer.consumer = consumer._create_consumer()
        consumer.producer = consumer._create_producer()
        
        # Process messages for specified duration
        start_time = time.time()
        message_count = 0
        prediction_count = 0
        
        consumer.consumer.subscribe([consumer.input_topic])
        
        while time.time() - start_time < duration:
            # Poll for messages
            message_batch = consumer.consumer.poll(timeout_ms=1000)
            
            if message_batch:
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            message_count += 1
                            print(f"  Received message {message_count}: {message.value['customerID']}")
                            
                            # Process message
                            prediction = consumer._process_message(message.value)
                            
                            if prediction:
                                # Send prediction
                                consumer.producer.send(
                                    consumer.output_topic,
                                    key=prediction['customerID'],
                                    value=prediction
                                )
                                
                                prediction_count += 1
                                print(f"    → Prediction: {prediction['prediction']} ({prediction['churn_probability']:.3f})")
                            
                        except Exception as e:
                            print(f"    ✗ Processing error: {e}")
                
                # Commit offsets
                consumer.consumer.commit()
        
        consumer.consumer.close()
        consumer.producer.close()
        
        print(f"✓ Demo consumer completed: {message_count} messages processed, {prediction_count} predictions made")
        
    except Exception as e:
        print(f"✗ Demo consumer failed: {e}")


def monitor_predictions(duration=30):
    """Monitor prediction topic"""
    try:
        from kafka import KafkaConsumer
        
        print(f"Monitoring predictions for {duration} seconds...")
        
        # Create consumer for predictions topic
        consumer = KafkaConsumer(
            'telco.churn.predictions',
            bootstrap_servers='localhost:9092',
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=1000
        )
        
        start_time = time.time()
        prediction_count = 0
        churn_predictions = 0
        
        while time.time() - start_time < duration:
            for message in consumer:
                prediction = message.value
                prediction_count += 1
                
                if prediction['prediction'] == 'Yes':
                    churn_predictions += 1
                
                print(f"📊 Prediction {prediction_count}: {prediction['customerID']} → "
                      f"{prediction['prediction']} ({prediction['churn_probability']:.3f})")
                
                if time.time() - start_time >= duration:
                    break
        
        consumer.close()
        
        if prediction_count > 0:
            churn_rate = (churn_predictions / prediction_count) * 100
            print(f"✓ Monitoring completed: {prediction_count} predictions, {churn_rate:.1f}% churn rate")
        else:
            print("⚠ No predictions received during monitoring period")
        
    except Exception as e:
        print(f"✗ Prediction monitoring failed: {e}")


def cleanup_demo_files():
    """Clean up demo files"""
    demo_files = ['demo_data.csv']
    
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Cleaned up: {file}")


def main():
    """Main demo function"""
    print("=== Kafka Integration Demo ===")
    print()
    
    # Step 1: Check Kafka connectivity
    if not check_kafka_connectivity():
        print("\nPlease ensure Kafka is running and try again.")
        sys.exit(1)
    
    # Step 2: Create demo data
    demo_file = create_demo_data()
    
    # Step 3: Check if model exists
    model_path = Path('artifacts/models/telco_analysis.joblib')
    if not model_path.exists():
        print(f"\n⚠ Model not found at {model_path}")
        print("The consumer will attempt to run but may fail without the trained model.")
        input("Press Enter to continue anyway...")
    
    try:
        print("\n=== Starting Demo ===")
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor_predictions, args=(35,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Small delay to let monitor start
        time.sleep(2)
        
        # Start consumer in background
        consumer_thread = threading.Thread(target=run_demo_consumer, args=(30,))
        consumer_thread.daemon = True
        consumer_thread.start()
        
        # Small delay to let consumer start
        time.sleep(3)
        
        # Run producer (main thread)
        run_demo_producer(demo_file, duration=25)
        
        # Wait for threads to complete
        print("\nWaiting for background processes to complete...")
        consumer_thread.join(timeout=35)
        monitor_thread.join(timeout=40)
        
        print("\n=== Demo Results ===")
        print("✓ Demo completed successfully!")
        print("\nWhat happened:")
        print("1. Producer sent customer events to telco.raw.customers topic")
        print("2. Consumer processed events and applied ML model")
        print("3. Predictions were published to telco.churn.predictions topic")
        print("4. Monitor displayed real-time predictions")
        
        print("\nNext steps:")
        print("- Run full streaming mode: python pipelines/producer.py --mode streaming --data-path data/hmQOVnDvRN.xls")
        print("- Run batch processing: python pipelines/producer.py --mode batch --data-path data/hmQOVnDvRN.xls")
        print("- Monitor with Kafka console tools")
        print("- Set up Airflow DAGs for orchestration")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
    finally:
        # Cleanup
        cleanup_demo_files()


if __name__ == "__main__":
    main()