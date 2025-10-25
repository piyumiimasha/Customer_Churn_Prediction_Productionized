#!/usr/bin/env python3
"""
Kafka Producer for Telco Customer Churn Data
Supports both streaming and batch modes for publishing customer events to Kafka.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from kafka import KafkaProducer
from kafka.errors import KafkaError


# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelcoKafkaProducer:
    """
    Kafka Producer for Telco Customer Churn Events
    Supports streaming and batch modes
    """
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize the producer with configuration"""
        self.config = self._load_config(config_path)
        self.producer = None
        self.topic = self.config['kafka']['topics']['raw_customers']
        self.checkpoint_file = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_producer(self) -> KafkaProducer:
        """Create Kafka producer instance"""
        kafka_config = self.config['kafka']
        
        producer_config = {
            'bootstrap_servers': kafka_config['broker_config']['bootstrap_servers'],
            'client_id': kafka_config['broker_config']['client_id'],
            'key_serializer': lambda x: str(x).encode('utf-8'),
            'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
            'acks': kafka_config['producer_config']['acks'],
            'retries': kafka_config['producer_config']['retries'],
            'batch_size': kafka_config['producer_config']['batch_size'],
            'linger_ms': kafka_config['producer_config']['linger_ms'],
            'buffer_memory': kafka_config['producer_config']['buffer_memory'],
            'compression_type': kafka_config['producer_config']['compression_type']
        }
        
        return KafkaProducer(**producer_config)
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load customer data from file"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Support both CSV and Excel files
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded {len(df)} records from {data_path}")
        return df
    
    def _prepare_message(self, row: pd.Series) -> Dict:
        """Convert DataFrame row to Kafka message format"""
        # Add event timestamp
        current_time = datetime.now(timezone.utc).isoformat()
        
        message = {
            "customerID": str(row.get('customerID', '')),
            "gender": str(row.get('gender', '')),
            "SeniorCitizen": int(row.get('SeniorCitizen', 0)),
            "Partner": str(row.get('Partner', '')),
            "Dependents": str(row.get('Dependents', '')),
            "tenure": int(row.get('tenure', 0)),
            "PhoneService": str(row.get('PhoneService', '')),
            "MultipleLines": str(row.get('MultipleLines', '')),
            "InternetService": str(row.get('InternetService', '')),
            "OnlineSecurity": str(row.get('OnlineSecurity', '')),
            "OnlineBackup": str(row.get('OnlineBackup', '')),
            "DeviceProtection": str(row.get('DeviceProtection', '')),
            "TechSupport": str(row.get('TechSupport', '')),
            "StreamingTV": str(row.get('StreamingTV', '')),
            "StreamingMovies": str(row.get('StreamingMovies', '')),
            "Contract": str(row.get('Contract', '')),
            "PaperlessBilling": str(row.get('PaperlessBilling', '')),
            "PaymentMethod": str(row.get('PaymentMethod', '')),
            "MonthlyCharges": float(row.get('MonthlyCharges', 0.0)),
            "TotalCharges": self._convert_total_charges(row.get('TotalCharges')),
            "Churn": str(row.get('Churn', '')),
            "event_ts": current_time
        }
        
        return message
    
    def _convert_total_charges(self, total_charges) -> float:
        """Convert TotalCharges to float, handling string values"""
        if pd.isna(total_charges) or total_charges == '' or total_charges == ' ':
            return 0.0
        try:
            return float(total_charges)
        except (ValueError, TypeError):
            return 0.0
    
    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint to resume from last processed record"""
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                return int(f.read().strip())
        return 0
    
    def _save_checkpoint(self, checkpoint_path: str, record_count: int):
        """Save current progress to checkpoint file"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            f.write(str(record_count))
    
    def run_streaming_mode(self, data_path: str, events_per_sec: int = 10):
        """
        Run producer in streaming mode
        Continuously samples rows from dataset and publishes to Kafka
        """
        logger.info(f"Starting streaming mode: {events_per_sec} events/sec")
        
        df = self._load_data(data_path)
        self.producer = self._create_producer()
        
        sleep_interval = 1.0 / events_per_sec
        
        try:
            while True:
                # Sample random row from dataset
                row = df.sample(n=1).iloc[0]
                message = self._prepare_message(row)
                
                # Send to Kafka
                future = self.producer.send(
                    self.topic,
                    key=message['customerID'],
                    value=message
                )
                
                # Log successful send
                try:
                    record_metadata = future.get(timeout=10)
                    logger.debug(
                        f"Sent record to {record_metadata.topic} "
                        f"partition {record_metadata.partition} "
                        f"offset {record_metadata.offset}"
                    )
                except KafkaError as e:
                    logger.error(f"Failed to send message: {e}")
                
                time.sleep(sleep_interval)
                
        except KeyboardInterrupt:
            logger.info("Streaming mode interrupted by user")
        finally:
            self.producer.close()
    
    def run_batch_mode(self, data_path: str, batch_size: int = 1000, 
                      resume: bool = True):
        """
        Run producer in batch mode
        Processes dataset in chunks and publishes to Kafka
        """
        logger.info(f"Starting batch mode: batch_size={batch_size}, resume={resume}")
        
        df = self._load_data(data_path)
        self.producer = self._create_producer()
        
        # Setup checkpoint
        checkpoint_dir = self.config['kafka']['batch']['checkpoint_dir']
        self.checkpoint_file = os.path.join(checkpoint_dir, 'producer_checkpoint.txt')
        
        start_idx = 0
        if resume:
            start_idx = self._load_checkpoint(self.checkpoint_file)
            logger.info(f"Resuming from record {start_idx}")
        
        total_records = len(df)
        
        try:
            for i in range(start_idx, total_records, batch_size):
                batch_end = min(i + batch_size, total_records)
                batch_df = df.iloc[i:batch_end]
                
                logger.info(f"Processing batch {i//batch_size + 1}: records {i}-{batch_end-1}")
                
                # Send batch
                for idx, row in batch_df.iterrows():
                    message = self._prepare_message(row)
                    
                    future = self.producer.send(
                        self.topic,
                        key=message['customerID'],
                        value=message
                    )
                
                # Flush to ensure all messages are sent
                self.producer.flush()
                
                # Save checkpoint
                self._save_checkpoint(self.checkpoint_file, batch_end)
                
                logger.info(f"Completed batch: {batch_end - i} records sent")
                
                # Small delay between batches
                time.sleep(0.1)
            
            logger.info(f"Batch processing completed: {total_records} records processed")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
        finally:
            self.producer.close()


def main():
    """Main function to run the producer"""
    parser = argparse.ArgumentParser(description='Telco Kafka Producer')
    parser.add_argument('--mode', choices=['streaming', 'batch'], required=True,
                       help='Producer mode: streaming or batch')
    parser.add_argument('--data-path', required=True,
                       help='Path to the dataset file')
    parser.add_argument('--config-path', default='config.yml',
                       help='Path to configuration file')
    parser.add_argument('--events-per-sec', type=int, default=10,
                       help='Events per second for streaming mode')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for batch mode')
    parser.add_argument('--no-resume', action='store_true',
                       help='Disable resume from checkpoint in batch mode')
    
    args = parser.parse_args()
    
    try:
        producer = TelcoKafkaProducer(args.config_path)
        
        if args.mode == 'streaming':
            producer.run_streaming_mode(args.data_path, args.events_per_sec)
        elif args.mode == 'batch':
            resume = not args.no_resume
            producer.run_batch_mode(args.data_path, args.batch_size, resume)
            
    except Exception as e:
        logger.error(f"Producer failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()