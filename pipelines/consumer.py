#!/usr/bin/env python3
"""
Kafka Consumer for Telco Customer Churn Prediction
Consumes customer events from Kafka, applies ML model, and publishes predictions.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import joblib
import pandas as pd
import numpy as np
import yaml
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError


# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

# Import preprocessing modules
from src.feature_engineering import ServicesScoreStrategy, VulnerabilityScoreStrategy
from src.feature_binning import TenureBinningStrategy
from src.handle_missing_values import MeanImputationStrategy, BinaryEncodingStrategy
from src.feature_encoding import NominalEncodingStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChurnPredictionProcessor:
    """
    Handles feature preprocessing and model inference for churn prediction
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.encoders = {}
        self.scaler = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load trained model and preprocessing artifacts"""
        # Load model
        model_path = os.path.join(
            self.config['data_paths']['model_artifacts_dir'], 
            'telco_analysis.joblib'
        )
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load encoders
        encoders_dir = self.config['data_paths']['artifacts_dir'] + '/encode'
        if os.path.exists(encoders_dir):
            for file in os.listdir(encoders_dir):
                if file.endswith('_encoder.json'):
                    feature_name = file.replace('_encoder.json', '')
                    with open(os.path.join(encoders_dir, file), 'r') as f:
                        self.encoders[feature_name] = json.load(f)
            logger.info(f"Loaded {len(self.encoders)} encoders")
    
    def _preprocess_message(self, message: Dict) -> pd.DataFrame:
        """
        Preprocess a single customer message for model prediction
        """
        # Convert message to DataFrame
        df = pd.DataFrame([message])
        logger.info(f"Initial DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        
        # Handle TotalCharges conversion
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.fillna({'TotalCharges': 0})
        
        # Apply preprocessing steps similar to training pipeline
        logger.info("Applying feature engineering...")
        df = self._apply_feature_engineering(df)
        logger.info(f"After feature engineering - shape: {df.shape}, columns: {list(df.columns)}")
        
        logger.info("Applying feature encoding...")
        df = self._apply_feature_encoding(df)
        logger.info(f"After feature encoding - shape: {df.shape}, columns: {list(df.columns)}")
        
        logger.info("Handling missing values...")
        df = self._handle_missing_values(df)
        logger.info(f"After handling missing values - shape: {df.shape}, columns: {list(df.columns)}")
        
        # Select features expected by the model
        feature_columns = self._get_model_features()
        logger.debug(f"Required model features: {feature_columns}")
        logger.debug(f"Available columns after encoding: {list(df.columns)}")
        
        # Ensure all required columns exist
        missing_cols = []
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
                missing_cols.append(col)
        
        if missing_cols:
            logger.warning(f"Created missing columns with default value 0: {missing_cols}")
        
        # Convert categorical tenure groups to numeric values for CatBoost
        final_df = df[feature_columns].copy()
        if 'tenure_group' in final_df.columns:
            # Map categorical values to numeric (as expected by trained model)
            tenure_mapping = {'New': 0, 'Intermediate': 1, 'Established': 2, 'Loyal': 3}
            final_df['tenure_group'] = final_df['tenure_group'].astype(str).map(tenure_mapping)
        
        logger.info(f"Final feature DataFrame shape: {final_df.shape}")
        return final_df
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        # Services Score
        services_strategy = ServicesScoreStrategy(
            self.config['feature_engineering']['services_score']['services']
        )
        df = services_strategy.transform(df)
        
        # Vulnerability Score
        vulnerability_strategy = VulnerabilityScoreStrategy(
            self.config['feature_engineering']['vulnerability_score']['tenure_threshold'],
            self.config['feature_engineering']['vulnerability_score']['weights']
        )
        df = vulnerability_strategy.transform(df)
        
        # Tenure Binning
        tenure_strategy = TenureBinningStrategy(
            self.config['feature_binning']['tenure']['bins'],
            self.config['feature_binning']['tenure']['labels']
        )
        df = tenure_strategy.transform(df)
        
        return df
    
    def _apply_feature_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature encoding using saved encoders"""
        # Binary encoding
        binary_mapping = self.config['feature_encoding']['binary_mapping']
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
        
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map(binary_mapping).fillna(0)
        
        # Handle special binary columns
        special_binary_cols = ['OnlineSecurity', 'TechSupport']
        for col in special_binary_cols:
            if col in df.columns:
                df[f'{col}_numeric'] = df[col].apply(
                    lambda x: 1 if x == 'Yes' else 0
                )
        
        # Label encoding using saved encoders (CatBoost expects original categorical columns)
        for feature_name, encoder_mapping in self.encoders.items():
            if feature_name in df.columns:
                # Apply label encoding (map categories to numbers)
                df[feature_name] = df[feature_name].map(encoder_mapping).fillna(0)
        
        # Handle tenure_group categorical column (keep as categorical for CatBoost)
        # CatBoost can handle categorical columns directly, so we keep tenure_group as-is
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill numeric columns with mean/median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df = df.fillna({col: df[col].mean()})
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_values = df[col].mode()
                fill_value = mode_values.iloc[0] if len(mode_values) > 0 else 'Unknown'
                df = df.fillna({col: fill_value})
        
        return df
    
    def _get_model_features(self) -> List[str]:
        """Get the feature names expected by the model"""
        # CRITICAL: This must match the EXACT order the model was trained with
        # The model expects original categorical columns, NOT one-hot encoded versions
        features = [
            'InternetService',      # Original categorical column
            'Contract',            # Original categorical column  
            'PaymentMethod',       # Original categorical column
            'MonthlyCharges',      # Numeric
            'TotalCharges',        # Numeric
            'OnlineSecurity_numeric',  # Binary encoded
            'TechSupport_numeric',     # Binary encoded
            'Services_Score',          # Engineered feature
            'Vulnerability_Score',     # Engineered feature  
            'tenure_group'            # Original categorical column (from binning)
        ]
        
        return features
    
    def predict(self, message: Dict) -> Dict:
        """
        Make churn prediction for a single customer message
        """
        try:
            # Preprocess the message
            logger.info(f"Starting prediction for customer {message.get('customerID', 'Unknown')}")
            processed_df = self._preprocess_message(message)
            logger.info(f"Preprocessing completed. DataFrame shape: {processed_df.shape}")
            logger.info(f"DataFrame columns: {list(processed_df.columns)}")
            
            # Make prediction
            prediction_proba = self.model.predict_proba(processed_df)[0]
            churn_probability = float(prediction_proba[1])  # Probability of churn
            prediction = "Yes" if churn_probability > 0.5 else "No"
            
            # Prepare result
            result = {
                "customerID": message["customerID"],
                "churn_probability": round(churn_probability, 4),
                "prediction": prediction,
                "event_ts": message["event_ts"],
                "processed_ts": datetime.now(timezone.utc).isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for customer {message.get('customerID', 'Unknown')}: {e}")
            return None


class TelcoKafkaConsumer:
    """
    Kafka Consumer for Telco Customer Churn Prediction
    Supports streaming and batch modes
    """
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize the consumer with configuration"""
        self.config = self._load_config(config_path)
        self.consumer = None
        self.producer = None
        self.processor = ChurnPredictionProcessor(self.config)
        
        # Topics
        self.input_topic = self.config['kafka']['topics']['raw_customers']
        self.output_topic = self.config['kafka']['topics']['churn_predictions']
        self.dead_letter_topic = self.config['kafka']['topics']['dead_letter']
        
        # Statistics for batch mode
        self.batch_stats = defaultdict(int)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_consumer(self) -> KafkaConsumer:
        """Create Kafka consumer instance"""
        kafka_config = self.config['kafka']
        
        consumer_config = {
            'bootstrap_servers': kafka_config['broker_config']['bootstrap_servers'],
            'group_id': kafka_config['consumer_config']['group_id'],
            'key_deserializer': lambda x: x.decode('utf-8') if x else None,
            'value_deserializer': lambda x: json.loads(x.decode('utf-8')),
            'auto_offset_reset': kafka_config['consumer_config']['auto_offset_reset'],
            'enable_auto_commit': kafka_config['consumer_config']['enable_auto_commit'],
            'max_poll_records': kafka_config['consumer_config']['max_poll_records'],
            'session_timeout_ms': kafka_config['consumer_config']['session_timeout_ms'],
            'heartbeat_interval_ms': kafka_config['consumer_config']['heartbeat_interval_ms']
        }
        
        return KafkaConsumer(self.input_topic, **consumer_config)
    
    def _create_producer(self) -> KafkaProducer:
        """Create Kafka producer instance for publishing predictions"""
        kafka_config = self.config['kafka']
        
        producer_config = {
            'bootstrap_servers': kafka_config['broker_config']['bootstrap_servers'],
            'client_id': kafka_config['broker_config']['client_id'] + "_consumer",
            'key_serializer': lambda x: str(x).encode('utf-8'),
            'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
            'acks': kafka_config['producer_config']['acks'],
            'retries': kafka_config['producer_config']['retries']
        }
        
        return KafkaProducer(**producer_config)
    
    def _send_to_dead_letter(self, message: Dict, error: str):
        """Send invalid message to dead letter queue"""
        dead_letter_message = {
            'original_message': message,
            'error': error,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            self.producer.send(
                self.dead_letter_topic,
                key=message.get('customerID', 'unknown'),
                value=dead_letter_message
            )
            logger.warning(f"Sent message to dead letter queue: {error}")
        except Exception as e:
            logger.error(f"Failed to send to dead letter queue: {e}")
    
    def _process_message(self, message: Dict) -> Optional[Dict]:
        """Process a single message and return prediction"""
        try:
            # Validate message structure
            required_fields = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']
            for field in required_fields:
                if field not in message:
                    raise ValueError(f"Missing required field: {field}")
            
            # Make prediction
            prediction = self.processor.predict(message)
            
            if prediction:
                logger.debug(f"Processed customer {message['customerID']}: {prediction['prediction']} ({prediction['churn_probability']:.4f})")
                return prediction
            else:
                raise ValueError("Prediction failed")
                
        except Exception as e:
            self._send_to_dead_letter(message, str(e))
            return None
    
    def run_streaming_mode(self):
        """
        Run consumer in streaming mode
        Continuously consumes messages and makes real-time predictions
        """
        logger.info("Starting streaming mode consumer")
        
        self.consumer = self._create_consumer()
        self.producer = self._create_producer()
        
        try:
            for message in self.consumer:
                try:
                    # Process message
                    prediction = self._process_message(message.value)
                    
                    if prediction:
                        # Send prediction to output topic
                        self.producer.send(
                            self.output_topic,
                            key=prediction['customerID'],
                            value=prediction
                        )
                        
                        logger.info(f"Published prediction for {prediction['customerID']}")
                    
                    # Commit offset manually
                    self.consumer.commit()
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Streaming mode interrupted by user")
        finally:
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()
    
    def run_batch_mode(self, max_records: int = 1000, timeout_ms: int = 30000):
        """
        Run consumer in batch mode
        Consumes a batch of messages and processes them together
        """
        logger.info(f"Starting batch mode consumer: max_records={max_records}")
        
        self.consumer = self._create_consumer()
        self.producer = self._create_producer()
        
        try:
            # Poll for messages
            message_batch = self.consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
            
            if not message_batch:
                logger.info("No messages received in batch mode")
                return
            
            total_messages = sum(len(messages) for messages in message_batch.values())
            logger.info(f"Processing batch of {total_messages} messages")
            
            predictions = []
            
            # Process all messages in batch
            for topic_partition, messages in message_batch.items():
                for message in messages:
                    prediction = self._process_message(message.value)
                    
                    if prediction:
                        predictions.append(prediction)
                        self.batch_stats['processed'] += 1
                        
                        # Update contract-wise stats
                        contract_type = message.value.get('Contract', 'Unknown')
                        if prediction['prediction'] == 'Yes':
                            self.batch_stats[f'churn_{contract_type}'] += 1
                        self.batch_stats[f'total_{contract_type}'] += 1
                    else:
                        self.batch_stats['failed'] += 1
            
            # Send all predictions
            for prediction in predictions:
                self.producer.send(
                    self.output_topic,
                    key=prediction['customerID'],
                    value=prediction
                )
            
            # Flush producer
            self.producer.flush()
            
            # Commit offsets
            self.consumer.commit()
            
            # Generate batch summary
            self._generate_batch_summary(predictions)
            
            logger.info(f"Batch processing completed: {len(predictions)} predictions published")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
        finally:
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()
    
    def _generate_batch_summary(self, predictions: List[Dict]):
        """Generate summary statistics for batch processing"""
        if not predictions:
            return
        
        summary = {
            'total_predictions': len(predictions),
            'churn_predictions': len([p for p in predictions if p['prediction'] == 'Yes']),
            'churn_percentage': len([p for p in predictions if p['prediction'] == 'Yes']) / len(predictions) * 100,
            'avg_churn_probability': np.mean([p['churn_probability'] for p in predictions]),
            'high_risk_customers': len([p for p in predictions if p['churn_probability'] > 0.8]),
            'processing_time': datetime.now(timezone.utc).isoformat()
        }
        
        # Contract-wise breakdown
        contract_breakdown = {}
        for prediction in predictions:
            # We don't have contract info in prediction, would need original message
            pass
        
        logger.info("=== BATCH SUMMARY ===")
        logger.info(f"Total Predictions: {summary['total_predictions']}")
        logger.info(f"Churn Predictions: {summary['churn_predictions']} ({summary['churn_percentage']:.2f}%)")
        logger.info(f"Average Churn Probability: {summary['avg_churn_probability']:.4f}")
        logger.info(f"High Risk Customers (>80%): {summary['high_risk_customers']}")
        
        # Save summary to file
        summary_path = os.path.join(
            self.config['data_paths']['artifacts_dir'], 
            'evaluation', 
            f"batch_summary_{int(time.time())}.json"
        )
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch summary saved to: {summary_path}")


def main():
    """Main function to run the consumer"""
    parser = argparse.ArgumentParser(description='Telco Kafka Consumer')
    parser.add_argument('--mode', choices=['streaming', 'batch'], required=True,
                       help='Consumer mode: streaming or batch')
    parser.add_argument('--config-path', default='config.yml',
                       help='Path to configuration file')
    parser.add_argument('--max-records', type=int, default=1000,
                       help='Maximum records to process in batch mode')
    parser.add_argument('--timeout-ms', type=int, default=30000,
                       help='Timeout in milliseconds for batch mode')
    
    args = parser.parse_args()
    
    try:
        consumer = TelcoKafkaConsumer(args.config_path)
        
        if args.mode == 'streaming':
            consumer.run_streaming_mode()
        elif args.mode == 'batch':
            consumer.run_batch_mode(args.max_records, args.timeout_ms)
            
    except Exception as e:
        logger.error(f"Consumer failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()