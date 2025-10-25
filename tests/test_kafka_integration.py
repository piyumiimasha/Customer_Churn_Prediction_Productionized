"""
Test suite for Kafka integration components
Tests producer, consumer, and end-to-end pipeline functionality
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'pipelines'))


class TestKafkaProducer(unittest.TestCase):
    """Test cases for Kafka Producer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'kafka': {
                'broker_config': {'bootstrap_servers': 'localhost:9092', 'client_id': 'test'},
                'topics': {'raw_customers': 'test.raw.customers'},
                'producer_config': {
                    'acks': 'all', 'retries': 3, 'batch_size': 16384,
                    'linger_ms': 10, 'buffer_memory': 33554432, 'compression_type': 'gzip'
                },
                'batch': {'checkpoint_dir': 'test_checkpoints/'}
            },
            'data_paths': {'artifacts_dir': 'test_artifacts'}
        }
        
        # Create test data
        self.test_data = pd.DataFrame({
            'customerID': ['TEST-001', 'TEST-002'],
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'Partner': ['Yes', 'No'],
            'Dependents': ['No', 'Yes'],
            'tenure': [12, 24],
            'PhoneService': ['Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic'],
            'OnlineSecurity': ['Yes', 'No'],
            'OnlineBackup': ['No', 'Yes'],
            'DeviceProtection': ['No', 'No'],
            'TechSupport': ['No', 'Yes'],
            'StreamingTV': ['No', 'Yes'],
            'StreamingMovies': ['No', 'No'],
            'Contract': ['Month-to-month', 'Two year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Credit card (automatic)'],
            'MonthlyCharges': [29.85, 89.10],
            'TotalCharges': [29.85, 2137.40],
            'Churn': ['No', 'Yes']
        })
    
    @patch('pipelines.producer.KafkaProducer')
    @patch('pipelines.producer.yaml.safe_load')
    def test_producer_initialization(self, mock_yaml, mock_kafka_producer):
        """Test producer initialization"""
        mock_yaml.return_value = self.config
        
        from producer import TelcoKafkaProducer
        
        producer = TelcoKafkaProducer('test_config.yml')
        self.assertIsNotNone(producer.config)
        self.assertEqual(producer.topic, 'test.raw.customers')
    
    @patch('pipelines.producer.KafkaProducer')
    @patch('pipelines.producer.yaml.safe_load')
    def test_message_preparation(self, mock_yaml, mock_kafka_producer):
        """Test message format preparation"""
        mock_yaml.return_value = self.config
        
        from producer import TelcoKafkaProducer
        
        producer = TelcoKafkaProducer('test_config.yml')
        
        # Test message preparation
        row = self.test_data.iloc[0]
        message = producer._prepare_message(row)
        
        # Verify message structure
        self.assertIn('customerID', message)
        self.assertIn('event_ts', message)
        self.assertEqual(message['customerID'], 'TEST-001')
        self.assertEqual(message['gender'], 'Male')
        self.assertEqual(message['SeniorCitizen'], 0)
    
    @patch('pipelines.producer.KafkaProducer')
    @patch('pipelines.producer.yaml.safe_load')
    @patch('pandas.read_csv')
    def test_batch_mode_processing(self, mock_read_csv, mock_yaml, mock_kafka_producer):
        """Test batch mode processing"""
        mock_yaml.return_value = self.config
        mock_read_csv.return_value = self.test_data
        
        # Mock Kafka producer
        mock_producer_instance = Mock()
        mock_kafka_producer.return_value = mock_producer_instance
        
        from producer import TelcoKafkaProducer
        
        producer = TelcoKafkaProducer('test_config.yml')
        
        # Create test data file
        test_file = 'test_data.csv'
        self.test_data.to_csv(test_file, index=False)
        
        try:
            # Test batch processing (small batch size)
            producer.run_batch_mode(test_file, batch_size=1, resume=False)
            
            # Verify producer.send was called
            self.assertTrue(mock_producer_instance.send.called)
            self.assertTrue(mock_producer_instance.flush.called)
            
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)


class TestKafkaConsumer(unittest.TestCase):
    """Test cases for Kafka Consumer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'kafka': {
                'broker_config': {'bootstrap_servers': 'localhost:9092', 'client_id': 'test'},
                'topics': {
                    'raw_customers': 'test.raw.customers',
                    'churn_predictions': 'test.churn.predictions',
                    'dead_letter': 'test.deadletter'
                },
                'consumer_config': {
                    'group_id': 'test_group',
                    'auto_offset_reset': 'earliest',
                    'enable_auto_commit': False,
                    'max_poll_records': 500,
                    'session_timeout_ms': 30000,
                    'heartbeat_interval_ms': 3000
                },
                'producer_config': {
                    'acks': 'all', 'retries': 3
                }
            },
            'data_paths': {
                'model_artifacts_dir': 'test_artifacts/models',
                'artifacts_dir': 'test_artifacts'
            }
        }
        
        self.test_message = {
            'customerID': 'TEST-001',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 29.85,
            'TotalCharges': 29.85,
            'Churn': 'No',
            'event_ts': '2025-10-25T10:00:00Z'
        }
    
    @patch('pipelines.consumer.yaml.safe_load')
    @patch('pipelines.consumer.joblib.load')
    @patch('os.path.exists')
    def test_processor_initialization(self, mock_exists, mock_joblib, mock_yaml):
        """Test processor initialization"""
        mock_yaml.return_value = self.config
        mock_exists.return_value = True
        mock_joblib.return_value = Mock()  # Mock model
        
        from consumer import ChurnPredictionProcessor
        
        processor = ChurnPredictionProcessor(self.config)
        self.assertIsNotNone(processor.model)
        self.assertIsInstance(processor.encoders, dict)
    
    @patch('pipelines.consumer.yaml.safe_load')
    @patch('pipelines.consumer.joblib.load')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_message_preprocessing(self, mock_listdir, mock_exists, mock_joblib, mock_yaml):
        """Test message preprocessing"""
        mock_yaml.return_value = self.config
        mock_exists.return_value = True
        mock_joblib.return_value = Mock()
        mock_listdir.return_value = ['Contract_encoder.json']
        
        # Mock encoder file
        with patch('builtins.open', unittest.mock.mock_open(read_data='{"Month-to-month": ["Contract_Month-to-month"]}')):
            from consumer import ChurnPredictionProcessor
            
            processor = ChurnPredictionProcessor(self.config)
            
            # Test preprocessing
            processed_df = processor._preprocess_message(self.test_message)
            
            self.assertIsInstance(processed_df, pd.DataFrame)
            self.assertEqual(len(processed_df), 1)
    
    @patch('pipelines.consumer.yaml.safe_load')
    @patch('pipelines.consumer.joblib.load')
    @patch('os.path.exists')
    def test_prediction_output_format(self, mock_exists, mock_joblib, mock_yaml):
        """Test prediction output format"""
        mock_yaml.return_value = self.config
        mock_exists.return_value = True
        
        # Mock model with predict_proba
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.3, 0.7]]  # 70% churn probability
        mock_joblib.return_value = mock_model
        
        with patch('os.listdir', return_value=[]):
            from consumer import ChurnPredictionProcessor
            
            processor = ChurnPredictionProcessor(self.config)
            
            # Test prediction
            result = processor.predict(self.test_message)
            
            self.assertIsNotNone(result)
            self.assertIn('customerID', result)
            self.assertIn('churn_probability', result)
            self.assertIn('prediction', result)
            self.assertIn('event_ts', result)
            self.assertIn('processed_ts', result)
            
            self.assertEqual(result['customerID'], 'TEST-001')
            self.assertEqual(result['churn_probability'], 0.7)
            self.assertEqual(result['prediction'], 'Yes')


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end pipeline tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_path = 'test_config.yml'
        self.test_config = {
            'kafka': {
                'broker_config': {'bootstrap_servers': 'localhost:9092', 'client_id': 'test'},
                'topics': {
                    'raw_customers': 'test.raw.customers',
                    'churn_predictions': 'test.churn.predictions',
                    'dead_letter': 'test.deadletter'
                },
                'producer_config': {
                    'acks': 'all', 'retries': 3, 'batch_size': 16384,
                    'linger_ms': 10, 'buffer_memory': 33554432, 'compression_type': 'gzip'
                },
                'consumer_config': {
                    'group_id': 'test_group',
                    'auto_offset_reset': 'earliest',
                    'enable_auto_commit': False,
                    'max_poll_records': 500,
                    'session_timeout_ms': 30000,
                    'heartbeat_interval_ms': 3000
                },
                'batch': {'checkpoint_dir': 'test_checkpoints/'}
            },
            'data_paths': {
                'model_artifacts_dir': 'test_artifacts/models',
                'artifacts_dir': 'test_artifacts'
            }
        }
    
    def test_configuration_loading(self):
        """Test configuration file loading"""
        # Create temporary config file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            self.assertEqual(loaded_config['kafka']['topics']['raw_customers'], 'test.raw.customers')
            self.assertEqual(loaded_config['kafka']['broker_config']['bootstrap_servers'], 'localhost:9092')
            
        finally:
            # Cleanup
            if os.path.exists(self.config_path):
                os.remove(self.config_path)
    
    def test_message_schema_validation(self):
        """Test message schema validation"""
        valid_message = {
            'customerID': 'TEST-001',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 29.85,
            'TotalCharges': 29.85,
            'Churn': 'No',
            'event_ts': '2025-10-25T10:00:00Z'
        }
        
        # Test required fields
        required_fields = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']
        
        for field in required_fields:
            self.assertIn(field, valid_message)
        
        # Test invalid message (missing required field)
        invalid_message = valid_message.copy()
        del invalid_message['customerID']
        
        # Should fail validation
        for field in required_fields:
            if field not in invalid_message:
                with self.assertRaises(AssertionError):
                    self.assertIn(field, invalid_message)
                break


class TestKafkaUtils(unittest.TestCase):
    """Test utility functions for Kafka integration"""
    
    def test_total_charges_conversion(self):
        """Test TotalCharges conversion utility"""
        from producer import TelcoKafkaProducer
        
        producer = TelcoKafkaProducer.__new__(TelcoKafkaProducer)  # Create without init
        
        # Test various input formats
        self.assertEqual(producer._convert_total_charges('100.50'), 100.50)
        self.assertEqual(producer._convert_total_charges(100.50), 100.50)
        self.assertEqual(producer._convert_total_charges(''), 0.0)
        self.assertEqual(producer._convert_total_charges(' '), 0.0)
        self.assertEqual(producer._convert_total_charges(None), 0.0)
        self.assertEqual(producer._convert_total_charges('invalid'), 0.0)
    
    def test_checkpoint_functionality(self):
        """Test checkpoint save/load functionality"""
        from producer import TelcoKafkaProducer
        
        producer = TelcoKafkaProducer.__new__(TelcoKafkaProducer)  # Create without init
        
        checkpoint_path = 'test_checkpoint.txt'
        
        try:
            # Test saving checkpoint
            producer._save_checkpoint(checkpoint_path, 150)
            
            # Test loading checkpoint
            loaded_count = producer._load_checkpoint(checkpoint_path)
            self.assertEqual(loaded_count, 150)
            
            # Test loading non-existent checkpoint
            non_existent_path = 'non_existent_checkpoint.txt'
            default_count = producer._load_checkpoint(non_existent_path)
            self.assertEqual(default_count, 0)
            
        finally:
            # Cleanup
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


if __name__ == '__main__':
    # Create test directories
    os.makedirs('test_artifacts/models', exist_ok=True)
    os.makedirs('test_checkpoints', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)