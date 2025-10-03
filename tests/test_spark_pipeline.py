"""
Test script to validate Spark pipeline setup and functionality
"""

import os
import sys
import logging
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_spark_imports():
    """Test if all Spark-related imports work correctly"""
    try:
        logger.info("🧪 Testing Spark imports...")
        
        # Test PySpark imports
        from pyspark.sql import SparkSession
        from pyspark.ml.classification import GBTClassifier
        from pyspark.ml.feature import VectorAssembler
        logger.info("✅ PySpark imports successful")
        
        # Test custom Spark modules
        from spark_session import create_spark_session
        logger.info("✅ spark_session import successful")
        
        from spark_data_pipeline import SparkDataPipeline
        logger.info("✅ spark_data_pipeline import successful")
        
        from spark_model_trainer import SparkModelTrainer
        logger.info("✅ spark_model_trainer import successful")
        
        # Test configuration
        from config import get_spark_config
        spark_config = get_spark_config()
        logger.info(f"✅ Spark configuration loaded: {spark_config['app_name']}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return False


def test_spark_session_creation():
    """Test if SparkSession can be created successfully"""
    try:
        logger.info("🧪 Testing SparkSession creation...")
        
        from spark_session import create_spark_session, stop_spark_session
        
        # Create SparkSession
        spark = create_spark_session(
            app_name="TestSparkPipeline",
            master="local[2]"  # Use 2 cores for testing
        )
        
        logger.info(f"✅ SparkSession created successfully")
        logger.info(f"  • Version: {spark.version}")
        logger.info(f"  • Master: {spark.sparkContext.master}")
        logger.info(f"  • Default Parallelism: {spark.sparkContext.defaultParallelism}")
        logger.info(f"  • App Name: {spark.conf.get('spark.app.name')}")
        
        # Test basic DataFrame operations
        data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
        columns = ["name", "age"]
        df = spark.createDataFrame(data, columns)
        
        count = df.count()
        logger.info(f"✅ Basic DataFrame operations work (test data count: {count})")
        
        # Stop session
        stop_spark_session(spark)
        logger.info("✅ SparkSession stopped successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SparkSession test failed: {str(e)}")
        return False


def test_data_pipeline_initialization():
    """Test if SparkDataPipeline can be initialized"""
    try:
        logger.info("🧪 Testing SparkDataPipeline initialization...")
        
        from spark_data_pipeline import SparkDataPipeline
        
        # Initialize pipeline
        pipeline = SparkDataPipeline()
        logger.info("✅ SparkDataPipeline initialized successfully")
        
        # Test if spark session is available
        if pipeline.spark:
            logger.info(f"  • Spark Version: {pipeline.spark.version}")
            logger.info(f"  • App Name: {pipeline.spark.conf.get('spark.app.name')}")
        
        # Clean up
        if pipeline.spark:
            pipeline.spark.stop()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SparkDataPipeline test failed: {str(e)}")
        return False


def test_model_trainer_initialization():
    """Test if SparkModelTrainer can be initialized"""
    try:
        logger.info("🧪 Testing SparkModelTrainer initialization...")
        
        from spark_model_trainer import SparkModelTrainer
        
        # Initialize trainer
        trainer = SparkModelTrainer()
        logger.info("✅ SparkModelTrainer initialized successfully")
        
        # Test model setup
        models = trainer.setup_models()
        logger.info(f"✅ {len(models)} models configured successfully")
        for model_name in models.keys():
            logger.info(f"  • {model_name}")
        
        # Clean up
        if trainer.spark:
            trainer.spark.stop()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SparkModelTrainer test failed: {str(e)}")
        return False


def test_sample_data_processing():
    """Test data processing with sample data"""
    try:
        logger.info("🧪 Testing sample data processing...")
        
        from spark_data_pipeline import SparkDataPipeline
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
        
        # Initialize pipeline
        pipeline = SparkDataPipeline()
        
        # Create sample data that mimics churn dataset structure
        schema = StructType([
            StructField("customerID", StringType(), True),
            StructField("gender", StringType(), True),
            StructField("SeniorCitizen", IntegerType(), True),
            StructField("Partner", StringType(), True),
            StructField("Dependents", StringType(), True),
            StructField("tenure", IntegerType(), True),
            StructField("PhoneService", StringType(), True),
            StructField("InternetService", StringType(), True),
            StructField("Contract", StringType(), True),
            StructField("PaymentMethod", StringType(), True),
            StructField("MonthlyCharges", DoubleType(), True),
            StructField("TotalCharges", DoubleType(), True),
            StructField("Churn", StringType(), True)
        ])
        
        sample_data = [
            ("C001", "Male", 0, "Yes", "No", 12, "Yes", "DSL", "Month-to-month", "Credit card", 45.50, 546.0, "No"),
            ("C002", "Female", 1, "No", "Yes", 24, "Yes", "Fiber optic", "One year", "Bank transfer", 89.99, 2159.76, "Yes"),
            ("C003", "Male", 0, "Yes", "Yes", 36, "No", "No", "Two year", "Electronic check", 25.25, 909.0, "No"),
            ("C004", "Female", 0, "No", "No", 6, "Yes", "DSL", "Month-to-month", "Mailed check", 55.75, 334.5, "Yes"),
            ("C005", "Male", 1, "Yes", "No", 48, "Yes", "Fiber optic", "Two year", "Credit card", 95.00, 4560.0, "No")
        ]
        
        df = pipeline.spark.createDataFrame(sample_data, schema)
        logger.info(f"✅ Sample data created ({df.count()} rows)")
        
        # Test data quality analysis
        quality_report = pipeline.analyze_data_quality(df)
        logger.info(f"✅ Data quality analysis completed")
        logger.info(f"  • Total rows: {quality_report['total_rows']}")
        logger.info(f"  • Total columns: {quality_report['total_columns']}")
        
        # Test missing value handling
        df_clean = pipeline.handle_missing_values(df)
        logger.info(f"✅ Missing value handling completed")
        
        # Test feature engineering
        df_engineered = pipeline.feature_engineering(df_clean)
        logger.info(f"✅ Feature engineering completed")
        logger.info(f"  • Final columns: {len(df_engineered.columns)}")
        
        # Clean up
        pipeline.spark.stop()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sample data processing test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and return summary"""
    logger.info("🚀 Starting Spark Pipeline Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Spark Imports", test_spark_imports),
        ("SparkSession Creation", test_spark_session_creation),
        ("Data Pipeline Init", test_data_pipeline_initialization),
        ("Model Trainer Init", test_model_trainer_initialization),
        ("Sample Data Processing", test_sample_data_processing)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'=' * 60}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            results[test_name] = False
            logger.error(f"❌ {test_name}: FAILED with exception: {str(e)}")
    
    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests PASSED! Spark pipeline is ready.")
    else:
        logger.warning("⚠️  Some tests FAILED. Please check the issues above.")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure