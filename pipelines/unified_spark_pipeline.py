"""
Unified Spark Pipeline for Customer Churn Prediction
Combines data processing and model training using PySpark for distributed processing
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# PySpark imports
from pyspark.sql import SparkSession

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from spark_data_pipeline import SparkDataPipeline
from spark_model_trainer import SparkModelTrainer
from spark_session import create_spark_session, stop_spark_session

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_model_config
from mlflow_utils import MLflowTracker, create_mlflow_run_tags

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedSparkPipeline:
    """
    Unified pipeline that orchestrates data processing and model training using Spark
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize Unified Spark Pipeline
        
        Args:
            spark: Existing SparkSession or None to create new one
        """
        self.spark = spark or self._create_spark_session()
        
        # Initialize pipeline components
        self.data_pipeline = SparkDataPipeline(self.spark)
        self.model_trainer = SparkModelTrainer(self.spark)
        
        # MLflow integration
        self.mlflow_tracker = MLflowTracker()
        
        logger.info("✓ UnifiedSparkPipeline initialized")
    
    def _create_spark_session(self) -> SparkSession:
        """Create optimized SparkSession for the complete pipeline"""
        config_options = {
            # Adaptive Query Execution
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.adaptive.localShuffleReader.enabled": "true",
            
            # Arrow optimization
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
            "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
            
            # ML optimizations
            "spark.ml.tuning.parallelism": "4",
            "spark.sql.autoBroadcastJoinThreshold": "10485760",  # 10MB
            
            # Serialization and performance
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.shuffle.partitions": "200",
            "spark.sql.parquet.compression.codec": "snappy",
            "spark.sql.parquet.filterPushdown": "true",
            "spark.sql.csv.parser.columnPruning.enabled": "true",
            
            # Memory management
            "spark.executor.memoryFraction": "0.8",
            "spark.storage.memoryFraction": "0.3"
        }
        
        return create_spark_session(
            app_name="UnifiedChurnPredictionPipeline",
            master="local[*]",
            config_options=config_options
        )
    
    def run_data_processing(self, data_path: str, output_dir: str = "artifacts/spark_pipeline") -> Dict[str, Any]:
        """
        Run the data processing pipeline
        
        Args:
            data_path: Path to raw data
            output_dir: Output directory for processed data
            
        Returns:
            Dict: Data processing results
        """
        try:
            logger.info("🔄 Starting data processing pipeline...")
            
            # Create data output directory
            data_output_dir = f"{output_dir}/data"
            
            # Run data pipeline
            data_results = self.data_pipeline.run_complete_pipeline(data_path, data_output_dir)
            
            logger.info("✅ Data processing completed successfully")
            return data_results
            
        except Exception as e:
            logger.error(f"❌ Data processing failed: {str(e)}")
            raise
    
    def run_model_training(self, data_results: Dict[str, Any], 
                          output_dir: str = "artifacts/spark_pipeline") -> Dict[str, Any]:
        """
        Run the model training pipeline
        
        Args:
            data_results: Results from data processing pipeline
            output_dir: Output directory for models
            
        Returns:
            Dict: Model training results
        """
        try:
            logger.info("🤖 Starting model training pipeline...")
            
            # Create model output directory
            model_output_dir = f"{output_dir}/models"
            
            # Get data paths from processing results
            train_path = data_results['train_data_path']
            test_path = data_results['test_data_path']
            
            # Run model training
            training_results = self.model_trainer.run_complete_training_pipeline(
                train_path, test_path, model_output_dir
            )
            
            logger.info("✅ Model training completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            raise
    
    def log_pipeline_to_mlflow(self, data_results: Dict[str, Any], 
                              training_results: Dict[str, Any], 
                              pipeline_config: Dict[str, Any]) -> None:
        """
        Log complete pipeline results to MLflow
        
        Args:
            data_results: Data processing results
            training_results: Model training results
            pipeline_config: Pipeline configuration
        """
        try:
            logger.info("📊 Logging pipeline results to MLflow...")
            
            # Create run tags
            run_tags = create_mlflow_run_tags('unified_spark_pipeline', {
                'pipeline_type': 'unified_spark',
                'data_processing': 'spark_dataframe',
                'model_training': 'spark_mllib',
                'distributed': 'true',
                'spark_version': self.spark.version
            })
            
            # Start MLflow run
            run = self.mlflow_tracker.start_run(
                run_name=f'unified_spark_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                tags=run_tags
            )
            
            # Log pipeline configuration
            self.mlflow_tracker.log_params({
                'spark_version': self.spark.version,
                'spark_master': self.spark.sparkContext.master,
                'default_parallelism': self.spark.sparkContext.defaultParallelism,
                'train_samples': data_results['train_count'],
                'test_samples': data_results['test_count'],
                'feature_columns_count': len(data_results['feature_columns']),
                'categorical_columns_count': len(data_results['categorical_columns']),
                'numerical_columns_count': len(data_results['numerical_columns']),
                'models_trained': len(training_results['models_trained'])
            })
            
            # Log data processing metrics
            data_quality = data_results['data_quality']
            self.mlflow_tracker.log_metrics({
                'total_rows': data_quality['total_rows'],
                'total_columns': data_quality['total_columns'],
                'duplicate_rows': data_quality['duplicate_rows'],
                'missing_values_columns': len([col for col, info in data_quality['missing_values'].items() 
                                             if info['count'] > 0])
            })
            
            # Log best model results
            best_model_results = None
            best_roc_auc = 0
            
            for model_name, results in training_results['evaluation_results'].items():
                model_roc_auc = results['test_roc_auc']
                if model_roc_auc > best_roc_auc:
                    best_roc_auc = model_roc_auc
                    best_model_results = results
                
                # Log individual model metrics
                self.mlflow_tracker.log_metrics({
                    f'{model_name}_roc_auc': results['test_roc_auc'],
                    f'{model_name}_accuracy': results['test_accuracy'],
                    f'{model_name}_f1_score': results['test_f1_score'],
                    f'{model_name}_precision': results['test_precision'],
                    f'{model_name}_recall': results['test_recall']
                })
            
            # Log best model metrics
            if best_model_results:
                self.mlflow_tracker.log_metrics({
                    'best_roc_auc': best_model_results['test_roc_auc'],
                    'best_accuracy': best_model_results['test_accuracy'],
                    'best_f1_score': best_model_results['test_f1_score'],
                    'best_precision': best_model_results['test_precision'],
                    'best_recall': best_model_results['test_recall']
                })
            
            # Log artifacts
            import mlflow
            if os.path.exists(training_results['comparison_results']):
                mlflow.log_artifact(training_results['comparison_results'], "model_comparison")
            
            # End MLflow run
            self.mlflow_tracker.end_run()
            
            logger.info("✅ Pipeline results logged to MLflow successfully")
            
        except Exception as e:
            logger.error(f"❌ Error logging to MLflow: {str(e)}")
            # Don't raise exception here to avoid stopping the pipeline\n    \n    def run_complete_pipeline(self, data_path: str, \n                             output_dir: str = \"artifacts/spark_pipeline\",\n                             log_to_mlflow: bool = True) -> Dict[str, Any]:\n        \"\"\"Run the complete unified Spark pipeline\n        \n        Args:\n            data_path: Path to raw data\n            output_dir: Output directory for all artifacts\n            log_to_mlflow: Whether to log results to MLflow\n            \n        Returns:\n            Dict: Complete pipeline results\n        \"\"\"\n        try:\n            start_time = datetime.now()\n            logger.info(\"🚀 Starting Unified Spark Pipeline for Customer Churn Prediction\")\n            logger.info(f\"📊 Spark Session Info:\")\n            logger.info(f\"  • Version: {self.spark.version}\")\n            logger.info(f\"  • Master: {self.spark.sparkContext.master}\")\n            logger.info(f\"  • Default Parallelism: {self.spark.sparkContext.defaultParallelism}\")\n            \n            # Create main output directory\n            os.makedirs(output_dir, exist_ok=True)\n            \n            # 1. Data Processing Pipeline\n            logger.info(\"\\n\" + \"=\"*80)\n            logger.info(\"PHASE 1: DATA PROCESSING\")\n            logger.info(\"=\"*80)\n            \n            data_results = self.run_data_processing(data_path, output_dir)\n            \n            # 2. Model Training Pipeline\n            logger.info(\"\\n\" + \"=\"*80)\n            logger.info(\"PHASE 2: MODEL TRAINING\")\n            logger.info(\"=\"*80)\n            \n            training_results = self.run_model_training(data_results, output_dir)\n            \n            # 3. Pipeline Summary\n            end_time = datetime.now()\n            execution_time = (end_time - start_time).total_seconds()\n            \n            pipeline_results = {\n                'execution_time_seconds': execution_time,\n                'start_time': start_time.isoformat(),\n                'end_time': end_time.isoformat(),\n                'data_processing_results': data_results,\n                'model_training_results': training_results,\n                'spark_info': {\n                    'version': self.spark.version,\n                    'master': self.spark.sparkContext.master,\n                    'default_parallelism': self.spark.sparkContext.defaultParallelism\n                },\n                'output_directory': output_dir\n            }\n            \n            # 4. MLflow Logging\n            if log_to_mlflow:\n                logger.info(\"\\n\" + \"=\"*80)\n                logger.info(\"PHASE 3: MLFLOW LOGGING\")\n                logger.info(\"=\"*80)\n                \n                pipeline_config = {\n                    'data_path': data_path,\n                    'output_dir': output_dir,\n                    'execution_time': execution_time\n                }\n                \n                self.log_pipeline_to_mlflow(data_results, training_results, pipeline_config)\n            \n            # 5. Final Summary\n            logger.info(\"\\n\" + \"=\"*80)\n            logger.info(\"PIPELINE COMPLETED SUCCESSFULLY!\")\n            logger.info(\"=\"*80)\n            logger.info(f\"⏱️  Total execution time: {execution_time:.2f} seconds\")\n            logger.info(f\"📊 Data processed: {data_results['train_count'] + data_results['test_count']:,} samples\")\n            logger.info(f\"🤖 Models trained: {len(training_results['models_trained'])}\")\n            logger.info(f\"🏆 Best model saved: {training_results['best_model_path']}\")\n            logger.info(f\"📁 Output directory: {output_dir}\")\n            \n            return pipeline_results\n            \n        except Exception as e:\n            logger.error(f\"❌ Unified pipeline failed: {str(e)}\")\n            raise\n        finally:\n            # Clean up\n            self.spark.catalog.clearCache()\n    \n    def stop(self):\n        \"\"\"Stop the Spark session\"\"\"\n        try:\n            if self.spark:\n                stop_spark_session(self.spark)\n                logger.info(\"✅ Spark session stopped\")\n        except Exception as e:\n            logger.error(f\"❌ Error stopping Spark session: {str(e)}\")\n\n\ndef main():\n    \"\"\"Main function to run the unified Spark pipeline\"\"\"\n    pipeline = None\n    try:\n        # Initialize unified pipeline\n        pipeline = UnifiedSparkPipeline()\n        \n        # Configuration\n        data_path = \"data/hmQOVnDvRN.xls\"  # Update path as needed\n        output_dir = \"artifacts/spark_pipeline\"\n        \n        # Run complete pipeline\n        results = pipeline.run_complete_pipeline(\n            data_path=data_path,\n            output_dir=output_dir,\n            log_to_mlflow=True\n        )\n        \n        print(f\"\\n🎉 Unified Spark Pipeline completed successfully!\")\n        print(f\"⏱️  Execution time: {results['execution_time_seconds']:.2f} seconds\")\n        print(f\"📊 Total samples processed: {results['data_processing_results']['train_count'] + results['data_processing_results']['test_count']:,}\")\n        print(f\"🤖 Models trained: {len(results['model_training_results']['models_trained'])}\")\n        print(f\"📁 Results saved to: {output_dir}\")\n        \n    except Exception as e:\n        logger.error(f\"Pipeline execution failed: {str(e)}\")\n        raise\n    finally:\n        # Clean shutdown\n        if pipeline:\n            pipeline.stop()\n\n\nif __name__ == \"__main__\":\n    main()" </n]