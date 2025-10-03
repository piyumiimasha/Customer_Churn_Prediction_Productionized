"""
Spark-based Model Training Pipeline for Customer Churn Prediction
Uses PySpark MLlib for distributed model training and evaluation
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.ml import Pipeline, Model
from pyspark.ml.classification import (
    GBTClassifier, RandomForestClassifier, LogisticRegression, 
    DecisionTreeClassifier, NaiveBayes
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.param.shared import *

# Custom imports
from spark_session import create_spark_session, configure_spark_for_ml
from spark_data_pipeline import SparkDataPipeline

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config, get_spark_config
from mlflow_utils import MLflowTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SparkModelTrainer:
    """
    Comprehensive Spark-based model training pipeline using MLlib
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize Spark Model Trainer
        
        Args:
            spark: Existing SparkSession or None to create new one
        """
        self.spark = spark or self._create_spark_session()
        self.spark = configure_spark_for_ml(self.spark)
        
        # Model registry
        self.models = {}
        self.trained_models = {}
        self.evaluation_results = {}
        
        # MLflow integration
        self.mlflow_tracker = MLflowTracker()
        
        logger.info("✓ SparkModelTrainer initialized")
    
    def _create_spark_session(self) -> SparkSession:
        """Create optimized SparkSession for ML training"""
        config_options = {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.ml.tuning.parallelism": "4",
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
        }
        
        return create_spark_session(
            app_name="ChurnPrediction-ModelTraining",
            master="local[*]",
            config_options=config_options
        )
    
    def load_processed_data(self, train_path: str, test_path: str) -> Tuple[DataFrame, DataFrame]:
        """
        Load processed training and test data
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            
        Returns:
            Tuple[DataFrame, DataFrame]: Train and test DataFrames
        """
        try:
            logger.info("📖 Loading processed data...")
            
            train_df = self.spark.read.parquet(train_path)
            test_df = self.spark.read.parquet(test_path)
            
            # Cache for better performance
            train_df.cache()
            test_df.cache()
            
            train_count = train_df.count()
            test_count = test_df.count()
            
            logger.info(f"✓ Data loaded successfully")
            logger.info(f"  • Training: {train_count:,} samples")
            logger.info(f"  • Test: {test_count:,} samples")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"✗ Error loading processed data: {str(e)}")
            raise
    
    def setup_models(self) -> Dict[str, Any]:
        """
        Setup different ML models with their configurations
        
        Returns:
            Dict: Dictionary of model configurations
        """
        try:
            logger.info("⚙️ Setting up ML models...")
            
            self.models = {
                'gradient_boosting': {
                    'model': GBTClassifier(
                        featuresCol="features",
                        labelCol="label",
                        predictionCol="prediction",
                        probabilityCol="probability",
                        rawPredictionCol="rawPrediction",
                        maxDepth=6,
                        maxBins=32,
                        maxIter=100,
                        stepSize=0.1,
                        subsamplingRate=1.0,
                        featureSubsetStrategy="auto",
                        seed=42
                    ),
                    'param_grid': ParamGridBuilder() \
                        .addGrid(GBTClassifier.maxDepth, [4, 6, 8]) \
                        .addGrid(GBTClassifier.maxIter, [50, 100, 150]) \
                        .addGrid(GBTClassifier.stepSize, [0.05, 0.1, 0.2]) \
                        .build()
                },
                
                'random_forest': {
                    'model': RandomForestClassifier(
                        featuresCol="features",
                        labelCol="label",
                        predictionCol="prediction",
                        probabilityCol="probability",
                        rawPredictionCol="rawPrediction",
                        numTrees=100,
                        maxDepth=6,
                        maxBins=32,
                        minInstancesPerNode=1,
                        minInfoGain=0.0,
                        subsamplingRate=1.0,
                        featureSubsetStrategy="auto",
                        seed=42
                    ),
                    'param_grid': ParamGridBuilder() \
                        .addGrid(RandomForestClassifier.numTrees, [50, 100, 200]) \
                        .addGrid(RandomForestClassifier.maxDepth, [4, 6, 8]) \
                        .addGrid(RandomForestClassifier.featureSubsetStrategy, ["auto", "sqrt", "log2"]) \
                        .build()
                },
                
                'logistic_regression': {
                    'model': LogisticRegression(
                        featuresCol="features",
                        labelCol="label",
                        predictionCol="prediction",
                        probabilityCol="probability",
                        rawPredictionCol="rawPrediction",
                        maxIter=100,
                        regParam=0.01,
                        elasticNetParam=0.0,
                        threshold=0.5,
                        family="binomial"
                    ),
                    'param_grid': ParamGridBuilder() \
                        .addGrid(LogisticRegression.regParam, [0.01, 0.1, 1.0]) \
                        .addGrid(LogisticRegression.elasticNetParam, [0.0, 0.5, 1.0]) \
                        .addGrid(LogisticRegression.maxIter, [50, 100, 200]) \
                        .build()
                },
                
                'decision_tree': {
                    'model': DecisionTreeClassifier(
                        featuresCol="features",
                        labelCol="label",
                        predictionCol="prediction",
                        probabilityCol="probability",
                        rawPredictionCol="rawPrediction",
                        maxDepth=6,
                        maxBins=32,
                        minInstancesPerNode=1,
                        minInfoGain=0.0,
                        seed=42
                    ),
                    'param_grid': ParamGridBuilder() \
                        .addGrid(DecisionTreeClassifier.maxDepth, [4, 6, 8, 10]) \
                        .addGrid(DecisionTreeClassifier.minInstancesPerNode, [1, 5, 10]) \
                        .build()
                }
            }
            
            logger.info(f"✓ {len(self.models)} models configured")
            for model_name in self.models.keys():
                logger.info(f"  • {model_name}")
            
            return self.models
            
        except Exception as e:
            logger.error(f"✗ Error setting up models: {str(e)}")
            raise
    
    def create_evaluators(self) -> Tuple[BinaryClassificationEvaluator, MulticlassClassificationEvaluator]:
        """
        Create evaluators for model assessment
        
        Returns:
            Tuple: Binary and multiclass evaluators
        """
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        multiclass_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        return binary_evaluator, multiclass_evaluator
    
    def train_model_with_cv(self, model_name: str, train_df: DataFrame, 
                           cv_folds: int = 3, validation_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Train model with cross-validation or train-validation split
        
        Args:
            model_name: Name of the model to train
            train_df: Training DataFrame
            cv_folds: Number of CV folds (if > 2, use CV; otherwise use train-validation split)
            validation_ratio: Ratio for train-validation split
            
        Returns:
            Dict: Training results including best model and metrics
        """
        try:
            logger.info(f"🚀 Training {model_name} with hyperparameter tuning...")
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found in configured models")
            
            model_config = self.models[model_name]
            model = model_config['model']
            param_grid = model_config['param_grid']
            
            # Create evaluators
            binary_evaluator, multiclass_evaluator = self.create_evaluators()
            
            # Choose between CV and Train-Validation Split based on data size
            train_count = train_df.count()
            
            if cv_folds > 2 and train_count > 10000:
                # Use Cross Validation for larger datasets
                logger.info(f"📊 Using {cv_folds}-fold Cross Validation")
                
                cv = CrossValidator(
                    estimator=model,
                    estimatorParamMaps=param_grid,
                    evaluator=binary_evaluator,
                    numFolds=cv_folds,
                    parallelism=2,
                    seed=42
                )
                
                cv_model = cv.fit(train_df)
                best_model = cv_model.bestModel
                
                # Get CV metrics
                avg_metrics = cv_model.avgMetrics
                best_metric = max(avg_metrics)
                best_idx = avg_metrics.index(best_metric)
                
                training_summary = {
                    'method': 'cross_validation',
                    'cv_folds': cv_folds,
                    'best_cv_score': best_metric,
                    'avg_cv_scores': avg_metrics,
                    'best_params': param_grid[best_idx]
                }
                
            else:
                # Use Train-Validation Split for smaller datasets or when CV is not preferred
                logger.info(f"📊 Using Train-Validation Split ({validation_ratio:.1%} train)")
                
                tvs = TrainValidationSplit(
                    estimator=model,
                    estimatorParamMaps=param_grid,
                    evaluator=binary_evaluator,
                    trainRatio=validation_ratio,
                    parallelism=2,
                    seed=42
                )
                
                tvs_model = tvs.fit(train_df)
                best_model = tvs_model.bestModel
                
                # Get validation metrics
                validation_metrics = tvs_model.validationMetrics
                best_metric = max(validation_metrics)
                best_idx = validation_metrics.index(best_metric)
                
                training_summary = {
                    'method': 'train_validation_split',
                    'train_ratio': validation_ratio,
                    'best_validation_score': best_metric,
                    'validation_scores': validation_metrics,
                    'best_params': param_grid[best_idx]
                }
            
            # Store the best model
            self.trained_models[model_name] = best_model
            
            logger.info(f"✓ {model_name} training completed")
            logger.info(f"  • Best score: {best_metric:.4f}")
            
            return {
                'model': best_model,
                'training_summary': training_summary,
                'model_name': model_name
            }
            
        except Exception as e:
            logger.error(f"✗ Error training {model_name}: {str(e)}")
            raise
    
    def evaluate_model(self, model_name: str, model: Any, test_df: DataFrame) -> Dict[str, Any]:
        """
        Comprehensive model evaluation on test data
        
        Args:
            model_name: Name of the model
            model: Trained model
            test_df: Test DataFrame
            
        Returns:
            Dict: Evaluation metrics and results
        """
        try:
            logger.info(f"📊 Evaluating {model_name} on test data...")
            
            # Make predictions
            predictions_df = model.transform(test_df)
            
            # Create evaluators
            binary_evaluator, multiclass_evaluator = self.create_evaluators()
            
            # Calculate metrics
            roc_auc = binary_evaluator.evaluate(predictions_df)
            
            # Change evaluator metric for different measurements
            binary_evaluator.setMetricName("areaUnderPR")
            pr_auc = binary_evaluator.evaluate(predictions_df)
            
            accuracy = multiclass_evaluator.evaluate(predictions_df)
            
            # Calculate additional metrics using confusion matrix
            multiclass_evaluator.setMetricName("f1")
            f1_score = multiclass_evaluator.evaluate(predictions_df)
            
            multiclass_evaluator.setMetricName("weightedPrecision")
            precision = multiclass_evaluator.evaluate(predictions_df)
            
            multiclass_evaluator.setMetricName("weightedRecall")
            recall = multiclass_evaluator.evaluate(predictions_df)
            
            # Get prediction distribution
            prediction_counts = predictions_df.groupBy("prediction").count().collect()
            prediction_distribution = {row['prediction']: row['count'] for row in prediction_counts}
            
            # Get actual distribution
            actual_counts = predictions_df.groupBy("label").count().collect()
            actual_distribution = {row['label']: row['count'] for row in actual_counts}
            
            # Calculate confusion matrix manually
            confusion_matrix = predictions_df.groupBy("label", "prediction").count().collect()
            cm_dict = {(row['label'], row['prediction']): row['count'] for row in confusion_matrix}
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'featureImportances'):
                feature_importance = model.featureImportances.toArray().tolist()
            
            evaluation_results = {
                'model_name': model_name,
                'test_roc_auc': roc_auc,
                'test_pr_auc': pr_auc,
                'test_accuracy': accuracy,
                'test_f1_score': f1_score,
                'test_precision': precision,
                'test_recall': recall,
                'prediction_distribution': prediction_distribution,
                'actual_distribution': actual_distribution,
                'confusion_matrix': cm_dict,
                'feature_importance': feature_importance,
                'test_samples': test_df.count()
            }
            
            self.evaluation_results[model_name] = evaluation_results
            
            logger.info(f"✓ {model_name} evaluation completed")
            logger.info(f"  • ROC AUC: {roc_auc:.4f}")
            logger.info(f"  • Accuracy: {accuracy:.4f}")
            logger.info(f"  • F1 Score: {f1_score:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"✗ Error evaluating {model_name}: {str(e)}")
            raise
    
    def compare_models(self) -> DataFrame:
        """
        Compare all trained models and create a comparison DataFrame
        
        Returns:
            DataFrame: Model comparison results
        """
        try:
            logger.info("📈 Comparing model performances...")
            
            if not self.evaluation_results:
                raise ValueError("No models have been evaluated yet")
            
            # Prepare comparison data
            comparison_data = []
            for model_name, results in self.evaluation_results.items():
                comparison_data.append({
                    'model_name': model_name,
                    'roc_auc': results['test_roc_auc'],
                    'pr_auc': results['test_pr_auc'],
                    'accuracy': results['test_accuracy'],
                    'f1_score': results['test_f1_score'],
                    'precision': results['test_precision'],
                    'recall': results['test_recall']
                })
            
            # Create comparison DataFrame
            comparison_df = self.spark.createDataFrame(comparison_data)
            
            # Show comparison
            logger.info("📊 Model Performance Comparison:")
            comparison_df.show(truncate=False)
            
            # Find best model by ROC AUC
            best_model_row = comparison_df.orderBy(desc("roc_auc")).first()
            best_model_name = best_model_row['model_name']
            
            logger.info(f"🏆 Best model: {best_model_name} (ROC AUC: {best_model_row['roc_auc']:.4f})")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"✗ Error comparing models: {str(e)}")
            raise
    
    def save_best_model(self, output_path: str, metric: str = 'test_roc_auc') -> str:
        """
        Save the best performing model
        
        Args:
            output_path: Path to save the model
            metric: Metric to use for selecting best model
            
        Returns:
            str: Path where the model was saved
        """
        try:
            logger.info(f"💾 Saving best model based on {metric}...")
            
            if not self.evaluation_results:
                raise ValueError("No models have been evaluated yet")
            
            # Find best model
            best_model_name = max(self.evaluation_results.keys(), 
                                key=lambda x: self.evaluation_results[x][metric])
            best_model = self.trained_models[best_model_name]
            best_score = self.evaluation_results[best_model_name][metric]
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save model
            best_model.write().overwrite().save(output_path)
            
            # Save model metadata
            metadata = {
                'model_name': best_model_name,
                'best_metric': metric,
                'best_score': best_score,
                'model_path': output_path,
                'evaluation_results': self.evaluation_results[best_model_name]
            }
            
            import json
            with open(f"{output_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✓ Best model saved: {best_model_name}")
            logger.info(f"  • Model path: {output_path}")
            logger.info(f"  • {metric}: {best_score:.4f}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"✗ Error saving best model: {str(e)}")
            raise
    
    def run_complete_training_pipeline(self, train_path: str, test_path: str, 
                                     output_dir: str = "artifacts/spark_models") -> Dict[str, Any]:
        """
        Run the complete Spark-based model training pipeline
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            output_dir: Directory to save models and results
            
        Returns:
            Dict: Complete pipeline results
        """
        try:
            logger.info("🚀 Starting complete Spark model training pipeline")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Load processed data
            train_df, test_df = self.load_processed_data(train_path, test_path)
            
            # 2. Setup models
            self.setup_models()
            
            # 3. Train all models
            training_results = {}
            for model_name in self.models.keys():
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_name.upper()}")
                logger.info(f"{'='*60}")
                
                training_result = self.train_model_with_cv(model_name, train_df)
                training_results[model_name] = training_result
                
                # Evaluate model
                evaluation_result = self.evaluate_model(
                    model_name, 
                    training_result['model'], 
                    test_df
                )
            
            # 4. Compare models
            comparison_df = self.compare_models()
            
            # 5. Save best model
            best_model_path = self.save_best_model(f"{output_dir}/best_model")
            
            # 6. Save comparison results
            comparison_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
                f"{output_dir}/model_comparison.csv"
            )
            
            # Prepare final results
            results = {
                'training_results': training_results,
                'evaluation_results': self.evaluation_results,
                'best_model_path': best_model_path,
                'comparison_results': f"{output_dir}/model_comparison.csv",
                'models_trained': list(self.models.keys()),
                'train_samples': train_df.count(),
                'test_samples': test_df.count()
            }
            
            logger.info("✅ Complete Spark model training pipeline finished successfully!")
            logger.info(f"📊 Results summary:")
            logger.info(f"  • Models trained: {len(self.models)}")
            logger.info(f"  • Best model saved: {best_model_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            raise
        finally:
            # Clean up cached DataFrames
            self.spark.catalog.clearCache()


def main():
    """Main function to run the Spark model training pipeline"""
    try:
        # Initialize trainer
        trainer = SparkModelTrainer()
        
        # Run pipeline
        train_path = "artifacts/spark_data/train_data.parquet"
        test_path = "artifacts/spark_data/test_data.parquet"
        results = trainer.run_complete_training_pipeline(train_path, test_path)
        
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"Models trained: {len(results['models_trained'])}")
        print(f"Best model: {results['best_model_path']}")
        
    except Exception as e:
        logger.error(f"Training pipeline execution failed: {str(e)}")
        raise
    finally:
        # Stop Spark session
        if 'trainer' in locals() and trainer.spark:
            trainer.spark.stop()


if __name__ == "__main__":
    main()