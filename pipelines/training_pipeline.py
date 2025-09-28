import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from data_pipeline import data_pipeline

from typing import Dict, Any, Tuple, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_building import CatBoostModelBuilder
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
import mlflow

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_model_config
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def training_pipeline(
                    data_path: str = 'data/hmQOVnDvRN.xls',
                    model_params: Optional[Dict[str, Any]] = None,
                    test_size: float = 0.2, 
                    random_state: int = 42,
                    model_path: str = 'artifacts/models/telco_analysis.joblib',
                    ):
    """
    Training pipeline matching notebook's approach
    """
    logger.info("Starting training pipeline")
    
    # Check if data pipeline outputs exist, run if needed
    if (not os.path.exists(get_data_paths()['X_train'])) or \
        (not os.path.exists(get_data_paths()['X_test'])) or \
        (not os.path.exists(get_data_paths()['y_train'])) or \
        (not os.path.exists(get_data_paths()['y_test'])):
        
        logger.info("Running data pipeline to generate training data")
        data_pipeline()
    else:
        logger.info("Loading existing data artifacts from data pipeline")

    mlflow_tracker = MLflowTracker()
    run_tags = create_mlflow_run_tags('training_pipeline', {
            'model_type': 'XGboost',
            'training_stratergy': 'simple',
            'other_models': 'randomforest'
        })
    run = mlflow_tracker.start_run(run_name='training_pipeline', tags=run_tags)
    
    # Load training and test data
    logger.info("Loading training and test data")
    X_train = pd.read_csv(get_data_paths()['X_train'])
    y_train = pd.read_csv(get_data_paths()['y_train']).squeeze()  # Convert to Series
    X_test = pd.read_csv(get_data_paths()['X_test'])
    y_test = pd.read_csv(get_data_paths()['y_test']).squeeze()   # Convert to Series
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Set default model params if not provided
    if model_params is None:
        model_params = {}
    
    # Build CatBoost model using optimal parameters from notebook
    logger.info("Building CatBoost model with optimal parameters")
    model_builder = CatBoostModelBuilder(**model_params)
    model = model_builder.build_model()
    
    # Train the model
    logger.info("Training the model")
    trainer = ModelTrainer()
    trained_model, train_score = trainer.train_simple(
        model=model,
        X_train=X_train,
        y_train=y_train
    )
    
    # Save the trained model
    logger.info(f"Saving trained model to {model_path}")
    trainer.save_model(trained_model, model_path)
    
    # Evaluate the model (matching notebook evaluation approach)
    logger.info("Evaluating model performance")
    evaluator = ModelEvaluator(trained_model, 'CatBoost')
    evaluation_results = evaluator.evaluate(
        X_test=X_test, 
        y_test=y_test,
        cv_score=None  # No cross-validation score in this simple pipeline
    )
    
    # Prepare metrics for MLflow (filter out non-scalar values)
    metrics_to_log = {}
    for key, value in evaluation_results.items():
        if key in ['cv_auc', 'test_auc', 'overfitting'] and value is not None:
            if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
                metrics_to_log[key] = float(value)
    
    # Add train score to metrics
    metrics_to_log['train_score'] = float(train_score)
    
    model_params = get_model_config().get('model_params', {})
    mlflow_tracker.log_training_metrics(trained_model, metrics_to_log, model_params=model_params)

    mlflow_tracker.end_run()
    
    # Get feature importance if available
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
        feature_importance = evaluator.get_feature_importance(feature_names)
        logger.info("Feature importance analysis completed")
    
    # Log final results
    logger.info("="*60)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("="*60)
    logger.info(f"Training Score: {train_score:.4f}")
    logger.info(f"Test AUC: {evaluation_results['test_auc']:.4f}")
    logger.info(f"Model saved to: {model_path}")
    
    return {
        'model': trained_model,
        'train_score': train_score,
        'evaluation_results': evaluation_results,
        'model_path': model_path
    }

if __name__ == '__main__':
    try:
        # Get model configuration
        model_config = get_model_config()
        # Get CatBoost optimal parameters instead of parameter grid
        model_params = model_config.get('optimal_params', {}).get('catboost', {})
        
        logger.info("Starting training pipeline with parameters:")
        logger.info(f"Model Parameters: {model_params}")
        
        # Run training pipeline
        results = training_pipeline(model_params=model_params)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise



    