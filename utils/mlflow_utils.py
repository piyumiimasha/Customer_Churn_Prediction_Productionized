import os
import logging
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import json


from config import get_mlflow_config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow tracking utilities for experiment management and model versioning"""
    
    def __init__(self):
        self.config = get_mlflow_config()
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Initialize MLflow tracking with configuration"""
        tracking_uri = self.config.get('tracking_uri', 'file:./mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        experiment_name = self.config.get('experiment_name', 'churn_prediction_experiment')
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
                
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run"""
        # Format timestamp for run name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if run_name is None:
            run_name_prefix = self.config.get('run_name_prefix', 'run')
            # Remove underscores and format with timestamp
            run_name_prefix = run_name_prefix.replace('_', ' ')
            run_name = f"{run_name_prefix} | {timestamp}"
        else:
            # Remove underscores from provided run name and append timestamp
            run_name = run_name.replace('_', ' ')
            run_name = f"{run_name} | {timestamp}"
        
        # Merge default tags with provided tags
        default_tags = self.config.get('tags', {})
        if tags:
            default_tags.update(tags)
            
        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        print(f"🎯 MLflow Run Name: {run_name}")
        return run
    
    def log_data_pipeline_metrics(self, dataset_info: Dict[str, Any]):
        """Log data pipeline metrics and artifacts"""
        try:
            # Log dataset metrics
            mlflow.log_metrics({
                'dataset_rows': dataset_info.get('total_rows', 0),
                'training_rows': dataset_info.get('train_rows', 0),
                'test_rows': dataset_info.get('test_rows', 0),
                'num_features': dataset_info.get('num_features', 0),
                'missing_values_count': dataset_info.get('missing_values', 0),
                'outliers_removed': dataset_info.get('outliers_removed', 0)
            })
            
            # Log dataset parameters
            mlflow.log_params({
                'test_size': dataset_info.get('test_size', 0.2),
                'random_state': dataset_info.get('random_state', 42),
                'missing_value_strategy': dataset_info.get('missing_strategy', 'unknown'),
                'outlier_detection_method': dataset_info.get('outlier_method', 'unknown'),
                'feature_encoding_applied': dataset_info.get('encoding_applied', False),
                'feature_scaling_applied': dataset_info.get('scaling_applied', False)
            })
            
            # Log feature names
            if 'feature_names' in dataset_info:
                mlflow.log_param('feature_names', str(dataset_info['feature_names']))
            
            logger.info("Logged data pipeline metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging data pipeline metrics: {e}")
    
    def log_training_metrics(self, model, training_metrics: Dict[str, Any], model_params: Dict[str, Any]):
        """Log training metrics, parameters, and model artifacts"""
        try:
            # Log model parameters
            mlflow.log_params(model_params)
            
            # Filter and log only scalar metrics
            scalar_metrics = {}
            for key, value in training_metrics.items():
                if value is not None:
                    try:
                        # Convert to float if possible
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            scalar_metrics[key] = float(value)
                        elif isinstance(value, np.ndarray) and value.ndim == 0:
                            scalar_metrics[key] = float(value.item())
                        else:
                            logger.warning(f"Skipping non-scalar metric '{key}' with type {type(value)}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert metric '{key}' to float: {e}")
            
            # Log the filtered metrics
            if scalar_metrics:
                mlflow.log_metrics(scalar_metrics)
            
            # Log the model
            artifact_path = self.config.get('artifact_path', 'model')
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=self.config.get('model_registry_name', 'churn_prediction_model')
            )
            
            logger.info("Logged training metrics and model to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging training metrics: {e}")
    
    def log_model_hyperparameters(self, model, additional_params: Optional[Dict[str, Any]] = None):
        """Log comprehensive model hyperparameters"""
        try:
            params_to_log = {}
            
            # Extract model parameters
            if hasattr(model, 'get_params'):
                model_params = model.get_params()
                params_to_log.update(model_params)
            
            # Add additional parameters if provided
            if additional_params:
                params_to_log.update(additional_params)
            
            # Filter out non-serializable parameters
            filtered_params = {}
            for key, value in params_to_log.items():
                try:
                    # Convert numpy types to Python types
                    if isinstance(value, np.integer):
                        filtered_params[key] = int(value)
                    elif isinstance(value, np.floating):
                        filtered_params[key] = float(value)
                    elif isinstance(value, (str, int, float, bool, type(None))):
                        filtered_params[key] = value
                    else:
                        filtered_params[key] = str(value)
                except Exception:
                    filtered_params[key] = str(value)
            
            mlflow.log_params(filtered_params)
            logger.info(f"Logged {len(filtered_params)} hyperparameters to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging hyperparameters: {e}")
    
    def log_comprehensive_evaluation_metrics(self, y_true, y_pred, y_pred_proba=None, 
                                           model_name: str = "model"):
        """Log comprehensive evaluation metrics"""
        try:
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                       f1_score, roc_auc_score, average_precision_score)
            
            # Handle string labels for metrics that support pos_label
            pos_label = 'Yes' if 'Yes' in y_true else 1
            
            # Basic classification metrics
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='weighted')),
                'recall': float(recall_score(y_true, y_pred, average='weighted')),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
                'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
                'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
                'f1_score_macro': float(f1_score(y_true, y_pred, average='macro'))
            }
            
            # Add probability-based metrics if available
            if y_pred_proba is not None:
                # Convert string labels to binary for probability metrics if needed
                if isinstance(y_true.iloc[0] if hasattr(y_true, 'iloc') else y_true[0], str):
                    y_true_binary = (y_true == pos_label).astype(int)
                    metrics.update({
                        'roc_auc': float(roc_auc_score(y_true_binary, y_pred_proba)),
                        'avg_precision_score': float(average_precision_score(y_true_binary, y_pred_proba))
                    })
                else:
                    metrics.update({
                        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
                        'avg_precision_score': float(average_precision_score(y_true, y_pred_proba))
                    })
            
            # Class distribution
            unique, counts = np.unique(y_true, return_counts=True)
            for cls, count in zip(unique, counts):
                metrics[f'true_class_{cls}_count'] = int(count)
                metrics[f'true_class_{cls}_percentage'] = float(count / len(y_true) * 100)
            
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            for cls, count in zip(unique_pred, counts_pred):
                metrics[f'pred_class_{cls}_count'] = int(count)
                metrics[f'pred_class_{cls}_percentage'] = float(count / len(y_pred) * 100)
            
            mlflow.log_metrics(metrics)
            
            # Log classification report as text artifact
            report = classification_report(y_true, y_pred, output_dict=True)
            report_path = f"artifacts/classification_report_{model_name}.json"
            os.makedirs("artifacts", exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            mlflow.log_artifact(report_path, "evaluation")
            
            logger.info(f"Logged {len(metrics)} comprehensive evaluation metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging comprehensive evaluation metrics: {e}")
    
    def log_visualization_plots(self, y_true, y_pred, y_pred_proba=None, 
                              feature_importance=None, feature_names=None,
                              model_name: str = "model"):
        """Generate and log visualization plots"""
        try:
            plots_dir = "artifacts/plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Set style for better-looking plots
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Confusion Matrix
            self._create_confusion_matrix_plot(y_true, y_pred, plots_dir, model_name)
            
            # 2. ROC Curve (if probabilities available)
            if y_pred_proba is not None:
                self._create_roc_curve_plot(y_true, y_pred_proba, plots_dir, model_name)
                
                # 3. Precision-Recall Curve
                self._create_precision_recall_curve_plot(y_true, y_pred_proba, plots_dir, model_name)
                
                # 4. Prediction Distribution
                self._create_prediction_distribution_plot(y_pred_proba, plots_dir, model_name)
            
            # 5. Feature Importance (if available)
            if feature_importance is not None and feature_names is not None:
                self._create_feature_importance_plot(feature_importance, feature_names, 
                                                   plots_dir, model_name)
            
            # 6. Model Performance Summary
            self._create_performance_summary_plot(y_true, y_pred, y_pred_proba, 
                                                plots_dir, model_name)
            
            # Log all plots as artifacts
            for plot_file in os.listdir(plots_dir):
                if plot_file.endswith('.png'):
                    mlflow.log_artifact(os.path.join(plots_dir, plot_file), "visualizations")
            
            logger.info("Logged visualization plots to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging visualization plots: {e}")
    
    def _create_confusion_matrix_plot(self, y_true, y_pred, plots_dir, model_name):
        """Create confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        # Determine labels based on the data
        if isinstance(y_true.iloc[0] if hasattr(y_true, 'iloc') else y_true[0], str):
            labels = ['No Churn', 'Churn'] if 'Yes' in y_true else list(np.unique(y_true))
        else:
            labels = ['No Churn', 'Churn']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, 
                   yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_roc_curve_plot(self, y_true, y_pred_proba, plots_dir, model_name):
        """Create ROC curve plot"""
        # Handle string labels by specifying pos_label
        pos_label = 'Yes' if 'Yes' in y_true else 1
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba, pos_label=pos_label)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_precision_recall_curve_plot(self, y_true, y_pred_proba, plots_dir, model_name):
        """Create Precision-Recall curve plot"""
        # Handle string labels by specifying pos_label
        pos_label = 'Yes' if 'Yes' in y_true else 1
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba, pos_label=pos_label)
        avg_precision = average_precision_score(y_true, y_pred_proba, pos_label=pos_label)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'{model_name} (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/precision_recall_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_prediction_distribution_plot(self, y_pred_proba, plots_dir, model_name):
        """Create prediction probability distribution plot"""
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_proba, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
        plt.xlabel('Predicted Churn Probability')
        plt.ylabel('Frequency')
        plt.title(f'Prediction Probability Distribution - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/prediction_distribution_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_importance_plot(self, feature_importance, feature_names, plots_dir, model_name):
        """Create feature importance plot"""
        # Get top 15 features
        top_n = min(15, len(feature_importance))
        indices = np.argsort(feature_importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), feature_importance[indices], alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_summary_plot(self, y_true, y_pred, y_pred_proba, plots_dir, model_name):
        """Create a comprehensive performance summary plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        # Determine labels based on the data
        if isinstance(y_true.iloc[0] if hasattr(y_true, 'iloc') else y_true[0], str):
            labels = ['No Churn', 'Churn'] if 'Yes' in y_true else list(np.unique(y_true))
        else:
            labels = ['No Churn', 'Churn']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                   xticklabels=labels, 
                   yticklabels=labels)
        axes[0,0].set_title('Confusion Matrix')
        
        # Class Distribution
        unique, counts = np.unique(y_true, return_counts=True)
        class_labels = ['No Churn', 'Churn'] if len(unique) == 2 else [str(u) for u in unique]
        axes[0,1].bar(class_labels, counts, alpha=0.7, color=['skyblue', 'orange'])
        axes[0,1].set_title('True Class Distribution')
        axes[0,1].set_ylabel('Count')
        
        # ROC Curve
        if y_pred_proba is not None:
            # Handle string labels
            pos_label = 'Yes' if 'Yes' in y_true else 1
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba, pos_label=pos_label)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, 
                          label=f'AUC = {auc_score:.3f}')
            axes[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1,0].set_xlabel('False Positive Rate')
            axes[1,0].set_ylabel('True Positive Rate')
            axes[1,0].set_title('ROC Curve')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Prediction Distribution
            axes[1,1].hist(y_pred_proba, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1,1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
            axes[1,1].set_xlabel('Predicted Probability')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('Prediction Distribution')
            axes[1,1].legend()
        
        plt.suptitle(f'Model Performance Summary - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/performance_summary_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def log_evaluation_metrics(self, evaluation_metrics: Dict[str, Any], confusion_matrix_path: Optional[str] = None):
        """Log evaluation metrics and artifacts"""
        try:
            # Log evaluation metrics
            if 'metrics' in evaluation_metrics:
                mlflow.log_metrics(evaluation_metrics['metrics'])
            
            # Log confusion matrix if provided
            if confusion_matrix_path and os.path.exists(confusion_matrix_path):
                mlflow.log_artifact(confusion_matrix_path, "evaluation")
            
            logger.info("Logged evaluation metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging evaluation metrics: {e}")
    
    def log_inference_metrics(self, predictions: np.ndarray, probabilities: Optional[np.ndarray] = None, 
                            input_data_info: Optional[Dict[str, Any]] = None):
        """Log inference metrics and results"""
        try:
            # Log inference metrics
            inference_metrics = {
                'num_predictions': len(predictions),
                'avg_prediction': float(np.mean(predictions)),
                'prediction_distribution_churn': int(np.sum(predictions)),
                'prediction_distribution_retain': int(len(predictions) - np.sum(predictions))
            }
            
            if probabilities is not None:
                inference_metrics.update({
                    'avg_churn_probability': float(np.mean(probabilities)),
                    'high_risk_predictions': int(np.sum(probabilities > 0.7)),
                    'medium_risk_predictions': int(np.sum((probabilities > 0.5) & (probabilities <= 0.7))),
                    'low_risk_predictions': int(np.sum(probabilities <= 0.5))
                })
            
            mlflow.log_metrics(inference_metrics)
            
            # Log input data info if provided
            if input_data_info:
                mlflow.log_params(input_data_info)
            
            logger.info("Logged inference metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging inference metrics: {e}")
    
    def load_model_from_registry(self, model_name: Optional[str] = None, 
                               version: Optional[Union[int, str]] = None, 
                               stage: Optional[str] = None):
        """Load model from MLflow Model Registry"""
        try:
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'churn_prediction_model')
            
            if stage:
                model_uri = f"models:/{model_name}/{stage}"
            elif version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from MLflow registry: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from MLflow registry: {e}")
            return None
    
    def get_latest_model_version(self, model_name: Optional[str] = None) -> Optional[str]:
        """Get the latest version of a registered model"""
        try:
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'churn_prediction_model')
            
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            
            if latest_version:
                return latest_version[0].version
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return None
    
    def transition_model_stage(self, model_name: Optional[str] = None, 
                             version: Optional[str] = None, 
                             stage: str = "Staging"):
        """Transition model to a specific stage"""
        try:
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'churn_prediction_model')
            
            if version is None:
                version = self.get_latest_model_version(model_name)
            
            if version:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage
                )
                logger.info(f"Transitioned model {model_name} version {version} to {stage}")
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
    
    def end_run(self):
        """End the current MLflow run"""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")




def setup_mlflow_autolog():
    """Setup MLflow autologging for supported frameworks"""
    mlflow_config = get_mlflow_config()
    if mlflow_config.get('autolog', True):
        mlflow.sklearn.autolog()
        logger.info("MLflow autologging enabled for scikit-learn")




def create_mlflow_run_tags(pipeline_type: str, additional_tags: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Create standardized tags for MLflow runs"""
    tags = {
        'pipeline_type': pipeline_type,
        'timestamp': datetime.now().isoformat(),
    }
    
    if additional_tags:
        tags.update(additional_tags)
    
    return tags
