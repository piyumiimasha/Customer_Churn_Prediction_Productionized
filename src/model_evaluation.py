import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import warnings
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(
        self,
        model,
        model_name: str
    ):
        self.model = model
        self.model_name = model_name
        self.evaluation_results = {}

    def evaluate(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        cv_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance matching notebook's approach
        
        Args:
            X_test: Test features
            y_test: Test labels  
            cv_score: Cross-validation score from GridSearchCV
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {self.model_name} on test set...")
        
        # Make predictions (same as notebook)
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics (same as notebook)
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate overfitting if CV score provided
        overfitting = None
        if cv_score is not None:
            overfitting = cv_score - test_auc
        
        # Store results matching notebook structure
        self.evaluation_results = {
            'cv_auc': cv_score,
            'test_auc': test_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'overfitting': overfitting
        }
        
        # Print results exactly like notebook
        if cv_score is not None:
            print(f"CV AUC: {cv_score:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        if overfitting is not None:
            print(f"Overfitting: {overfitting:.4f}")
        
        print(f"\nClassification Report for {self.model_name}:")
        print(classification_report(y_test, y_pred))
        
        return self.evaluation_results

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importances exactly as done in notebook
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance rankings
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        # Get feature importances (same as notebook)
        importances = self.model.feature_importances_
        
        # Create DataFrame (same as notebook)
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Print top features (same as notebook)
        print(f"\nTop 10 Feature Importances for {self.model_name}:")
        for i, (_, row) in enumerate(feature_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<35}: {row['importance']:.4f}")
        
        return feature_df