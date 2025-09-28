import joblib, os
from typing import Dict, Any
from datetime import datetime
from xgboost import XGBClassifier
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
import catboost as cb


class BaseModelBuilder(ABC):
    def __init__(
                self,
                model_name:str,
                **kwargs
                ):
        self.model_name = model_name
        self.model = None 
        self.model_params = kwargs
    
    @abstractmethod
    def build_model(self):
        pass 


    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save. Build the model first.")
        
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError("Can't load. File not found.")
        
        self.model = joblib.load(filepath)


class RandomForestModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
                        'criterion': 'entropy',
                        'max_depth': 20,
                        'max_features': 'sqrt',
                        'min_samples_split': 2,
                        'n_estimators': 300,
                        'random_state': 42,
                        'n_jobs': -1
                        }
        default_params.update(kwargs)
        super().__init__('RandomForest', **default_params)


    def build_model(self):
        self.model = RandomForestClassifier(**self.model_params)
        return self.model
    
class XGboostModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
                        'colsample_bytree': 0.9,
                        'learning_rate': 0.1,
                        'max_depth': 6,
                        'n_estimators': 200,
                        'subsample': 0.9,
                        'random_state': 42,
                        'eval_metric': 'logloss',
                        'n_jobs': -1
                        }
        default_params.update(kwargs)
        super().__init__('XGboost', **default_params)


    def build_model(self):
        self.model = XGBClassifier(**self.model_params)
        return self.model


class CatBoostModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
                        'border_count': 32,
                        'depth': 6,
                        'iterations': 300,
                        'l2_leaf_reg': 1,
                        'learning_rate': 0.1,
                        'random_state': 42,
                        'verbose': False
                        }
        default_params.update(kwargs)
        super().__init__('CatBoost', **default_params)

    def build_model(self):
        self.model = cb.CatBoostClassifier(**self.model_params)
        return self.model

# rf= CatBoostModelBuilder()
# rf_model = rf.build_model()
# print(f"Model created: {rf_model}")
# print(f"Model parameters: {rf_model.get_params()}")