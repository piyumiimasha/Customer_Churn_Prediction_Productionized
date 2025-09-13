import yaml
import os
from typing import Dict, Any

class Config:
    """Configuration manager for the project"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._config = None
        self.load_config()
    
    def load_config(self, config_path: str = "config.yml"):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Error loading config file: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        
        Args:
            key: Configuration key in dot notation (e.g., 'data_paths.raw_data')
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        try:
            value = self._config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        Get a nested configuration value
        
        Args:
            *keys: Sequence of keys to traverse
            default: Default value if path is not found
            
        Returns:
            Configuration value
        """
        try:
            value = self._config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    @property
    def raw_data_path(self) -> str:
        """Get the raw data file path"""
        return self.get('data_paths.raw_data')
    
    @property
    def artifacts_dir(self) -> str:
        """Get the artifacts directory path"""
        return self.get('data_paths.artifacts_dir')
    
    @property
    def data_artifacts_dir(self) -> str:
        """Get the data artifacts directory path"""
        return self.get('data_paths.data_artifacts_dir')
    
    @property
    def model_artifacts_dir(self) -> str:
        """Get the model artifacts directory path"""
        return self.get('data_paths.model_artifacts_dir')
    
    @property
    def feature_groups(self) -> Dict[str, list]:
        """Get feature group definitions"""
        return self.get('columns.feature_groups', {})
    
    @property
    def drop_columns(self) -> list:
        """Get list of columns to drop"""
        return self.get('columns.drop_columns', [])
    
    @property
    def target_column(self) -> str:
        """Get the target column name"""
        return self.get('columns.target')
    
    def get_feature_binning_config(self, feature: str) -> Dict:
        """
        Get binning configuration for a specific feature
        
        Args:
            feature: Name of the feature
            
        Returns:
            Dictionary with binning configuration
        """
        return self.get(f'feature_binning.{feature}', {})
    
    def get_feature_engineering_config(self, feature: str) -> Dict:
        """
        Get feature engineering configuration for a specific feature
        
        Args:
            feature: Name of the feature
            
        Returns:
            Dictionary with feature engineering configuration
        """
        return self.get(f'feature_engineering.{feature}', {})
    
    def get_model_params(self, model_name: str) -> Dict:
        """
        Get model parameters for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model parameters
        """
        return self.get(f'model_training.models.{model_name}.param_grid', {})

# Create a global instance
config = Config()

# Example usage:
if __name__ == "__main__":
    # Load configuration
    print("Raw data path:", config.raw_data_path)
    print("Feature groups:", config.feature_groups)
    print("Tenure binning config:", config.get_feature_binning_config('tenure'))
    print("Services score config:", config.get_feature_engineering_config('services_score'))
