import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class MeanImputationStrategy(MissingValueHandlingStrategy):
    def __init__(self, columns: List[str]):
        self.columns = columns
        logging.info(f"Using mean imputation for columns: {self.columns}")

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            if column in df.columns:
                mean_value = df[column].mean()
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    df[column] = df[column].fillna(mean_value)
                    logging.info(f"Imputed {missing_count} missing values in {column} with mean value: {mean_value:.2f}")
        return df

class ModeImputationStrategy(MissingValueHandlingStrategy):
    def __init__(self, columns: List[str]):
        self.columns = columns
        logging.info(f"Using mode imputation for columns: {self.columns}")

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            if column in df.columns:
                mode_value = df[column].mode()[0]
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    df[column] = df[column].fillna(mode_value)
                    logging.info(f"Imputed {missing_count} missing values in {column} with mode value: {mode_value}")
        return df

class BinaryEncodingStrategy(MissingValueHandlingStrategy):
    def __init__(self, columns: Dict[str, str]):
        self.columns = columns
        logging.info(f"Using binary encoding for columns: {list(columns.keys())}")

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        for source_col, target_col in self.columns.items():
            if source_col in df.columns:
                # Convert to binary (0/1)
                df[target_col] = df[source_col].map({'No': 0, 'Yes': 1})
                # Drop original column
                df.drop(columns=[source_col], inplace=True)
                logging.info(f"Encoded {source_col} to {target_col} with binary values")
        return df

class CustomPreprocessingStrategy(MissingValueHandlingStrategy):
    def __init__(self, columns: List[str]):
        self.columns = columns
        logging.info(f"Using custom preprocessing for columns: {self.columns}")

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert TotalCharges to numeric, replacing spaces and empty strings with NaN
        if 'TotalCharges' in df.columns:
            # Replace both spaces and empty strings with NaN
            df['TotalCharges'] = df['TotalCharges'].replace([' ', ''], np.nan)
            # Convert to numeric, handling any remaining non-numeric values
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            missing_count = df['TotalCharges'].isnull().sum()
            logging.info(f"Converted TotalCharges to numeric format. Found {missing_count} missing values")
        return df

class MissingValueHandler:
    def __init__(self):
        self.strategies = []
        
    def add_strategy(self, strategy: MissingValueHandlingStrategy):
        self.strategies.append(strategy)
        return self

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for strategy in self.strategies:
            df = strategy.handle(df.copy())
        return df

def create_default_handler() -> MissingValueHandler:
    """
    Create a default handler with strategies based on the notebook implementation
    """
    # Import config here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from config import config
    
    handler = MissingValueHandler()
    
    # Add preprocessing strategy for TotalCharges
    handler.add_strategy(CustomPreprocessingStrategy(['TotalCharges']))
    
    # Add mean imputation for numeric columns from config
    numeric_columns = config.get('missing_values.columns_to_impute.numeric', ['TotalCharges'])
    handler.add_strategy(MeanImputationStrategy(numeric_columns))
    
    # Add binary column mappings from config
    binary_columns = config.get('missing_values.binary_encoding', {})
    handler.add_strategy(BinaryEncodingStrategy(binary_columns))
    
    # Add mode imputation for binary columns from config
    categorical_columns = config.get('missing_values.columns_to_impute.categorical', [])
    handler.add_strategy(ModeImputationStrategy(categorical_columns))
    
    return handler

if __name__ == "__main__":
    # Example usage
    try:
        # Load your data
        df = pd.read_csv("data/hmQOVnDvRN.xls")
        print("\nOriginal Data Info:")
        print("-------------------")
        print(f"Data shape: {df.shape}")
        print("\nMissing values before handling:")
        print(df.isnull().sum())
        
        # Create and configure the handler
        handler = create_default_handler()
        
        # Handle missing values
        df_cleaned = handler.handle_missing_values(df)
        
        print("\nProcessed Data Info:")
        print("-------------------")
        print(f"Data shape: {df_cleaned.shape}")
        print("\nMissing values after handling:")
        print(df_cleaned.isnull().sum())
        
        # Show sample of transformed columns
        if 'OnlineSecurity_numeric' in df_cleaned.columns:
            print("\nSample of transformed OnlineSecurity column:")
            print(pd.DataFrame({
                'Original': df['OnlineSecurity'].head(),
                'Transformed': df_cleaned['OnlineSecurity_numeric'].head()
            }))
        
        if 'TechSupport_numeric' in df_cleaned.columns:
            print("\nSample of transformed TechSupport column:")
            print(pd.DataFrame({
                'Original': df['TechSupport'].head(),
                'Transformed': df_cleaned['TechSupport_numeric'].head()
            }))
            
        logging.info("Missing value handling completed successfully")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise
