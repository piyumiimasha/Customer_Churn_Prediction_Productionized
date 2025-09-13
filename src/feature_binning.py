import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass

class TenureBinningStrategy(FeatureBinningStrategy):
    def __init__(self, bins: List[int], labels: List[str]):
        """
        Initialize tenure binning strategy
        
        Args:
            bins: List of bin edges for tenure binning
            labels: Labels for the tenure bins
        """
        self.bins = bins
        self.labels = labels
        logging.info("Initialized tenure binning strategy")

    def bin_feature(self, df: pd.DataFrame, column: str = 'tenure') -> pd.DataFrame:
        try:
            # Create tenure groups using the predefined bins and labels
            df = df.copy()
            df[f'{column}_group'] = pd.cut(
                df[column],
                bins=self.bins,
                labels=self.labels,
                right=True,
                include_lowest=True
            )
            
            logging.info(f"Successfully binned {column} into {len(self.labels)} categories")
            logging.info(f"{column} group distribution:\n{df[f'{column}_group'].value_counts()}")
            
            # Drop original column as done in the notebook
            df.drop(columns=[column], inplace=True)
            logging.info(f"Dropped original {column} column")
            
            return df
            
        except Exception as e:
            logging.error(f"Error during {column} binning: {str(e)}")
            raise

class SpendBinningStrategy(FeatureBinningStrategy):
    def __init__(self, bins: List[float], labels: List[str]):
        """
        Initialize spend binning strategy
        
        Args:
            bins: List of bin edges for spend binning
            labels: Labels for the spend bins
        """
        self.bins = bins
        self.labels = labels
        logging.info("Initialized spend binning strategy")
    
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        try:
            df = df.copy()
            df[f'{column}_category'] = pd.cut(
                df[column],
                bins=self.bins,
                labels=self.labels,
                right=True,
                include_lowest=True
            )
            
            logging.info(f"Successfully binned {column} into {len(self.labels)} categories")
            logging.info(f"{column} category distribution:\n{df[f'{column}_category'].value_counts()}")
            
            # Drop original column
            df.drop(columns=[column], inplace=True)
            logging.info(f"Dropped original {column} column")
            
            return df
            
        except Exception as e:
            logging.error(f"Error during {column} binning: {str(e)}")
            raise

class FeatureBinningHandler:
    def __init__(self, strategies: Optional[Dict[str, FeatureBinningStrategy]] = None):
        self.strategies = strategies or {}
    
    def add_strategy(self, column: str, strategy: FeatureBinningStrategy):
        """Add a new feature binning strategy"""
        self.strategies[column] = strategy
    
    def bin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            
            for column, strategy in self.strategies.items():
                if column in df.columns:
                    df = strategy.bin_feature(df, column)
                else:
                    logging.warning(f"Column {column} not found in DataFrame")
                
            logging.info("All feature binning strategies applied successfully")
            return df
            
        except Exception as e:
            logging.error(f"Error in feature binning: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        # Sample data
        df = pd.DataFrame({
            'tenure': [1, 15, 30, 50, 70],
            'MonthlyCharges': [30, 45, 80, 120, 250]
        })
        
        print("Original Data:")
        print(df)
        print("\nApplying feature binning...")
        
        # Create strategies with parameters
        tenure_strategy = TenureBinningStrategy(
            bins=[0, 12, 24, 48, 72],
            labels=['New', 'Intermediate', 'Established', 'Loyal']
        )
        spend_strategy = SpendBinningStrategy(
            bins=[0, 50, 100, 200, float('inf')],
            labels=['Low', 'Medium', 'High', 'Extreme']
        )
        
        # Create handler and add strategies
        handler = FeatureBinningHandler()
        handler.add_strategy('tenure', tenure_strategy)
        handler.add_strategy('MonthlyCharges', spend_strategy)
        
        # Apply feature binning
        df_binned = handler.bin_features(df)
        
        print("\nBinned features:")
        print(df_binned)
        
    except Exception as e:
        logging.error(f"Error in example: {str(e)}")