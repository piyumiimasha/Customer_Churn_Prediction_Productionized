import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def engineer_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class ServicesScoreStrategy(FeatureEngineeringStrategy):
    
    def __init__(self, service_columns: List[str]):
        """
        Initialize services score strategy
        
        Args:
            service_columns: List of service columns to calculate score from
        """
        self.service_columns = service_columns
    
    def engineer_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df['Services_Score'] = 0
            
            for service in self.service_columns:
                if service in df.columns:
                    df['Services_Score'] += (df[service] == 'Yes').astype(int)
                else:
                    logging.warning(f"Service column {service} not found in DataFrame")
            
            logging.info("Successfully calculated Services Score")
            logging.info(f"Score distribution:\n{df['Services_Score'].value_counts()}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating services score: {str(e)}")
            raise

class VulnerabilityScoreStrategy(FeatureEngineeringStrategy):
    
    def __init__(self, tenure_threshold: int, weights: Dict[str, int]):
        """
        Initialize vulnerability score strategy
        
        Args:
            tenure_threshold: Threshold for considering a customer new
            weights: Dictionary of weights for different factors:
                    - senior_citizen: Weight for senior citizen status
                    - no_partner: Weight for no partner
                    - no_dependents: Weight for no dependents
                    - month_to_month: Weight for month-to-month contract
                    - new_customer: Weight for new customer
        """
        self.tenure_threshold = tenure_threshold
        self.weights = weights
        
    def engineer_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            
            # Convert SeniorCitizen to numeric if it's not already
            senior_citizen_numeric = (
                df['SeniorCitizen'].map({'Yes': 1, 'yes': 1, 'No': 0, 'no': 0})
                if df['SeniorCitizen'].dtype == 'object'
                else df['SeniorCitizen']
            )
            
            # Calculate vulnerability score
            df['Vulnerability_Score'] = (
                senior_citizen_numeric * self.weights['senior_citizen'] +  # Senior citizen
                (df['Partner'] == 'No').astype(int) * self.weights['no_partner'] +  # No partner
                (df['Dependents'] == 'No').astype(int) * self.weights['no_dependents'] +  # No dependents
                (df['Contract'] == 'Month-to-month').astype(int) * self.weights['month_to_month'] +  # Month-to-month
                (df['tenure'] < self.tenure_threshold).astype(int) * self.weights['new_customer']  # New customer
            )
            
            logging.info("Successfully calculated Vulnerability Score")
            logging.info("Score interpretation:")
            logging.info("0-1: Low vulnerability")
            logging.info("2-4: Moderate vulnerability")
            logging.info("5-8: High vulnerability")
            logging.info(f"Score distribution:\n{df['Vulnerability_Score'].value_counts()}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating vulnerability score: {str(e)}")
            raise

class FeatureEngineeringHandler:
    def __init__(self, strategies: Optional[List[FeatureEngineeringStrategy]] = None):
        self.strategies = strategies or []
    
    def add_strategy(self, strategy: FeatureEngineeringStrategy):
        """Add a new feature engineering strategy"""
        self.strategies.append(strategy)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            
            for strategy in self.strategies:
                df = strategy.engineer_feature(df)
                
            logging.info("All feature engineering strategies applied successfully")
            return df
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        # Sample data
        df = pd.DataFrame({
            'SeniorCitizen': ['No', 'Yes', 'No', 'Yes'],
            'Partner': ['Yes', 'No', 'No', 'Yes'],
            'Dependents': ['No', 'No', 'Yes', 'Yes'],
            'tenure': [6, 24, 48, 12],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year'],
            'OnlineSecurity': ['No', 'Yes', 'Yes', 'No'],
            'TechSupport': ['No', 'Yes', 'No', 'Yes'],
            'OnlineBackup': ['Yes', 'Yes', 'No', 'No'],
            'DeviceProtection': ['No', 'Yes', 'Yes', 'No']
        })
        
        print("Original Data:")
        print(df)
        print("\nApplying feature engineering...")
        
        # Create strategies
        services_strategy = ServicesScoreStrategy()
        vulnerability_strategy = VulnerabilityScoreStrategy(tenure_threshold=12)
        
        # Create handler and add strategies
        handler = FeatureEngineeringHandler([services_strategy, vulnerability_strategy])
        
        # Apply feature engineering
        df_engineered = handler.engineer_features(df)
        
        print("\nEngineered features:")
        print("Services Score:")
        print(df_engineered['Services_Score'])
        print("\nVulnerability Score:")
        print(df_engineered['Vulnerability_Score'])
        
    except Exception as e:
        logging.error(f"Error in example: {str(e)}")
