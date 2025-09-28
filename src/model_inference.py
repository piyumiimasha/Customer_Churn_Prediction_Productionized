import json
import logging
import os
import joblib, sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Import existing src modules
from feature_engineering import ServicesScoreStrategy, VulnerabilityScoreStrategy
from feature_binning import TenureBinningStrategy
from handle_missing_values import MeanImputationStrategy, BinaryEncodingStrategy
from feature_encoding import NominalEncodingStrategy

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(self, model_path, preprocessor_path=None):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.encoders = {}

    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model file not found at: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")

    def load_preprocessor(self):
        """Load the fitted preprocessor (ColumnTransformer)"""
        if self.preprocessor_path and os.path.exists(self.preprocessor_path):
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info(f"Preprocessor loaded from {self.preprocessor_path}")
        else:
            logger.warning("No preprocessor path provided or file not found")

    def load_encoders(self, encoders_dir):
        """Load saved encoders for categorical features"""
        if os.path.exists(encoders_dir):
            for file in os.listdir(encoders_dir):
                if file.endswith('_encoder.json'):
                    feature_name = file.split('_encoder.json')[0]
                    with open(os.path.join(encoders_dir, file), 'r') as f:
                        self.encoders[feature_name] = json.load(f)
            logger.info(f"Loaded {len(self.encoders)} encoders")

    def preprocess_input(self, data):
        """
        Preprocess input data using existing src modules:
        1. Handle missing values using MeanImputationStrategy
        2. Create engineered features using ServicesScoreStrategy and VulnerabilityScoreStrategy
        3. Apply feature binning using TenureBinningStrategy
        4. Handle binary encoding using BinaryEncodingStrategy
        5. Apply preprocessing pipeline
        """
        # Convert to DataFrame if it's a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            data = data.to_frame().T
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # 1. Handle TotalCharges conversion and missing values 
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Use existing MeanImputationStrategy
            mean_imputer = MeanImputationStrategy(['TotalCharges'])
            df = mean_imputer.handle(df)

        # Convert SeniorCitizen to string format
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].apply(
                lambda x: "yes" if x == 1 else "no"
            )

        # 2. Create engineered features 
        # Services Score 
        support_services = ['OnlineBackup', 'DeviceProtection', 'OnlineSecurity', 'TechSupport']
        services_strategy = ServicesScoreStrategy(support_services)
        df = services_strategy.engineer_feature(df)

        # Vulnerability Score 
        vulnerability_weights = {
            'senior_citizen': 2,
            'no_partner': 1,
            'no_dependents': 1,
            'month_to_month': 2,
            'new_customer': 2
        }
        vulnerability_strategy = VulnerabilityScoreStrategy(
            tenure_threshold=12,
            weights=vulnerability_weights
        )
        df = vulnerability_strategy.engineer_feature(df)

        # 3. Tenure binning 
        if 'tenure' in df.columns:
            bins = [0, 12, 24, 48, 72]
            labels = ['New', 'Intermediate', 'Established', 'Loyal']
            tenure_binning = TenureBinningStrategy(bins=bins, labels=labels)
            df = tenure_binning.bin_feature(df, 'tenure')

        # Drop features
        features_to_drop = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'OnlineBackup', 'DeviceProtection',
            'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn'
        ]
        df = df.drop(columns=[col for col in features_to_drop if col in df.columns])

        # 4. Handle binary encoding
        binary_columns = {
            'OnlineSecurity': 'OnlineSecurity_numeric',
            'TechSupport': 'TechSupport_numeric'
        }
        
        for source_col, target_col in binary_columns.items():
            if source_col in df.columns:
                df[target_col] = df[source_col].map({'No': 0, 'Yes': 1})
                # Handle missing values with mode (0 as per notebook)
                df[target_col] = df[target_col].fillna(0)
                df = df.drop(columns=[source_col])

        # 5. Use preprocessor if available
        if self.preprocessor is not None:
            # Apply preprocessing
            processed_data = self.preprocessor.transform(df)
            
            # If preprocessor returns sparse matrix, convert to dense
            if hasattr(processed_data, 'toarray'):
                processed_data = processed_data.toarray()
                
            return processed_data
        else:
            # Manual preprocessing if no preprocessor available
            logger.warning("No preprocessor available, returning raw processed data")
            return df

    def segment_customer(self, probability):
        """
        Segment customer into risk categories based on churn probability
        Same logic as in notebook
        """
        if probability >= 0.7:
            return 'High-Risk'
        elif probability >= 0.4:
            return 'Medium-Risk'
        else:
            return 'Low-Risk'

    def get_retention_strategy(self, risk_segment):
        """
        Get retention strategy based on risk segment
        Based on notebook's business impact analysis
        """
        strategies = {
            'High-Risk': {
                'strategy': 'Immediate Intervention',
                'actions': [
                    'Personal retention specialist contact within 24 hours',
                    'Offer contract migration (month-to-month to long-term)',
                    'Provide service bundles (OnlineSecurity + TechSupport)',
                    'Loyalty discount (10-15% for 12 months)',
                    'Free service upgrades or premium features'
                ],
                'priority': 'Critical'
            },
            'Medium-Risk': {
                'strategy': 'Proactive Engagement',
                'actions': [
                    'Targeted email campaigns with service recommendations',
                    'Offer contract incentives (upgrade to annual plans)',
                    'Cross-sell complementary services',
                    'Satisfaction surveys and feedback collection',
                    'Loyalty program enrollment'
                ],
                'priority': 'High'
            },
            'Low-Risk': {
                'strategy': 'Upselling & Loyalty',
                'actions': [
                    'Upsell premium services and features',
                    'Loyalty rewards and recognition programs',
                    'Referral incentives',
                    'Exclusive offers for long-term customers',
                    'Regular satisfaction monitoring'
                ],
                'priority': 'Medium'
            }
        }
        return strategies.get(risk_segment, {})

    def predict(self, data):
        """
        Make prediction and return comprehensive results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Preprocess input data
        processed_data = self.preprocess_input(data)
        
        # Make predictions
        y_pred = self.model.predict(processed_data)
        y_proba = self.model.predict_proba(processed_data)[:, 1]
        
        # Get results
        churn_prediction = 'Churn' if y_pred[0] == 1 else 'No Churn'
        churn_probability = round(y_proba[0] * 100, 2)
        
        # Customer segmentation
        risk_segment = self.segment_customer(y_proba[0])
        retention_strategy = self.get_retention_strategy(risk_segment)
        
        return {
            "prediction": churn_prediction,
            "churn_probability": churn_probability,
            "risk_segment": risk_segment,
            "retention_strategy": retention_strategy,
            "confidence": {
                "high_confidence": churn_probability > 80 or churn_probability < 20,
                "prediction_strength": "High" if churn_probability > 80 or churn_probability < 20 else "Medium"
            }
        }
    




    
