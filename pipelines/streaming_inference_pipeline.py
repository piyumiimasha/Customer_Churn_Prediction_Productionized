import os
import sys
import json
import pandas as pd
import logging

# Add paths for src and utils modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_inference import ModelInference

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize ModelInference with the correct model path from training pipeline
inference = ModelInference('artifacts/models/telco_analysis.joblib')


def streaming_inference(inference_engine, customer_data):
    """
    Perform real-time churn prediction for telecom customer
    
    Args:
        inference_engine: ModelInference instance
        customer_data: Dict containing customer information
    
    Returns:
        Dict containing prediction results and recommendations
    """
    try:
        logger.info("Processing customer data for churn prediction")
        
        # Load encoders and model if not already loaded
        inference_engine.load_encoders('artifacts/encode')
        inference_engine.load_model()
        
        # Get prediction
        prediction_result = inference_engine.predict(customer_data)
        
        logger.info(f"Prediction completed for customer")
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error in streaming inference: {str(e)}")
        raise

if __name__ == '__main__':
    # Sample telecom customer data for churn prediction
    sample_customer = {
        "MonthlyCharges": 85.5,
        "TotalCharges": "2560.1",  # String format as in original data
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic", 
        "PaymentMethod": "Electronic check",
        "tenure": 30,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes", 
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "Yes"
    }
    
    try:
        logger.info("Starting streaming inference example")
        prediction_result = streaming_inference(inference, sample_customer)
        
        print("\n" + "="*50)
        print("TELECOM CHURN PREDICTION RESULTS")
        print("="*50)
        print(f"Customer Churn Prediction: {prediction_result}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Streaming inference failed: {str(e)}")
        print(f"Error: {str(e)}")