
### Steps To Run 
1. Create Python UV environment
    - uv venv
    - uv --version to check version

2. Activate the environment
    - copy paste the output comes with Activate with:

3. install requirements
    - uv pip install -r requirements.txt

4. Run pipelines separately
    - **Pandas Pipeline**: `python pipelines/data_pipeline.py`
    - **Spark Pipeline**: `python pipelines/spark_data_pipeline.py` (distributed processing)
    - **Model Training**: `python pipelines/training_pipeline.py`
    - **Spark Model Training**: `python pipelines/spark_model_trainer.py`


#### MLflow Interface

1. **MLflow Components**
    - **mlflow_utils.py** - Comprehensive MLflow integration utilities for experiment tracking, model versioning, and performance monitoring

2. **MLflow Utilities Features**
    - 🎯 **Experiment Management**: Automatic experiment creation and run tracking
    - 📊 **Data Pipeline Metrics**: Logs dataset info, missing values, outliers, feature counts
    - 🤖 **Model Tracking**: Hyperparameters, training metrics, model artifacts
    - 📈 **Visualization**: Auto-generated plots (confusion matrix, ROC curves, feature importance)
    - 🏷️ **Model Registry**: Model versioning and stage transitions (staging, production)
    - 📋 **Comprehensive Evaluation**: Precision, recall, F1-score, AUC metrics
    - 🔍 **Inference Tracking**: Prediction distribution and performance monitoring

3. **MLflow Setup**
    - MLflow is already installed with requirements
    - Experiments are automatically created when pipelines run
    - All artifacts saved to `./mlruns` directory

4. **Launch MLflow UI**
    ```bash
    # Method 1: Using environment variable
    mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)
    
    # Method 2: Direct port specification
    mlflow ui --host 0.0.0.0 --port 5000
    ```
    - UI will be available at: `http://localhost:5000` (or your specified port)
    - Open another terminal and run pipelines to see real-time results

5. **What Gets Tracked Automatically**
    - **Dataset Metrics**: Row counts, feature counts, missing values, outliers
    - **Model Parameters**: All hyperparameters and configuration settings
    - **Performance Metrics**: Accuracy, precision, recall, F1-score, AUC-ROC
    - **Visualizations**: Confusion matrices, ROC curves, feature importance plots
    - **Model Artifacts**: Trained models with full reproducibility
    - **Run Metadata**: Timestamps, tags, experiment organization


7. **Model Registry Features**
    - **Versioning**: Automatic model version management
    - **Stage Transitions**: Move models between Staging → Production
    - **Model Loading**: Easy model retrieval for inference
    - **Performance Comparison**: Compare different model versions

### Spark 



<img width="500" height="500" alt="2025-09-28 (1)" src="https://github.com/user-attachments/assets/c79e0dbe-512a-4fff-b173-2dc8951c92ae" />
<img width="500" height="500" alt="2025-09-28 (3)" src="https://github.com/user-attachments/assets/32edca49-6b98-4158-a256-1c21052ce076" />
<img width="500" height="500" alt="2025-09-28 (4)" src="https://github.com/user-attachments/assets/015136e6-875f-4f25-823f-dcfc93b9fce6" />
