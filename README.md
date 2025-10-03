
#  Customer Churn Prediction - Productionized

A **production-ready machine learning system** for customer churn prediction with both **traditional pandas** and **distributed Spark** processing pipelines.


### Initial Steps To Run 
1. **Create Python UV environment**
    - uv venv
    - uv --version to check version

2. **Activate the environment**
    - copy paste the output comes with Activate with:

3. **install requirements**
    - uv pip install -r requirements.txt

4. **Choose Your Pipeline** (run separately)

   **🐼 Pandas Pipeline (Single Machine)**
   ```bash
   python pipelines/data_pipeline.py      # Data preprocessing
   python pipelines/training_pipeline.py  # Model training
   ```

   **⚡ Spark Pipeline (Distributed)**
   ```bash
   python pipelines/spark_data_pipeline.py    # Data preprocessing only
   python pipelines/spark_model_trainer.py    # Model training only
   python pipelines/unified_spark_pipeline.py # Complete end-to-end (recommended)
   ```


### MLflow Interface

1. **MLflow Components**
    - **mlflow_utils.py** - Comprehensive MLflow integration utilities for experiment tracking, model versioning, and performance monitoring

2. **MLflow Utilities Features**
    -  **Experiment Management**: Automatic experiment creation and run tracking
    -  **Data Pipeline Metrics**: Logs dataset info, missing values, outliers, feature counts
    -  **Model Tracking**: Hyperparameters, training metrics, model artifacts
    -  **Visualization**: Auto-generated plots (confusion matrix, ROC curves, feature importance)
    -  **Model Registry**: Model versioning and stage transitions (staging, production)
    -  **Comprehensive Evaluation**: Precision, recall, F1-score, AUC metrics
    -  **Inference Tracking**: Prediction distribution and performance monitoring

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



### 📸 MLflow UI Screenshots

<img width="500" height="500" alt="2025-09-28 (1)" src="https://github.com/user-attachments/assets/c79e0dbe-512a-4fff-b173-2dc8951c92ae" />
<img width="500" height="500" alt="2025-09-28 (3)" src="https://github.com/user-attachments/assets/32edca49-6b98-4158-a256-1c21052ce076" />
<img width="500" height="500" alt="2025-09-28 (4)" src="https://github.com/user-attachments/assets/015136e6-875f-4f25-823f-dcfc93b9fce6" />


### Spark Distributed Processing

#### 1. **Run Spark Pipelines**
```bash
# Complete end-to-end Spark pipeline (recommended)
python pipelines/unified_spark_pipeline.py

# Individual Spark components
python pipelines/spark_data_pipeline.py      # Data preprocessing only
python pipelines/spark_model_trainer.py      # Model training only

```

#### 2. **Core Spark Files**
- **`pipelines/spark_data_pipeline.py`** - Distributed data preprocessing with MLflow integration
- **`pipelines/spark_model_trainer.py`** - MLlib model training (LogisticRegression, RandomForest, GBT)
- **`pipelines/unified_spark_pipeline.py`** - Complete end-to-end Spark pipeline
- **`src/spark_session.py`** - Optimized SparkSession management
- **`src/spark_utils.py`** - Spark utility functions

#### 3. **Spark Features & Capabilities**
-  **Distributed Processing**: Automatic data partitioning and parallel execution
-  **MLlib Integration**: Native Spark ML algorithms (LogisticRegression, RandomForest, GBT)
-  **Performance Optimization**: Adaptive Query Execution, Arrow support, Kryo serialization
-  **Feature Engineering**: Distributed feature creation matching pandas implementation
-  **Scalability**: Handles datasets from MBs to TBs
-  **Cross-Validation**: Distributed hyperparameter tuning
-  **Data Formats**: Support for Parquet, CSV, JSON with intelligent format detection



#### 4. **Spark Configuration**
The pipeline uses optimized Spark configurations:
```python
# Key optimizations applied
"spark.sql.adaptive.enabled": "true"                    # Adaptive Query Execution
"spark.sql.adaptive.coalescePartitions.enabled": "true" # Dynamic partitioning
"spark.sql.execution.arrow.pyspark.enabled": "true"     # Arrow optimization
"spark.serializer": "org.apache.spark.serializer.KryoSerializer"  # Fast serialization
```


#### 5. **Model Training (MLlib)**
```python
# Available algorithms
- LogisticRegression (with regularization)
- RandomForestClassifier (100+ trees)
- GBTClassifier (Gradient Boosted Trees - Spark's XGBoost equivalent)

# Features
- Distributed cross-validation
- Hyperparameter tuning with grid search
- Model evaluation with comprehensive metrics
- Automatic model registry integration
```

