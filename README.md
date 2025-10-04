
#  Customer Churn Prediction - Productionized

A **production-ready machine learning system** for customer churn prediction with both **traditional pandas** and **distributed Spark** processing pipelines, orchestrated with **Apache Airflow**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.11.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.x-green)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-red)
![Production Ready](https://img.shields.io/badge/Production-Ready-brightgreen)


## 🚀 Quick Start

### Initial Steps To Run 
1. **Create Python UV environment**
    ```bash
    - uv venv
    - uv --version to check version
    ```

2. **Activate the environment**
    - copy paste the output comes with Activate with:

3. **install requirements**
    ```bash
    - uv pip install -r requirements.txt
    ```

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

   **🚀 Airflow Orchestration (Production)**
   ```bash
   # Start Airflow services
   .\run.ps1 airflow-start
   
   # Access Airflow Web UI
   # http://localhost:8080 (admin/admin)
   
   # Access MLflow UI  
   # http://localhost:5000
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

### Apache Airflow Orchestration

#### 1. **Airflow DAGs Overview**

#### **1.1 Data Pipeline DAG** (`dags/data_pipeline_dag.py`)
- **Schedule**: Every 5 minutes (`*/5 * * * *`)
- **Purpose**: Automated data preprocessing and feature engineering
- **Tasks**:
  - `validate_input_data` - Check data availability and integrity
  - `run_data_pipeline` - Execute preprocessing, cleaning, encoding, scaling
- **Output**: Clean, processed data ready for training

#### **1.2. Training Pipeline DAG** (`dags/train_pipeline_dag.py`)
- **Schedule**: Daily at 1:00 AM IST (`30 19 * * *`)
- **Purpose**: Automated model training with hyperparameter tuning
- **Tasks**:
  - `validate_processed_data` - Ensure training data is available
  - `run_training_pipeline` - Execute model training with optimization
- **Features**:
  - **Automated Hyperparameter Tuning** with CrossValidator
  - **MLflow Integration** for experiment tracking
  - **Model Evaluation** with comprehensive metrics
  - **Model Registry** management

#### **1.3. Inference Pipeline DAG** (`dags/inference_pipeline_dag.py`)
- **Schedule**: Every minute (`* * * * *`)
- **Purpose**: Automated prediction generation and monitoring
- **Tasks**:
  - `validate_trained_model` - Check model availability
  - `run_inference_pipeline` - Generate predictions and store results
- **Features**:
  - **Real-time Predictions** with latest trained model
  - **Result Storage** and performance monitoring
  - **Model Validation** before inference

#### 2. **Airflow Setup & Installation**

#### **2.1 Install Apache Airflow**
```bash
# Install Airflow with all required providers
pip install "apache-airflow[celery,redis,postgres,http,email]==2.11.0"
```

#### **2.2 Initialize Airflow Database**
```bash
# Initialize Airflow metadata database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

#### **2.3 Start Airflow Services**

**Option A: Using the provided script (Recommended)**
```bash
# For Windows
.\run.ps1 airflow-start

# For Linux/Mac
./run.sh airflow-start
```

**Option B: Manual startup**
```bash
# Terminal 1: Start webserver
airflow webserver --port 8080

# Terminal 2: Start scheduler
airflow scheduler

# Terminal 3: Start triggerer (for sensors)
airflow triggerer
```

#### **2.4 Access Airflow Web UI**
- **URL**: `http://localhost:8080`
- **Username**: `admin`
- **Password**: `admin`

#### 3. **Airflow Task Implementation**

All Airflow tasks are implemented in `utils/airflow_tasks.py` with professional error handling:


#### 📁 **Project Structure **

```
├── 🗂️ dags/                           # Airflow DAG definitions
│   ├── data_pipeline_dag.py          # Data preprocessing workflow
│   ├── train_pipeline_dag.py         # Model training workflow
│   └── inference_pipeline_dag.py     # Inference workflow
│
├── 🗂️ utils/                          # Airflow utilities
│   ├── airflow_tasks.py              # Professional task wrappers
│   ├── config.py                     # Configuration management
│   └── mlflow_utils.py               # MLflow integration
│
├── 🗂️ pipelines/                      # Core ML pipelines
│   ├── data_pipeline.py              # Pandas data processing
│   ├── training_pipeline.py          # Model training pipeline
│   ├── spark_data_pipeline.py        # Distributed data processing
│   ├── spark_model_trainer.py        # Distributed model training
│   ├── unified_spark_pipeline.py     # End-to-end Spark pipeline
│   └── streaming_inference_pipeline.py # Real-time inference
│
├── 🗂️ src/                           # Core ML components
│   ├── data_ingestion.py             # Data loading utilities
│   ├── feature_engineering.py        # Feature creation
│   ├── model_building.py             # Model architecture
│   ├── model_training.py             # Training logic
│   ├── model_evaluation.py           # Evaluation metrics
│   ├── model_inference.py            # Prediction generation
│   └── spark_*.py                    # Spark-specific implementations
│
├── 🗂️ artifacts/                      # Generated outputs
│   ├── data/                         # Processed datasets
│   ├── models/                       # Trained models
│   ├── plots/                        # Visualizations
│   └── evaluation/                   # Evaluation results
│
├── 🗂️ .airflow/                      # Airflow metadata (auto-generated)
│   ├── airflow.cfg                   # Airflow configuration
│   ├── airflow.db                    # SQLite database
│   └── logs/                         # Execution logs
│
├── 🗂️ mlruns/                        # MLflow experiment tracking
│   └── [experiment_folders]/         # MLflow artifacts
│
├── 📄 run.ps1                        # Windows Airflow launcher
├── 📄 airflow_settings.yaml          # Airflow configuration
├── 📄 config.yml                     # Project configuration
└── 📄 requirements.txt               # Dependencies (includes Airflow)
```
