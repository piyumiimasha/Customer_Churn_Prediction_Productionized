import os
import sys
import pandas as pd
from typing import Dict
import numpy as np
import mlflow

from imblearn.over_sampling import SMOTE

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import create_default_handler
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_engineering import FeatureEngineeringHandler, ServicesScoreStrategy, VulnerabilityScoreStrategy
from feature_binning import FeatureBinningHandler, TenureBinningStrategy
from feature_encoding import NominalEncodingStrategy, OrdinalEncodingStratergy
from feature_scaling import MinMaxScalingStratergy
from data_splitter import SimpleTrainTestSplitStratergy

from config import config

def data_pipeline(force_rebuild: bool = False) -> Dict[str, np.ndarray]:
    
    data_paths = config.get('data_paths')
    columns_config = config.get('columns')
    target_column = columns_config['target']
    data_path = data_paths['raw_data']

    # Initialize MLflow tracking
    mlflow_tracker = MLflowTracker()
    setup_mlflow_autolog()
    run_tags = create_mlflow_run_tags('data_pipeline', {
            'data_source': data_path,
            'force_rebuild': str(force_rebuild),
            'target_column': target_column
        })
    run = mlflow_tracker.start_run(run_name='data_pipeline', tags=run_tags)
    
    print('Step 1: Data Ingestion')
    artifacts_dir = data_paths['data_artifacts_dir']
    x_train_path = os.path.join(artifacts_dir, 'X_train.csv')
    x_test_path = os.path.join(artifacts_dir, 'X_test.csv')
    y_train_path = os.path.join(artifacts_dir, 'Y_train.csv')
    y_test_path = os.path.join(artifacts_dir, 'Y_test.csv')

    if not force_rebuild and all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path]):
        X_train = pd.read_csv(x_train_path)
        X_test = pd.read_csv(x_test_path)
        Y_train = pd.read_csv(y_train_path).squeeze()
        Y_test = pd.read_csv(y_test_path).squeeze()
        
        # Log artifact paths as parameters
        mlflow.log_params({
            'x_train_path': x_train_path,
            'x_test_path': x_test_path,
            'y_train_path': y_train_path,
            'y_test_path': y_test_path,
            'data_loaded_from_cache': True
        })
        
        mlflow_tracker.end_run()
        return {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test}

    # Create artifacts directory
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Data Ingestion
    ingestor = DataIngestorCSV()
    df = ingestor.ingest(data_paths['raw_data'])
    print(f"Loaded data shape: {df.shape}")
    
    # Log initial data info
    mlflow.log_params({
        'original_data_shape': str(df.shape),
        'original_features': df.shape[1],
        'original_samples': df.shape[0]
    })

    print('\nStep 2: Handle Missing Values')
    missing_values_before = df.isnull().sum().sum()
    missing_handler = create_default_handler()
    df = missing_handler.handle_missing_values(df)
    missing_values_after = df.isnull().sum().sum()
    print(f"Data shape after missing values: {df.shape}")
    
    # Log missing values handling
    mlflow.log_params({
        'missing_values_before': missing_values_before,
        'missing_values_after': missing_values_after,
        'missing_values_handled': missing_values_before - missing_values_after
    })

    print('\nStep 3: Feature Engineering')
    features_before = df.shape[1]
    engineering_handler = FeatureEngineeringHandler()
    services_config = config.get('feature_engineering.services_score')
    engineering_handler.add_strategy(ServicesScoreStrategy(services_config['services']))
    vulnerability_config = config.get('feature_engineering.vulnerability_score')
    engineering_handler.add_strategy(VulnerabilityScoreStrategy(
        vulnerability_config['tenure_threshold'], 
        vulnerability_config['weights']
    ))
    df = engineering_handler.engineer_features(df)
    features_after = df.shape[1]
    print(f"Data shape after feature engineering: {df.shape}")
    
    # Log feature engineering
    mlflow.log_params({
        'features_before_engineering': features_before,
        'features_after_engineering': features_after,
        'new_features_created': features_after - features_before,
        'services_config': str(services_config),
        'vulnerability_config': str(vulnerability_config)
    })

    print('\nStep 4: Drop Unnecessary Columns')
    drop_columns = config.get('columns.drop_columns', [])
    columns_to_drop = [col for col in drop_columns if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Dropped columns: {columns_to_drop}")
    print(f"Data shape after dropping columns: {df.shape}")
    
    # Log column dropping
    mlflow.log_params({
        'columns_dropped': str(columns_to_drop),
        'num_columns_dropped': len(columns_to_drop),
        'final_features_count': df.shape[1] - 1  # -1 for target column
    })

    print('\nStep 5: Feature Binning')
    binning_handler = FeatureBinningHandler()
    tenure_config = config.get('feature_binning.tenure')
    binning_handler.add_strategy('tenure', TenureBinningStrategy(tenure_config['bins'], tenure_config['labels']))
    df = binning_handler.bin_features(df)
    print(f"Data shape after binning: {df.shape}")
    
    # Log binning configuration
    mlflow.log_params({
        'tenure_bins': str(tenure_config['bins']),
        'tenure_labels': str(tenure_config['labels']),
        'binning_applied': True
    })

    print('\nStep 6: Feature Encoding')
    encoding_config = config.get('feature_encoding')
    nominal_strategy = NominalEncodingStrategy(encoding_config['nominal_features'])
    
    # Get ordinal mappings from config
    ordinal_mappings = encoding_config.get('ordinal_mappings', {}).copy()
    
    # For features not in config, create automatic mappings
    for feature in encoding_config['ordinal_features']:
        if feature not in ordinal_mappings and feature in df.columns:
            unique_vals = df[feature].unique()
            ordinal_mappings[feature] = {val: i for i, val in enumerate(sorted(unique_vals))}
    
    ordinal_strategy = OrdinalEncodingStratergy(ordinal_mappings)
    df = nominal_strategy.encode(df)
    df = ordinal_strategy.encode(df)
    print(f"Data shape after encoding: {df.shape}")
    
    # Log encoding configuration
    mlflow.log_params({
        'nominal_features': str(encoding_config['nominal_features']),
        'ordinal_features': str(encoding_config['ordinal_features']),
        'ordinal_mappings': str(ordinal_mappings),
        'encoding_applied': True
    })
    print(df)

    print('\nStep 7: Feature Scaling')
    scaling_config = config.get('feature_scaling')
    numeric_features = [col for col in scaling_config['numeric_features'] if col in df.columns]
    if numeric_features:
        scaling_strategy = MinMaxScalingStratergy()
        df = scaling_strategy.scale(df, numeric_features)
    print(f"Data shape after scaling: {df.shape}")
    
    # Log scaling configuration
    mlflow.log_params({
        'numeric_features_scaled': str(numeric_features),
        'num_features_scaled': len(numeric_features),
        'scaling_method': 'MinMaxScaling',
        'scaling_applied': len(numeric_features) > 0
    })

    print('\nStep 8: Data Splitting')
    target_col = columns_config['target']
    split_config = config.get('data_splitting')
    splitting_strategy = SimpleTrainTestSplitStratergy(test_size=split_config['test_size'])
    X_train, X_test, Y_train, Y_test = splitting_strategy.split_data(df, target_col)
    
    # Log splitting configuration
    mlflow.log_params({
        'test_size': split_config['test_size'],
        'random_state': split_config.get('random_state', 'default'),
        'target_column': target_col
    })

    print('\nStep 9: Handle Class Imbalance')
    balance_config = config.get('class_balancing')
    original_train_size = len(X_train)
    original_class_distribution = Y_train.value_counts().to_dict()
    
    if balance_config['method'] == 'smote':
        smote = SMOTE(random_state=balance_config['random_state'])
        X_train, Y_train = smote.fit_resample(X_train, Y_train)
        
    new_class_distribution = Y_train.value_counts().to_dict()
    
    # Log class balancing
    mlflow.log_params({
        'balancing_method': balance_config['method'],
        'balancing_random_state': balance_config['random_state'],
        'original_train_size': original_train_size,
        'final_train_size': len(X_train),
        'original_class_distribution': str(original_class_distribution),
        'final_class_distribution': str(new_class_distribution)
    })

    # Save processed data
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    Y_train.to_csv(y_train_path, index=False)
    Y_test.to_csv(y_test_path, index=False)

    print(f"X train size: {X_train.shape}")
    print(f"X test size: {X_test.shape}")
    print(f"Y train size: {Y_train.shape}")
    print(f"Y test size: {Y_test.shape}")
    
    # Log comprehensive pipeline metrics using the MLflow tracker
    dataset_info = {
        'total_rows': len(X_train) + len(X_test),
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'num_features': X_train.shape[1],
        'missing_values': missing_values_before,
        'outliers_removed': 0,  # Add outlier detection if implemented
        'test_size': split_config['test_size'],
        'random_state': split_config.get('random_state', 42),
        'missing_strategy': 'default_handler',
        'outlier_method': 'none',
        'encoding_applied': True,
        'scaling_applied': len(numeric_features) > 0,
        'feature_names': list(X_train.columns)
    }
    
    mlflow_tracker.log_data_pipeline_metrics(dataset_info)
    
    # Log artifact paths
    mlflow.log_params({
        'x_train_path': x_train_path,
        'x_test_path': x_test_path,
        'y_train_path': y_train_path,
        'y_test_path': y_test_path,
        'data_loaded_from_cache': False
    })
    
    # End MLflow run
    mlflow_tracker.end_run()

    return {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test}

if __name__ == "__main__":
    result = data_pipeline(force_rebuild=True)
