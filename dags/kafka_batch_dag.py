"""
Airflow DAG for Kafka Batch Processing Pipeline
Hourly/daily pipeline: producer → consumer → summary
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Default arguments for the DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

# DAG definition
dag = DAG(
    'kafka_batch_pipeline',
    default_args=default_args,
    description='Batch processing pipeline: producer → consumer → summary',
    schedule_interval=timedelta(hours=1),  # Run every hour
    catchup=False,
    tags=['kafka', 'batch', 'telco-churn', 'ml-pipeline'],
)

def validate_data_quality(**context):
    """Validate the quality of input data before processing"""
    import pandas as pd
    import yaml
    
    # Load config
    config_path = os.path.join(project_root, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    data_path = os.path.join(project_root, config['data_paths']['raw_data'])
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    
    # Basic quality checks
    total_records = len(df)
    missing_customer_ids = df['customerID'].isnull().sum()
    duplicate_customers = df['customerID'].duplicated().sum()
    
    print(f"Data Quality Report:")
    print(f"Total records: {total_records}")
    print(f"Missing customer IDs: {missing_customer_ids}")
    print(f"Duplicate customers: {duplicate_customers}")
    
    # Fail if data quality is poor
    if missing_customer_ids > total_records * 0.1:  # More than 10% missing IDs
        raise ValueError(f"Too many missing customer IDs: {missing_customer_ids}")
    
    if duplicate_customers > total_records * 0.05:  # More than 5% duplicates
        raise ValueError(f"Too many duplicate customers: {duplicate_customers}")
    
    print("Data quality validation passed")
    return True

def generate_batch_summary(**context):
    """Generate summary report for the batch processing results"""
    import json
    import glob
    
    # Find the latest batch summary file
    summary_pattern = os.path.join(
        project_root, 
        'artifacts', 
        'evaluation', 
        'batch_summary_*.json'
    )
    
    summary_files = glob.glob(summary_pattern)
    
    if not summary_files:
        print("No batch summary files found")
        return
    
    # Get the most recent summary
    latest_summary = max(summary_files, key=os.path.getctime)
    
    with open(latest_summary, 'r') as f:
        summary = json.load(f)
    
    # Generate enhanced summary
    enhanced_summary = {
        **summary,
        'pipeline_run_date': context['ds'],
        'dag_run_id': context['dag_run'].run_id,
        'execution_date': context['execution_date'].isoformat()
    }
    
    # Save enhanced summary
    enhanced_path = os.path.join(
        project_root,
        'artifacts',
        'evaluation',
        f"enhanced_summary_{context['ds']}.json"
    )
    
    with open(enhanced_path, 'w') as f:
        json.dump(enhanced_summary, f, indent=2)
    
    print(f"Enhanced summary saved to: {enhanced_path}")
    
    # Print key metrics
    print("\n=== BATCH PROCESSING SUMMARY ===")
    print(f"Execution Date: {context['ds']}")
    print(f"Total Predictions: {summary.get('total_predictions', 0)}")
    print(f"Churn Rate: {summary.get('churn_percentage', 0):.2f}%")
    print(f"High Risk Customers: {summary.get('high_risk_customers', 0)}")
    
    return enhanced_summary

# Task 1: Validate input data quality
validate_data = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag,
)

# Task 2: Run batch producer
run_producer = BashOperator(
    task_id='run_batch_producer',
    bash_command="""
    cd {{ params.project_root }}
    python pipelines/producer.py \
        --mode batch \
        --data-path {{ params.data_path }} \
        --batch-size {{ params.batch_size }} \
        --config-path config.yml
    """,
    params={
        'project_root': str(project_root),
        'data_path': 'data/hmQOVnDvRN.xls',  # Update with actual data path
        'batch_size': 1000
    },
    dag=dag,
)

# Task 3: Wait for producer to complete and messages to be available
wait_for_messages = BashOperator(
    task_id='wait_for_messages',
    bash_command="""
    echo "Waiting for messages to be available in Kafka..."
    sleep 30  # Wait 30 seconds for messages to be produced
    
    # In production, you'd implement proper message availability check
    echo "Messages should be available for consumption"
    """,
    dag=dag,
)

# Task 4: Run batch consumer
run_consumer = BashOperator(
    task_id='run_batch_consumer',
    bash_command="""
    cd {{ params.project_root }}
    python pipelines/consumer.py \
        --mode batch \
        --max-records {{ params.max_records }} \
        --timeout-ms {{ params.timeout_ms }} \
        --config-path config.yml
    """,
    params={
        'project_root': str(project_root),
        'max_records': 1000,
        'timeout_ms': 60000  # 60 seconds timeout
    },
    dag=dag,
)

# Task 5: Generate batch processing summary
generate_summary = PythonOperator(
    task_id='generate_batch_summary',
    python_callable=generate_batch_summary,
    dag=dag,
)

# Task 6: Cleanup temporary files and optimize storage
cleanup_and_optimize = BashOperator(
    task_id='cleanup_and_optimize',
    bash_command="""
    cd {{ params.project_root }}
    
    echo "Cleaning up temporary files..."
    
    # Clean up old checkpoint files (keep last 7 days)
    find checkpoints/ -name "*.txt" -mtime +7 -delete 2>/dev/null || echo "No old checkpoints to clean"
    
    # Clean up old batch summaries (keep last 30 days)
    find artifacts/evaluation/ -name "batch_summary_*.json" -mtime +30 -delete 2>/dev/null || echo "No old summaries to clean"
    
    # Compress old log files
    find logs/ -name "*.log" -mtime +1 -exec gzip {} \\; 2>/dev/null || echo "No logs to compress"
    
    echo "Cleanup completed"
    """,
    params={'project_root': str(project_root)},
    dag=dag,
)

# Task 7: Send notification (placeholder)
send_notification = BashOperator(
    task_id='send_pipeline_notification',
    bash_command="""
    echo "Batch pipeline completed successfully"
    echo "Execution date: {{ ds }}"
    echo "DAG run ID: {{ dag_run.run_id }}"
    
    # In production, you would send actual notifications (email, Slack, etc.)
    # slack-cli send "Batch pipeline completed for {{ ds }}"
    """,
    dag=dag,
)

# Define task dependencies
validate_data >> run_producer >> wait_for_messages >> run_consumer
run_consumer >> generate_summary >> cleanup_and_optimize >> send_notification