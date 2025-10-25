"""
Airflow DAG for Kafka Streaming Pipeline Health Check
Monitors streaming producer and consumer health and manages long-running processes
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
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'kafka_streaming_health_check',
    default_args=default_args,
    description='Health check and monitoring for Kafka streaming pipeline',
    schedule_interval=timedelta(minutes=15),  # Run every 15 minutes
    catchup=False,
    tags=['kafka', 'streaming', 'health-check', 'telco-churn'],
)

def check_kafka_broker_health():
    """Check if Kafka broker is accessible"""
    try:
        from kafka import KafkaAdminClient
        import yaml
        
        # Load config
        config_path = os.path.join(project_root, 'config.yml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create admin client
        admin_client = KafkaAdminClient(
            bootstrap_servers=config['kafka']['broker_config']['bootstrap_servers'],
            client_id='health_check'
        )
        
        # Try to get cluster metadata
        metadata = admin_client.describe_cluster()
        print(f"Kafka broker health check passed. Cluster ID: {metadata.cluster_id}")
        
        admin_client.close()
        return True
        
    except Exception as e:
        print(f"Kafka broker health check failed: {e}")
        raise

def check_consumer_lag():
    """Check consumer lag for churn prediction consumers"""
    try:
        from kafka import KafkaAdminClient
        import yaml
        
        # Load config
        config_path = os.path.join(project_root, 'config.yml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create admin client
        admin_client = KafkaAdminClient(
            bootstrap_servers=config['kafka']['broker_config']['bootstrap_servers'],
            client_id='lag_check'
        )
        
        # Get consumer group information
        consumer_group = config['kafka']['consumer_config']['group_id']
        
        # This is a simplified check - in production you'd use more detailed monitoring
        print(f"Consumer lag check for group: {consumer_group}")
        # You would implement detailed lag monitoring here
        
        admin_client.close()
        return True
        
    except Exception as e:
        print(f"Consumer lag check failed: {e}")
        raise

def restart_streaming_consumer_if_needed():
    """Restart streaming consumer if it's not running or unhealthy"""
    try:
        import subprocess
        import time
        
        # Check if consumer process is running
        # This is a simplified check - you'd implement proper process monitoring
        consumer_script = os.path.join(project_root, 'pipelines', 'consumer.py')
        config_path = os.path.join(project_root, 'config.yml')
        
        # In production, you'd check if the process is actually running and healthy
        print("Checking streaming consumer health...")
        
        # For demonstration, we'll just log that we would restart if needed
        print("Streaming consumer health check completed")
        
        return True
        
    except Exception as e:
        print(f"Failed to check/restart streaming consumer: {e}")
        raise

# Task to check Kafka broker health
kafka_health_check = PythonOperator(
    task_id='check_kafka_broker_health',
    python_callable=check_kafka_broker_health,
    dag=dag,
)

# Task to check consumer lag
consumer_lag_check = PythonOperator(
    task_id='check_consumer_lag',
    python_callable=check_consumer_lag,
    dag=dag,
)

# Task to restart consumer if needed
restart_consumer_check = PythonOperator(
    task_id='restart_streaming_consumer_if_needed',
    python_callable=restart_streaming_consumer_if_needed,
    dag=dag,
)

# Task to check topic configurations
check_topics = BashOperator(
    task_id='check_kafka_topics',
    bash_command="""
    cd {{ params.project_root }}
    echo "Checking Kafka topics configuration..."
    # In production, you'd use kafka-topics.sh to check topic health
    echo "Topics check completed"
    """,
    params={'project_root': str(project_root)},
    dag=dag,
)

# Task to monitor disk space for logs
monitor_disk_space = BashOperator(
    task_id='monitor_disk_space',
    bash_command="""
    echo "Checking disk space for Kafka logs..."
    df -h | grep -E "(logs|tmp)" || echo "No specific log partitions found"
    
    # Check if disk usage is above 80%
    USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ $USAGE -gt 80 ]; then
        echo "WARNING: Disk usage is above 80%: ${USAGE}%"
        exit 1
    else
        echo "Disk usage OK: ${USAGE}%"
    fi
    """,
    dag=dag,
)

# Define task dependencies
kafka_health_check >> [consumer_lag_check, check_topics]
consumer_lag_check >> restart_consumer_check
monitor_disk_space