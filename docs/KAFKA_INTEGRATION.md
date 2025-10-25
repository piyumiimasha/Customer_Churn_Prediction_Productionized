# Kafka Integration for Telco Customer Churn Prediction

This document provides comprehensive guidance for setting up and running the Kafka integration pipeline for real-time customer churn prediction.

## Overview

The Kafka integration extends the existing ML pipeline with real-time streaming capabilities:

- **Producer**: Streams customer events to Kafka topics
- **Consumer**: Consumes events, applies ML model, and publishes predictions
- **Airflow DAGs**: Orchestrates batch processing and monitors streaming health

## Architecture

```
Telco Customer Data → Producer → Kafka (telco.raw.customers) 
                                        ↓
ML Predictions ← Consumer ← Kafka (telco.churn.predictions)
```

### Kafka Topics

| Topic | Purpose | Key | Value Schema |
|-------|---------|-----|--------------|
| `telco.raw.customers` | Customer events input | customerID | Customer record JSON |
| `telco.churn.predictions` | Churn predictions output | customerID | Prediction result JSON |
| `telco.deadletter` | Invalid/failed records | customerID | Error information |

## Setup Instructions

### Prerequisites

1. **Apache Kafka** (2.8+)
   - Download from [Apache Kafka](https://kafka.apache.org/downloads)
   - Ensure `kafka-topics.sh` (Linux/Mac) or `kafka-topics.bat` (Windows) is in PATH

2. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Trained ML Model**
   - Ensure `artifacts/models/telco_analysis.joblib` exists
   - Run Mini Project 1 training pipeline if needed

### Quick Setup

Run the automated setup script:

```bash
python setup_kafka.py
```

This script will:
- Install Python dependencies
- Check Kafka installation
- Create necessary directories
- Create Kafka topics
- Validate model artifacts
- Run connectivity tests

### Manual Setup

1. **Start Kafka Services**
   ```bash
   # Start Zookeeper
   bin/zookeeper-server-start.sh config/zookeeper.properties
   
   # Start Kafka Broker
   bin/kafka-server-start.sh config/server.properties
   ```

2. **Create Topics**
   ```bash
   # Create raw customers topic
   kafka-topics.sh --create --topic telco.raw.customers --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
   
   # Create predictions topic
   kafka-topics.sh --create --topic telco.churn.predictions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
   
   # Create dead letter topic
   kafka-topics.sh --create --topic telco.deadletter --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
   ```

## Usage

### Producer

The producer streams customer data to Kafka in two modes:

#### Streaming Mode
Continuously samples and streams customer records:

```bash
python pipelines/producer.py \
    --mode streaming \
    --data-path data/hmQOVnDvRN.xls \
    --events-per-sec 10
```

**Parameters:**
- `--mode`: `streaming` for continuous streaming
- `--data-path`: Path to customer dataset
- `--events-per-sec`: Events per second (default: 10)

#### Batch Mode
Processes dataset in chunks with checkpoint support:

```bash
python pipelines/producer.py \
    --mode batch \
    --data-path data/hmQOVnDvRN.xls \
    --batch-size 1000
```

**Parameters:**
- `--mode`: `batch` for batch processing
- `--data-path`: Path to customer dataset
- `--batch-size`: Records per batch (default: 1000)
- `--no-resume`: Disable checkpoint resume

### Consumer

The consumer processes events and publishes predictions:

#### Streaming Mode
Continuous real-time processing:

```bash
python pipelines/consumer.py --mode streaming
```

#### Batch Mode
Process a defined batch of messages:

```bash
python pipelines/consumer.py \
    --mode batch \
    --max-records 1000 \
    --timeout-ms 60000
```

**Parameters:**
- `--max-records`: Maximum records to process
- `--timeout-ms`: Timeout for polling messages

## Message Schemas

### Input Message (telco.raw.customers)

```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85,
  "Churn": "No",
  "event_ts": "2025-10-25T10:00:00Z"
}
```

### Output Message (telco.churn.predictions)

```json
{
  "customerID": "7590-VHVEG",
  "churn_probability": 0.82,
  "prediction": "Yes",
  "event_ts": "2025-10-25T10:00:00Z",
  "processed_ts": "2025-10-25T10:00:01Z"
}
```

## Airflow DAGs (Bonus)

Two DAGs are provided for orchestration:

### 1. Streaming Health Check DAG (`kafka_streaming_dag.py`)

**Schedule**: Every 15 minutes  
**Purpose**: Monitor streaming pipeline health

**Tasks:**
- Check Kafka broker health
- Monitor consumer lag
- Restart consumers if needed
- Monitor disk space
- Validate topic configurations

### 2. Batch Processing DAG (`kafka_batch_dag.py`)

**Schedule**: Every hour  
**Purpose**: Run complete batch pipeline

**Tasks:**
- Validate data quality
- Run batch producer
- Wait for message availability
- Run batch consumer
- Generate processing summary
- Cleanup and optimize
- Send notifications

### Running DAGs

1. **Start Airflow**
   ```bash
   airflow webserver --port 8080
   airflow scheduler
   ```

2. **Access Web UI**: http://localhost:8080

3. **Enable DAGs** in the web interface

## Monitoring and Observability

### Kafka Topic Monitoring

```bash
# List topics
kafka-topics.sh --list --bootstrap-server localhost:9092

# Describe topic
kafka-topics.sh --describe --topic telco.raw.customers --bootstrap-server localhost:9092

# Monitor consumer groups
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group churn_prediction_consumers
```

### Message Consumption

```bash
# Consume raw customer events
kafka-console-consumer.sh --topic telco.raw.customers --from-beginning --bootstrap-server localhost:9092

# Consume predictions
kafka-console-consumer.sh --topic telco.churn.predictions --from-beginning --bootstrap-server localhost:9092
```

### Logs and Metrics

- **Application logs**: Check console output from producer/consumer
- **Batch summaries**: `artifacts/evaluation/batch_summary_*.json`
- **Checkpoints**: `checkpoints/producer_checkpoint.txt`

## Configuration

Key configuration options in `config.yml`:

```yaml
kafka:
  broker_config:
    bootstrap_servers: "localhost:9092"
    client_id: "telco_churn_pipeline"
    
  topics:
    raw_customers: "telco.raw.customers"
    churn_predictions: "telco.churn.predictions"
    dead_letter: "telco.deadletter"
    
  streaming:
    events_per_second: 10
    
  batch:
    batch_size: 1000
    checkpoint_dir: "checkpoints/"
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_kafka_integration.py -v
```

**Test Coverage:**
- Producer message formatting
- Consumer preprocessing pipeline
- End-to-end message flow
- Error handling
- Checkpoint functionality

## Troubleshooting

### Common Issues

1. **Kafka Connection Failed**
   ```
   Error: NoBrokersAvailable
   Solution: Ensure Kafka broker is running on localhost:9092
   ```

2. **Topic Not Found**
   ```
   Error: UnknownTopicOrPartitionError
   Solution: Run topic creation commands
   ```

3. **Model Not Found**
   ```
   Error: FileNotFoundError: Model not found
   Solution: Train and save the model first
   ```

4. **Consumer Lag**
   ```
   Issue: Consumer falling behind producer
   Solution: Increase consumer instances or batch size
   ```

### Performance Tuning

1. **Producer Optimization**
   - Increase `batch_size` and `linger_ms` for throughput
   - Use compression (`gzip`)
   - Tune `buffer_memory`

2. **Consumer Optimization**
   - Increase `max_poll_records`
   - Use multiple consumer instances
   - Optimize ML model inference

3. **Kafka Optimization**
   - Increase topic partitions for parallelism
   - Tune `num.network.threads` and `num.io.threads`
   - Configure appropriate `log.retention.hours`

## Integration Examples

### Example 1: Simple Streaming Pipeline

```bash
# Terminal 1: Start streaming producer
python pipelines/producer.py --mode streaming --data-path data/hmQOVnDvRN.xls --events-per-sec 5

# Terminal 2: Start streaming consumer
python pipelines/consumer.py --mode streaming

# Terminal 3: Monitor predictions
kafka-console-consumer.sh --topic telco.churn.predictions --bootstrap-server localhost:9092
```

### Example 2: Batch Processing with Airflow

1. Enable the batch DAG in Airflow UI
2. Trigger manual run or wait for scheduled execution
3. Monitor progress in Airflow web interface
4. Check batch summary in `artifacts/evaluation/`

## Security Considerations

For production deployments:

1. **Enable SSL/TLS**
2. **Configure SASL authentication**
3. **Set up proper ACLs**
4. **Network security (VPC, firewall rules)**
5. **Encrypt sensitive data in messages**

## Performance Benchmarks

Expected performance on standard hardware:

- **Producer**: 1000+ events/second
- **Consumer**: 500+ predictions/second
- **End-to-end latency**: <100ms (streaming mode)
- **Batch processing**: 10,000+ records/minute

## Future Enhancements

1. **Schema Registry integration**
2. **Advanced monitoring with Prometheus/Grafana**
3. **Multi-model ensemble predictions**
4. **A/B testing framework**
5. **Real-time feature engineering**
6. **Auto-scaling consumer groups**

## Support

For issues and questions:
1. Check logs and error messages
2. Review configuration settings
3. Validate Kafka cluster health
4. Check model artifacts and dependencies