.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all spark-pipeline spark-data spark-train test-spark help

# Default Python interpreter
PYTHON = python
VENV = .venv/bin/activate

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  make install             - Install project dependencies and set up environment"
	@echo "  make data-pipeline       - Run the data pipeline"
	@echo "  make train-pipeline      - Run the training pipeline"
	@echo "  make streaming-inference - Run the streaming inference pipeline with the sample JSON"
	@echo "  make run-all             - Run all pipelines in sequence"
	@echo "  make spark-pipeline      - Run the unified Spark pipeline (distributed processing)"
	@echo "  make spark-data          - Run Spark data processing pipeline only"
	@echo "  make spark-train         - Run Spark model training pipeline only"
	@echo "  make test-spark          - Test Spark pipeline setup and functionality"
	@echo "  make clean               - Clean up artifacts"

# Install project dependencies and set up environment
install:
	@echo "Installing project dependencies and setting up environment..."
	@echo "Creating virtual environment..."
	@python3 -m venv .venv
	@echo "Activating virtual environment and installing dependencies..."
	@source .venv/bin/activate && pip install --upgrade pip
	@source .venv/bin/activate && pip install -r requirements.txt
	@echo "Installation completed successfully!"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

# Clean up
clean:
	@echo "Cleaning up artifacts..."
	rm -rf artifacts/models/*
	rm -rf artifacts/evaluation/*
	rm -rf artifacts/predictions/*
	rm -rf data/processed/*
	@echo "Cleanup completed!"



# Run data pipeline
data-pipeline:
	@echo "Running data pipeline..."
	@source $(VENV) && $(PYTHON) pipelines/data_pipeline.py
	@echo "Data pipeline completed successfully!"

# Run training pipeline
train-pipeline:
	@echo "Running training pipeline..."
	@source $(VENV) && $(PYTHON) pipelines/training_pipeline.py

# Run streaming inference pipeline with sample JSON
streaming-inference:
	@echo "Running streaming inference pipeline with sample JSON..."
	@source $(VENV) && $(PYTHON) pipelines/streaming_inference_pipeline.py

# Run all pipelines in sequence
run-all:
	@echo "Running all pipelines in sequence..."
	@echo "========================================"
	@echo "Step 1: Running data pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/data_pipeline.py
	@echo "\n========================================"
	@echo "Step 2: Running training pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/training_pipeline.py
	@echo "\n========================================"
	@echo "Step 3: Running streaming inference pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/streaming_inference_pipeline.py
	@echo "\n========================================"
	@echo "All pipelines completed successfully!"
	@echo "========================================"

# Run unified Spark pipeline (distributed processing)
spark-pipeline:
	@echo "Running unified Spark pipeline for distributed processing..."
	@echo "========================================"
	@echo "🚀 SPARK DISTRIBUTED PROCESSING PIPELINE"
	@echo "========================================"
	@echo "This will run both data processing and model training using Spark"
	@echo "Check Spark UI at http://localhost:4040 during execution"
	@source $(VENV) && $(PYTHON) pipelines/unified_spark_pipeline.py
	@echo "✅ Spark pipeline completed successfully!"

# Run Spark data processing pipeline only
spark-data:
	@echo "Running Spark data processing pipeline..."
	@echo "========================================"
	@echo "📊 SPARK DATA PROCESSING"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) src/spark_data_pipeline.py
	@echo "✅ Spark data processing completed!"

# Run Spark model training pipeline only
spark-train:
	@echo "Running Spark model training pipeline..."
	@echo "========================================"
	@echo "🤖 SPARK MODEL TRAINING"
	@echo "========================================"
	@echo "Note: This requires processed data from spark-data pipeline"
	@source $(VENV) && $(PYTHON) src/spark_model_trainer.py
	@echo "✅ Spark model training completed!"

# Test Spark pipeline setup and functionality
test-spark:
	@echo "Testing Spark pipeline setup..."
	@echo "========================================"
	@echo "🧪 SPARK PIPELINE TESTS"
	@echo "========================================"
	@echo "This will validate Spark imports, session creation, and basic functionality"
	@source $(VENV) && $(PYTHON) tests/test_spark_pipeline.py
	@echo "========================================"


mlflow-ui:
	@echo "Launching MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop the server"
	@source $(VENV) && mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)

# Stop all running MLflow servers
stop-all:
	@echo "Stopping all MLflow servers..."
	@echo "Finding MLflow processes on port $(MLFLOW_PORT)..."
	@-lsof -ti:$(MLFLOW_PORT) | xargs kill -9 2>/dev/null || true
	@echo "Finding other MLflow UI processes..."
	@-ps aux | grep '[m]lflow ui' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@-ps aux | grep '[g]unicorn.*mlflow' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@echo "✅ All MLflow servers have been stopped"