# Windows PowerShell equivalent of Makefile commands
# Usage: .\run.ps1 <command>

param(
    [Parameter(Mandatory=$false)]
    [string]$Command = "help"
)

# Configuration
$VENV_ACTIVATE = ".\.venv\Scripts\Activate.ps1"

# Helper function to run commands in virtual environment
function Invoke-InVenv {
    param([string]$ScriptBlock)
    
    if (-not (Test-Path $VENV_ACTIVATE)) {
        Write-Host "Virtual environment not found. Run: .\run.ps1 install" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $VENV_ACTIVATE
    Write-Host "Executing: $ScriptBlock" -ForegroundColor Cyan
    Invoke-Expression $ScriptBlock
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "help" {
        Write-Host @"
Available targets:
  .\run.ps1 install             - Install project dependencies and set up environment
  .\run.ps1 data-pipeline       - Run the data pipeline
  .\run.ps1 train-pipeline      - Run the training pipeline
  .\run.ps1 streaming-inference - Run the streaming inference pipeline with the sample JSON
  .\run.ps1 run-all             - Run all pipelines in sequence
  .\run.ps1 spark-pipeline      - Run the unified Spark pipeline (distributed processing)
  .\run.ps1 spark-data          - Run Spark data processing pipeline only
  .\run.ps1 spark-train         - Run Spark model training pipeline only
  .\run.ps1 test-spark          - Test Spark pipeline setup and functionality
  .\run.ps1 clean               - Clean up artifacts
  .\run.ps1 airflow-init        - Initialize Apache Airflow
  .\run.ps1 airflow-start       - Start Airflow in standalone mode
  .\run.ps1 airflow-webserver   - Start Airflow webserver
  .\run.ps1 airflow-scheduler   - Start Airflow scheduler
  .\run.ps1 airflow-kill        - Kill all running Airflow processes
  .\run.ps1 airflow-reset       - Reset Airflow database and fix login issues
"@ -ForegroundColor Green
    }

    "install" {
        Write-Host "Installing project dependencies and setting up environment..." -ForegroundColor Green
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv .venv
        Write-Host "Activating virtual environment and installing dependencies..." -ForegroundColor Yellow
        & $VENV_ACTIVATE
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        Write-Host "Installation completed successfully!" -ForegroundColor Green
        Write-Host "To activate the virtual environment, run: $VENV_ACTIVATE" -ForegroundColor Cyan
    }

    "clean" {
        Write-Host "Cleaning up artifacts..." -ForegroundColor Green
        if (Test-Path "artifacts\models") { Remove-Item "artifacts\models\*" -Recurse -Force -ErrorAction SilentlyContinue }
        if (Test-Path "artifacts\evaluation") { Remove-Item "artifacts\evaluation\*" -Recurse -Force -ErrorAction SilentlyContinue }
        if (Test-Path "artifacts\predictions") { Remove-Item "artifacts\predictions\*" -Recurse -Force -ErrorAction SilentlyContinue }
        if (Test-Path "data\processed") { Remove-Item "data\processed\*" -Recurse -Force -ErrorAction SilentlyContinue }
        Write-Host "Cleanup completed!" -ForegroundColor Green
    }

    "data-pipeline" {
        Write-Host "Running data pipeline..." -ForegroundColor Green
        Invoke-InVenv "python pipelines/data_pipeline.py"
        Write-Host "Data pipeline completed successfully!" -ForegroundColor Green
    }

    "train-pipeline" {
        Write-Host "Running training pipeline..." -ForegroundColor Green
        Invoke-InVenv "python pipelines/training_pipeline.py"
    }

    "streaming-inference" {
        Write-Host "Running streaming inference pipeline with sample JSON..." -ForegroundColor Green
        Invoke-InVenv "python pipelines/streaming_inference_pipeline.py"
    }

    "run-all" {
        Write-Host "Running all pipelines in sequence..." -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "Step 1: Running data pipeline" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Cyan
        Invoke-InVenv "python pipelines/data_pipeline.py"
        Write-Host "`n========================================" -ForegroundColor Cyan
        Write-Host "Step 2: Running training pipeline" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Cyan
        Invoke-InVenv "python pipelines/training_pipeline.py"
        Write-Host "`n========================================" -ForegroundColor Cyan
        Write-Host "Step 3: Running streaming inference pipeline" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Cyan
        Invoke-InVenv "python pipelines/streaming_inference_pipeline.py"
        Write-Host "`n========================================" -ForegroundColor Cyan
        Write-Host "All pipelines completed successfully!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
    }

    "spark-pipeline" {
        Write-Host "Running unified Spark pipeline for distributed processing..." -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "SPARK DISTRIBUTED PROCESSING PIPELINE" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "This will run both data processing and model training using Spark"
        Write-Host "Check Spark UI at http://localhost:4040 during execution"
        Invoke-InVenv "python pipelines/unified_spark_pipeline.py"
        Write-Host "Spark pipeline completed successfully!" -ForegroundColor Green
    }

    "spark-data" {
        Write-Host "Running Spark data processing pipeline..." -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "SPARK DATA PROCESSING" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Cyan
        Invoke-InVenv "python pipelines/spark_data_pipeline.py"
        Write-Host "Spark data processing completed!" -ForegroundColor Green
    }

    "spark-train" {
        Write-Host "Running Spark model training pipeline..." -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "SPARK MODEL TRAINING" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "Note: This requires processed data from spark-data pipeline"
        Invoke-InVenv "python pipelines/spark_model_trainer.py"
        Write-Host "Spark model training completed!" -ForegroundColor Green
    }

    "test-spark" {
        Write-Host "Testing Spark pipeline setup..." -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "SPARK PIPELINE TESTS" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "This will validate Spark imports, session creation, and basic functionality"
        Invoke-InVenv "python tests/test_spark_pipeline.py"
        Write-Host "========================================" -ForegroundColor Cyan
    }

    "airflow-init" {
        Write-Host "Initializing Apache Airflow..." -ForegroundColor Green
        $env:AIRFLOW_HOME = "$(Get-Location)\.airflow"
        Write-Host "Setting AIRFLOW_HOME to: $env:AIRFLOW_HOME" -ForegroundColor Yellow
        
        if (-not (Test-Path $env:AIRFLOW_HOME)) {
            New-Item -ItemType Directory -Path $env:AIRFLOW_HOME -Force | Out-Null
        }
        
        Invoke-InVenv 'pip install "apache-airflow>=2.10.0,<3.0.0" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.3/constraints-3.9.txt"'
        Invoke-InVenv "pip install apache-airflow-providers-apache-spark"
        Invoke-InVenv "airflow db migrate"
        Invoke-InVenv "airflow users create -u admin -p admin -r Admin -e admin@example.com -f Admin -l User"
        
        $dagsDir = "$env:AIRFLOW_HOME\dags"
        if (-not (Test-Path $dagsDir)) {
            New-Item -ItemType Directory -Path $dagsDir -Force | Out-Null
        }
        
        if (Test-Path "dags") {
            Copy-Item "dags\*.py" $dagsDir -Force -ErrorAction SilentlyContinue
        }
        
        Write-Host "Airflow initialized successfully!" -ForegroundColor Green
    }

    "airflow-webserver" {
        Write-Host "Starting Airflow webserver on http://localhost:8080..." -ForegroundColor Green
        $env:AIRFLOW_HOME = "$(Get-Location)\.airflow"
        Invoke-InVenv "airflow webserver --port 8080"
    }

    "airflow-scheduler" {
        Write-Host "Starting Airflow scheduler..." -ForegroundColor Green
        $env:AIRFLOW_HOME = "$(Get-Location)\.airflow"
        Invoke-InVenv "airflow scheduler"
    }

    "airflow-start" {
        Write-Host "Starting Airflow in standalone mode..." -ForegroundColor Green
        Write-Host "Webserver will be available at http://localhost:8080" -ForegroundColor Cyan
        Write-Host "Login with: admin / admin" -ForegroundColor Cyan
        $env:AIRFLOW_HOME = "$(Get-Location)\.airflow"
        $env:PYTHONWARNINGS = "ignore::DeprecationWarning"
        
        $dagsDir = "$env:AIRFLOW_HOME\dags"
        if (Test-Path "dags") {
            if (-not (Test-Path $dagsDir)) {
                New-Item -ItemType Directory -Path $dagsDir -Force | Out-Null
            }
            Copy-Item "dags\*.py" $dagsDir -Force -ErrorAction SilentlyContinue
        }
        
        Invoke-InVenv "airflow standalone"
    }

    "airflow-kill" {
        Write-Host "Killing all Airflow processes..." -ForegroundColor Green
        Get-Process -Name "*airflow*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        
        # Kill processes using Airflow ports
        $ports = @(8080, 8793, 8794)
        foreach ($port in $ports) {
            $netstat = netstat -ano | Select-String ":$port "
            if ($netstat) {
                $pids = $netstat | ForEach-Object { ($_ -split '\s+')[-1] } | Sort-Object -Unique
                foreach ($pid in $pids) {
                    if ($pid -and $pid -ne "0") {
                        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                    }
                }
            }
        }
        
        Write-Host "All Airflow processes killed and ports freed successfully!" -ForegroundColor Green
    }

    "airflow-reset" {
        Write-Host "Resetting Airflow database and fixing login issues..." -ForegroundColor Green
        
        # Kill processes first
        Get-Process -Name "*airflow*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        
        $env:AIRFLOW_HOME = "$(Get-Location)\.airflow"
        
        # Remove old database and logs
        if (Test-Path "$env:AIRFLOW_HOME\airflow.db") {
            Remove-Item "$env:AIRFLOW_HOME\airflow.db" -Force
        }
        if (Test-Path "$env:AIRFLOW_HOME\logs") {
            Remove-Item "$env:AIRFLOW_HOME\logs\*" -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        $env:PYTHONWARNINGS = "ignore::DeprecationWarning"
        Invoke-InVenv "airflow db migrate"
        Invoke-InVenv "airflow users create -u admin -f Admin -l User -p admin -r Admin -e admin@example.com"
        
        $dagsDir = "$env:AIRFLOW_HOME\dags"
        if (Test-Path "dags") {
            if (-not (Test-Path $dagsDir)) {
                New-Item -ItemType Directory -Path $dagsDir -Force | Out-Null
            }
            Copy-Item "dags\*.py" $dagsDir -Force -ErrorAction SilentlyContinue
        }
        
        Write-Host "Airflow reset complete! Login: admin/admin" -ForegroundColor Green
        Write-Host "Start with: .\run.ps1 airflow-start" -ForegroundColor Cyan
    }

    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run '.\run.ps1 help' to see all available commands" -ForegroundColor Yellow
        exit 1
    }
}