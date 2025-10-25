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
        Write-Host "Available targets:" -ForegroundColor Green
        Write-Host "  .\run.ps1 install             - Install project dependencies and set up environment"
        Write-Host "  .\run.ps1 data-pipeline       - Run the data pipeline"
        Write-Host "  .\run.ps1 train-pipeline      - Run the training pipeline"
        Write-Host "  .\run.ps1 streaming-inference - Run the streaming inference pipeline with the sample JSON"
        Write-Host "  .\run.ps1 run-all             - Run all pipelines in sequence"
        Write-Host "  .\run.ps1 spark-pipeline      - Run the unified Spark pipeline (distributed processing)"
        Write-Host "  .\run.ps1 spark-data          - Run Spark data processing pipeline only"
        Write-Host "  .\run.ps1 spark-train         - Run Spark model training pipeline only"
        Write-Host "  .\run.ps1 test-spark          - Test Spark pipeline setup and functionality"
        Write-Host "  .\run.ps1 clean               - Clean up artifacts"
        Write-Host "  .\run.ps1 airflow-init            - Initialize Apache Airflow"
        Write-Host "  .\run.ps1 airflow-start           - Start Airflow in standalone mode"
        Write-Host "  .\run.ps1 airflow-webserver-only  - Start Airflow webserver only (Windows compatible)"
        Write-Host "  .\run.ps1 airflow-webserver       - Start Airflow webserver"
        Write-Host "  .\run.ps1 airflow-scheduler       - Start Airflow scheduler"
        Write-Host "  .\run.ps1 airflow-kill            - Kill all running Airflow processes"
        Write-Host "  .\run.ps1 airflow-reset           - Reset Airflow database and fix login issues"
        Write-Host ""
        Write-Host "🌊 KAFKA STREAMING COMMANDS:" -ForegroundColor Cyan
        Write-Host "  .\run.ps1 kafka-help              - Show detailed Kafka commands help"
        Write-Host "  .\run.ps1 kafka-check             - Check Kafka installation and status"
        Write-Host "  .\run.ps1 kafka-start             - Start Zookeeper and Kafka server"
        Write-Host "  .\run.ps1 kafka-start-bg          - Start native Kafka broker in background"
        Write-Host "  .\run.ps1 kafka-stop              - Stop native Kafka broker"
        Write-Host "  .\run.ps1 kafka-topics            - Create churn prediction topics"
        Write-Host "  .\run.ps1 kafka-list-topics       - List all Kafka topics"
        Write-Host "  .\run.ps1 kafka-producer-stream   - Start streaming producer (1 event/sec, 5 mins)"
        Write-Host "  .\run.ps1 kafka-producer-batch    - Start batch producer"
        Write-Host "  .\run.ps1 kafka-consumer-stream   - Start streaming consumer (real-time ML)"
        Write-Host "  .\run.ps1 kafka-consumer-batch    - Start batch consumer"
        Write-Host "  .\run.ps1 kafka-monitor           - Monitor prediction results"
        Write-Host "  .\run.ps1 kafka-demo              - Run complete Kafka demo"
        Write-Host "  .\run.ps1 kafka-clean             - Clean up Kafka data and topics"
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

    "airflow-webserver-only" {
        Write-Host "Starting Airflow webserver only (Windows compatible)..." -ForegroundColor Green
        Write-Host "Webserver will be available at http://localhost:8080" -ForegroundColor Cyan
        Write-Host "Login with: admin / admin" -ForegroundColor Cyan
        Write-Host "Note: Scheduler disabled due to Windows compatibility" -ForegroundColor Yellow
        
        $env:AIRFLOW_HOME = "$(Get-Location)\.airflow"
        $env:PYTHONWARNINGS = "ignore::DeprecationWarning"
        
        # Copy DAGs
        $dagsDir = "$env:AIRFLOW_HOME\dags"
        if (Test-Path "dags") {
            if (-not (Test-Path $dagsDir)) {
                New-Item -ItemType Directory -Path $dagsDir -Force | Out-Null
            }
            Copy-Item "dags\*.py" $dagsDir -Force -ErrorAction SilentlyContinue
            Write-Host "DAGs copied to: $dagsDir" -ForegroundColor Green
        }
        
        # Start only webserver (no scheduler/triggerer)
        Write-Host "Starting webserver..." -ForegroundColor Yellow
        Invoke-InVenv "airflow webserver --port 8080"
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

    # ========================================================================================
    # KAFKA STREAMING COMMANDS
    # ========================================================================================
    
    "kafka-help" {
        Write-Host "🔧 Kafka Commands Help" -ForegroundColor Green
        Write-Host "=============================================="
        Write-Host "📋 Setup Commands:" -ForegroundColor Yellow
        Write-Host "  .\run.ps1 kafka-check             - Check Kafka installation and KAFKA_HOME"
        Write-Host "  .\run.ps1 kafka-start             - Start Zookeeper and Kafka server (interactive)"
        Write-Host "  .\run.ps1 kafka-start-bg          - Start native Kafka broker in background"
        Write-Host "  .\run.ps1 kafka-stop              - Stop native Kafka broker using PID files"
        Write-Host "  .\run.ps1 kafka-topics            - Create churn prediction topics (churn_predictions, churn_predictions_scored)"
        Write-Host "  .\run.ps1 kafka-list-topics       - List all Kafka topics"
        Write-Host ""
        Write-Host "📊 Data Production Commands:" -ForegroundColor Yellow
        Write-Host "  .\run.ps1 kafka-producer-stream   - Stream real customer events (1 event/sec for 5 mins)"
        Write-Host "  .\run.ps1 kafka-producer-batch    - Batch produce events"
        Write-Host ""
        Write-Host "🤖 ML Processing Commands:" -ForegroundColor Yellow
        Write-Host "  .\run.ps1 kafka-consumer-stream   - Continuous ML consumer (real-time)"
        Write-Host "  .\run.ps1 kafka-consumer-batch    - Batch ML consumer"
        Write-Host ""
        Write-Host "🔍 Monitoring Commands:" -ForegroundColor Yellow
        Write-Host "  .\run.ps1 kafka-monitor           - Monitor prediction results"
        Write-Host "  .\run.ps1 kafka-demo              - Run complete demo"
        Write-Host ""
        Write-Host "🧹 Utility Commands:" -ForegroundColor Yellow
        Write-Host "  .\run.ps1 kafka-clean             - Clean up Kafka data and topics"
        Write-Host ""
        Write-Host "💡 Quick Start:" -ForegroundColor Cyan
        Write-Host "  1. .\run.ps1 kafka-start-bg"
        Write-Host "  2. .\run.ps1 kafka-topics"
        Write-Host "  3. .\run.ps1 kafka-producer-batch  (Terminal 1)"
        Write-Host "  4. .\run.ps1 kafka-consumer-batch  (Terminal 2)"
        Write-Host "  5. .\run.ps1 kafka-monitor         (Terminal 3)"
        Write-Host ""
        Write-Host "🔄 Streaming Demo:" -ForegroundColor Cyan
        Write-Host "  1. .\run.ps1 kafka-consumer-stream  (Terminal 1)"
        Write-Host "  2. .\run.ps1 kafka-producer-stream  (Terminal 2)"
        Write-Host "  3. .\run.ps1 kafka-monitor          (Terminal 3)"
    }

    "kafka-check" {
        Write-Host "🔍 Checking Kafka installation and status..." -ForegroundColor Green
        
        # Check KAFKA_HOME environment variable
        $kafkaHome = $env:KAFKA_HOME
        if (-not $kafkaHome) {
            Write-Host "⚠️  KAFKA_HOME environment variable not set" -ForegroundColor Yellow
            Write-Host "💡 Please set KAFKA_HOME to your Kafka installation directory" -ForegroundColor Cyan
            Write-Host "   Example: `$env:KAFKA_HOME = 'C:\kafka'" -ForegroundColor Gray
            
            # Try default location
            $kafkaHome = "C:\kafka"
            Write-Host "🔍 Checking default location: $kafkaHome" -ForegroundColor Yellow
        }
        
        # Check if Kafka is installed
        if (Test-Path "$kafkaHome\bin\windows\kafka-server-start.bat") {
            Write-Host "✅ Kafka installation found at: $kafkaHome" -ForegroundColor Green
            Write-Host "📁 KAFKA_HOME: $kafkaHome" -ForegroundColor Cyan
        } else {
            Write-Host "❌ Kafka not found at $kafkaHome" -ForegroundColor Red
            Write-Host "💡 Please ensure Kafka is properly installed and KAFKA_HOME is set correctly" -ForegroundColor Yellow
            return
        }
        
        # Check if Kafka is running using PID files
        $isKafkaRunning = $false
        $isZookeeperRunning = $false
        
        if (Test-Path "runtime\pids\kafka.pid") {
            $kafkaPid = Get-Content "runtime\pids\kafka.pid" -ErrorAction SilentlyContinue
            if ($kafkaPid -and (Get-Process -Id $kafkaPid -ErrorAction SilentlyContinue)) {
                Write-Host "✅ Kafka broker is running (PID: $kafkaPid)" -ForegroundColor Green
                $isKafkaRunning = $true
            } else {
                Write-Host "⚠️  Kafka PID file exists but process not found" -ForegroundColor Yellow
            }
        }
        
        if (Test-Path "runtime\pids\zookeeper.pid") {
            $zkPid = Get-Content "runtime\pids\zookeeper.pid" -ErrorAction SilentlyContinue
            if ($zkPid -and (Get-Process -Id $zkPid -ErrorAction SilentlyContinue)) {
                Write-Host "✅ Zookeeper is running (PID: $zkPid)" -ForegroundColor Green
                $isZookeeperRunning = $true
            }
        }
        
        # Test broker connection
        if ($isKafkaRunning) {
            $kafkaTopicsCmd = "$kafkaHome\bin\windows\kafka-topics.bat"
            try {
                & $kafkaTopicsCmd --bootstrap-server localhost:9092 --list 2>$null | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "✅ Kafka broker is accessible at localhost:9092" -ForegroundColor Green
                } else {
                    Write-Host "⚠️  Kafka process running but broker not accessible" -ForegroundColor Yellow
                }
            } catch {
                Write-Host "⚠️  Unable to test broker connection" -ForegroundColor Yellow
            }
        } else {
            Write-Host "⚠️  Kafka broker not running" -ForegroundColor Yellow
            Write-Host "💡 Start with: .\run.ps1 kafka-start-bg" -ForegroundColor Cyan
        }
        
        # Show logs location if available
        if (Test-Path "runtime\kafka.log") {
            Write-Host "📄 Kafka logs: runtime\kafka.log" -ForegroundColor Gray
        }
    }

    "kafka-start" {
        Write-Host "🚀 Starting Kafka services..." -ForegroundColor Green
        $kafkaPath = "C:\kafka"
        
        if (-not (Test-Path "$kafkaPath\bin\windows\kafka-server-start.bat")) {
            Write-Host "❌ Kafka not found at $kafkaPath" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "📁 Creating necessary directories..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path "runtime\kafka-logs" -Force | Out-Null
        New-Item -ItemType Directory -Path "runtime\pids" -Force | Out-Null
        
        Write-Host "🔑 Starting Zookeeper..." -ForegroundColor Cyan
        Start-Process -FilePath "$kafkaPath\bin\windows\zookeeper-server-start.bat" -ArgumentList "$kafkaPath\config\zookeeper.properties" -WindowStyle Minimized
        Write-Host "Waiting for Zookeeper to initialize..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        
        Write-Host "🌐 Starting Kafka broker..." -ForegroundColor Cyan
        Start-Process -FilePath "$kafkaPath\bin\windows\kafka-server-start.bat" -ArgumentList "$kafkaPath\config\server.properties" -WindowStyle Minimized
        Write-Host "Waiting for Kafka broker to initialize..." -ForegroundColor Yellow
        Start-Sleep -Seconds 15
        
        Write-Host "✅ Kafka services started successfully!" -ForegroundColor Green
        Write-Host "Broker available at: localhost:9092" -ForegroundColor Cyan
    }

    "kafka-start-bg" {
        Write-Host "🚀 Starting native Kafka broker in background..." -ForegroundColor Green
        
        # Check if KAFKA_HOME environment variable is set
        $kafkaHome = $env:KAFKA_HOME
        if (-not $kafkaHome) {
            Write-Host "❌ KAFKA_HOME environment variable not set" -ForegroundColor Red
            Write-Host "💡 Please set KAFKA_HOME to your Kafka installation directory" -ForegroundColor Yellow
            Write-Host "   Example: `$env:KAFKA_HOME = 'C:\kafka'" -ForegroundColor Cyan
            exit 1
        }
        
        if (-not (Test-Path "$kafkaHome\bin\windows\kafka-server-start.bat")) {
            Write-Host "❌ Kafka not found at $kafkaHome" -ForegroundColor Red
            Write-Host "💡 Please ensure KAFKA_HOME points to a valid Kafka installation" -ForegroundColor Yellow
            exit 1
        }
        
        # Create runtime directories
        Write-Host "📁 Creating runtime directories..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path "runtime" -Force | Out-Null
        New-Item -ItemType Directory -Path "runtime\pids" -Force | Out-Null
        
        # Start Zookeeper first (required for Kafka)
        Write-Host "🔑 Starting Zookeeper in background..." -ForegroundColor Cyan
        $zkLogFile = "runtime\zookeeper.log"
        $zkProcess = Start-Process -FilePath "$kafkaHome\bin\windows\zookeeper-server-start.bat" -ArgumentList "$kafkaHome\config\zookeeper.properties" -WindowStyle Hidden -RedirectStandardOutput $zkLogFile -RedirectStandardError $zkLogFile -PassThru
        "$($zkProcess.Id)" | Out-File "runtime\pids\zookeeper.pid" -Encoding ASCII
        Start-Sleep -Seconds 8
        
        # Start Kafka broker
        Write-Host "🌐 Starting Kafka broker in background..." -ForegroundColor Cyan
        $kafkaLogFile = "runtime\kafka.log"
        $kafkaProcess = Start-Process -FilePath "$kafkaHome\bin\windows\kafka-server-start.bat" -ArgumentList "$kafkaHome\config\server.properties" -WindowStyle Hidden -RedirectStandardOutput $kafkaLogFile -RedirectStandardError $kafkaLogFile -PassThru
        "$($kafkaProcess.Id)" | Out-File "runtime\pids\kafka.pid" -Encoding ASCII
        Start-Sleep -Seconds 12
        
        Write-Host "✅ Kafka broker started in background (PID: $($kafkaProcess.Id))" -ForegroundColor Green
        Write-Host "📄 Logs: runtime\kafka.log" -ForegroundColor Cyan
        Write-Host "🔍 Zookeeper PID: $($zkProcess.Id)" -ForegroundColor Gray
        Write-Host "🌐 Broker available at: localhost:9092" -ForegroundColor Cyan
    }

    "kafka-stop" {
        Write-Host "🛑 Stopping native Kafka broker..." -ForegroundColor Green
        
        # Check if KAFKA_HOME environment variable is set
        $kafkaHome = $env:KAFKA_HOME
        if (-not $kafkaHome) {
            Write-Host "❌ KAFKA_HOME environment variable not set" -ForegroundColor Red
            exit 1
        }
        
        # Stop using PID file if available
        if (Test-Path "runtime\pids\kafka.pid") {
            $kafkaPid = Get-Content "runtime\pids\kafka.pid" -ErrorAction SilentlyContinue
            if ($kafkaPid) {
                Write-Host "🔍 Found Kafka PID: $kafkaPid" -ForegroundColor Cyan
                try {
                    Stop-Process -Id $kafkaPid -Force -ErrorAction Stop
                    Write-Host "✅ Kafka broker stopped" -ForegroundColor Green
                } catch {
                    Write-Host "⚠️ Failed to stop process $kafkaPid, trying alternative method..." -ForegroundColor Yellow
                }
                Remove-Item "runtime\pids\kafka.pid" -ErrorAction SilentlyContinue
            }
        } else {
            Write-Host "⚠️ PID file not found, trying graceful shutdown..." -ForegroundColor Yellow
            # Try graceful shutdown using Kafka's stop script
            if (Test-Path "$kafkaHome\bin\windows\kafka-server-stop.bat") {
                try {
                    & "$kafkaHome\bin\windows\kafka-server-stop.bat"
                } catch {
                    Write-Host "⚠️ Graceful shutdown failed, forcing process termination..." -ForegroundColor Yellow
                }
            }
        }
        
        # Also stop Zookeeper if PID file exists
        if (Test-Path "runtime\pids\zookeeper.pid") {
            $zkPid = Get-Content "runtime\pids\zookeeper.pid" -ErrorAction SilentlyContinue
            if ($zkPid) {
                Stop-Process -Id $zkPid -Force -ErrorAction SilentlyContinue
                Remove-Item "runtime\pids\zookeeper.pid" -ErrorAction SilentlyContinue
                Write-Host "🔍 Stopped Zookeeper (PID: $zkPid)" -ForegroundColor Gray
            }
        }
        
        # Kill any remaining Java processes related to Kafka/Zookeeper
        Get-Process -Name "java" -ErrorAction SilentlyContinue | Where-Object { 
            $_.CommandLine -like "*kafka*" -or $_.CommandLine -like "*zookeeper*" 
        } | Stop-Process -Force -ErrorAction SilentlyContinue
        
        Write-Host "✅ Kafka services stopped successfully" -ForegroundColor Green
    }

    "kafka-topics" {
        Write-Host "📋 Creating churn prediction topics on native broker..." -ForegroundColor Green
        
        # Use kafka-topics.sh from PATH or KAFKA_HOME
        $kafkaTopicsCmd = "kafka-topics.bat"
        $kafkaHome = $env:KAFKA_HOME
        
        if ($kafkaHome -and (Test-Path "$kafkaHome\bin\windows\kafka-topics.bat")) {
            $kafkaTopicsCmd = "$kafkaHome\bin\windows\kafka-topics.bat"
        }
        
        # Test connection first
        Write-Host "Testing connection to native Kafka broker..." -ForegroundColor Yellow
        try {
            & $kafkaTopicsCmd --bootstrap-server localhost:9092 --list 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "Connection failed"
            }
        } catch {
            Write-Host "❌ Cannot connect to native Kafka broker at localhost:9092" -ForegroundColor Red
            Write-Host "💡 Please start broker with '.\run.ps1 kafka-start-bg' in another terminal" -ForegroundColor Yellow
            exit 1
        }
        
        # Create churn prediction topics
        Write-Host "🔮 Creating churn_predictions topic..." -ForegroundColor Cyan
        & $kafkaTopicsCmd --bootstrap-server localhost:9092 --create --topic churn_predictions --partitions 1 --replication-factor 1 --if-not-exists
        
        Write-Host "🔮 Creating churn_predictions_scored topic..." -ForegroundColor Cyan
        & $kafkaTopicsCmd --bootstrap-server localhost:9092 --create --topic churn_predictions_scored --partitions 1 --replication-factor 1 --if-not-exists
        
        Write-Host "✅ Churn prediction topics created successfully" -ForegroundColor Green
        
        Write-Host "📋 Current topics on native broker:" -ForegroundColor Cyan
        & $kafkaTopicsCmd --bootstrap-server localhost:9092 --list
    }

    "kafka-list-topics" {
        Write-Host "📋 Listing Kafka topics..." -ForegroundColor Green
        $kafkaPath = "C:\kafka"
        
        & "$kafkaPath\bin\windows\kafka-topics.bat" --bootstrap-server localhost:9092 --list
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Topics listed successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to connect to Kafka broker" -ForegroundColor Red
        }
    }

    "kafka-producer-stream" {
        Write-Host "🌊 Starting Kafka streaming producer (real data sampling)..." -ForegroundColor Green
        
        # Use kafka-topics.sh from PATH or KAFKA_HOME for connection test
        $kafkaTopicsCmd = "kafka-topics.bat"
        $kafkaHome = $env:KAFKA_HOME
        
        if ($kafkaHome -and (Test-Path "$kafkaHome\bin\windows\kafka-topics.bat")) {
            $kafkaTopicsCmd = "$kafkaHome\bin\windows\kafka-topics.bat"
        }
        
        # Test Kafka connection
        try {
            & $kafkaTopicsCmd --bootstrap-server localhost:9092 --list 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "Connection failed"
            }
        } catch {
            Write-Host "❌ Cannot connect to native Kafka broker" -ForegroundColor Red
            Write-Host "💡 Please start broker with '.\run.ps1 kafka-start-bg'" -ForegroundColor Yellow
            exit 1
        }
        
        Write-Host "🎯 Streaming real customer events to localhost:9092 (1 event/sec for 5 mins)" -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop streaming" -ForegroundColor Yellow
        
        # Check if we have the required data file
        $dataFiles = @("data\hmQOVnDvRN.xls", "data\customer_data.csv", "data\telco_customer_churn.csv")
        $dataFile = $null
        
        foreach ($file in $dataFiles) {
            if (Test-Path $file) {
                $dataFile = $file
                break
            }
        }
        
        if (-not $dataFile) {
            Write-Host "⚠️ No data file found. Searched for:" -ForegroundColor Yellow
            foreach ($file in $dataFiles) {
                Write-Host "  - $file" -ForegroundColor Gray
            }
            Write-Host "Using producer without specific data file..." -ForegroundColor Yellow
            Invoke-InVenv "python pipelines\kafka_producer.py --mode streaming --rate 1 --duration 300"
        } else {
            Write-Host "📄 Using data file: $dataFile" -ForegroundColor Green
            Invoke-InVenv "python pipelines\producer.py --mode streaming --data-path `"$dataFile`" --events-per-sec 1"
        }
    }

    "kafka-producer-batch" {
        Write-Host "📦 Starting Kafka batch producer..." -ForegroundColor Green
        Write-Host "This will send 100 customer events in batch mode" -ForegroundColor Cyan
        
        # Test Kafka connection
        $kafkaPath = "C:\kafka"
        & "$kafkaPath\bin\windows\kafka-topics.bat" --bootstrap-server localhost:9092 --list 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Cannot connect to Kafka broker" -ForegroundColor Red
            Write-Host "Please start Kafka: .\run.ps1 kafka-start-bg" -ForegroundColor Yellow
            exit 1
        }
        
        Invoke-InVenv "python pipelines\producer.py --mode batch --data-path data\hmQOVnDvRN.xls --batch-size 100"
    }

    "kafka-consumer-stream" {
        Write-Host "🔄 Starting Kafka streaming consumer..." -ForegroundColor Green
        Write-Host "This will continuously process messages and make ML predictions" -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
        
        Invoke-InVenv "python pipelines\consumer.py --mode streaming"
    }

    "kafka-consumer-batch" {
        Write-Host "🤖 Starting Kafka batch consumer..." -ForegroundColor Green
        Write-Host "This will process available messages in batches" -ForegroundColor Cyan
        
        Invoke-InVenv "python pipelines\consumer.py --mode batch --max-records 100"
    }

    "kafka-monitor" {
        Write-Host "📊 Monitoring Kafka prediction results..." -ForegroundColor Green
        Write-Host "This will show real-time predictions from the churn.predictions topic" -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Yellow
        
        $kafkaPath = "C:\kafka"
        & "$kafkaPath\bin\windows\kafka-console-consumer.bat" --bootstrap-server localhost:9092 --topic telco.churn.predictions --from-beginning
    }

    "kafka-demo" {
        Write-Host "🎯 Running complete Kafka demo..." -ForegroundColor Green
        Write-Host "This will demonstrate the full pipeline with sample data" -ForegroundColor Cyan
        
        Invoke-InVenv "python demo_kafka.py"
    }

    "kafka-clean" {
        Write-Host "🧹 Cleaning up Kafka data and topics..." -ForegroundColor Green
        
        $kafkaPath = "C:\kafka"
        
        # Delete topics
        $topics = @("telco.raw.customers", "telco.churn.predictions", "telco.deadletter")
        foreach ($topic in $topics) {
            Write-Host "Deleting topic: $topic" -ForegroundColor Yellow
            & "$kafkaPath\bin\windows\kafka-topics.bat" --bootstrap-server localhost:9092 --delete --topic $topic 2>$null
        }
        
        # Clean up runtime data
        if (Test-Path "runtime\kafka-logs") {
            Remove-Item "runtime\kafka-logs" -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "Cleaned up Kafka logs" -ForegroundColor Green
        }
        
        if (Test-Path "checkpoints") {
            Remove-Item "checkpoints\*" -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "Cleaned up checkpoints" -ForegroundColor Green
        }
        
        Write-Host "✅ Kafka cleanup completed" -ForegroundColor Green
    }

    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run '.\run.ps1 help' to see all available commands" -ForegroundColor Yellow
        exit 1
    }
}