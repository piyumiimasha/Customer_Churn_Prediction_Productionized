@echo off
title Simple Kafka Demo - Telco Customer Churn
color 0A

echo.
echo =========================================================
echo         SIMPLE KAFKA DEMO - TELCO CHURN
echo =========================================================
echo.
echo Based on Makefile commands - clean and simple!
echo.

:menu
cls
echo.
echo =========================================================
echo                    KAFKA COMMANDS
echo =========================================================
echo.
echo 1. Format Kafka (first time setup)
echo 2. Start Kafka Broker
echo 3. Create Topics  
echo 4. Check Status
echo 5. Run Producer
echo 6. Run Consumer
echo 7. Stop Kafka
echo 8. Clean Reset (if needed)
echo 9. Exit
echo.
set /p choice="Enter choice (1-9): "

if "%choice%"=="1" goto kafka_format
if "%choice%"=="2" goto kafka_start
if "%choice%"=="3" goto kafka_topics
if "%choice%"=="4" goto kafka_check
if "%choice%"=="5" goto kafka_producer
if "%choice%"=="6" goto kafka_consumer
if "%choice%"=="7" goto kafka_stop
if "%choice%"=="8" goto kafka_reset
if "%choice%"=="9" goto exit_script
goto menu

:kafka_format
echo.
echo 🔧 Formatting Kafka storage (KRaft mode)...
cd /d C:\kafka
echo Checking for existing cluster ID...
set CLUSTER_ID=
if exist "/tmp/kraft-combined-logs/meta.properties" (
    echo Found existing storage - using existing cluster ID
    set CLUSTER_ID=ghVN2oxvTEyhUO8w40ii6g
) else (
    echo No existing storage - generating new cluster ID...
    for /f %%i in ('bin\windows\kafka-storage.bat random-uuid 2^>nul') do set CLUSTER_ID=%%i
)
echo Using Cluster ID: %CLUSTER_ID%
bin\windows\kafka-storage.bat format -t %CLUSTER_ID% -c config\server.properties --standalone --ignore-formatted
if errorlevel 1 (
    echo ℹ️ Storage already formatted - ready to use!
) else (
    echo ✅ Formatted successfully
)
pause
goto menu

:kafka_start
echo.
echo 🚀 Starting Kafka broker...
cd /d C:\kafka
start "Kafka" cmd /k "bin\windows\kafka-server-start.bat config\server.properties"
echo ✅ Started in new window
pause
goto menu

:kafka_topics
echo.
echo 📋 Creating topics...
cd /d C:\kafka
bin\windows\kafka-topics.bat --bootstrap-server localhost:9092 --create --topic telco.raw.customers --partitions 1 --replication-factor 1 --if-not-exists
bin\windows\kafka-topics.bat --bootstrap-server localhost:9092 --create --topic telco.churn.predictions --partitions 1 --replication-factor 1 --if-not-exists
echo ✅ Topics created
bin\windows\kafka-topics.bat --bootstrap-server localhost:9092 --list
pause
goto menu

:kafka_check
echo.
echo 🔍 Checking status...
cd /d C:\kafka
bin\windows\kafka-topics.bat --bootstrap-server localhost:9092 --list
pause
goto menu

:kafka_producer
echo.
echo 🌊 Starting producer...
cd /d C:\Users\viraj\Zuu\Customer_Churn_Prediction_Productionized
call .venv\Scripts\activate.bat
python pipelines\producer.py --mode streaming --events-per-sec 2
pause
goto menu

:kafka_consumer
echo.
echo 🔄 Starting consumer...
cd /d C:\Users\viraj\Zuu\Customer_Churn_Prediction_Productionized
call .venv\Scripts\activate.bat
python pipelines\consumer.py --mode streaming
pause
goto menu

:kafka_stop
echo.
echo 🛑 Stopping Kafka...
taskkill /f /im java.exe >nul 2>&1
echo ✅ Stopped
pause
goto menu

:kafka_reset
echo.
echo 🧹 Clean Reset - This will delete ALL Kafka data!
set /p confirm="Are you sure? (y/N): "
if /i not "%confirm%"=="y" goto menu
echo Stopping Kafka...
taskkill /f /im java.exe >nul 2>&1
timeout /t 2 >nul
echo Cleaning storage...
cd /d C:\kafka
rmdir /s /q logs 2>nul
rmdir /s /q /tmp 2>nul
if exist C:\tmp rmdir /s /q C:\tmp 2>nul
echo ✅ Clean reset completed
echo 💡 Run option 1 to format fresh storage
pause
goto menu

:exit_script
echo.
echo Thanks for using Simple Kafka Demo!
pause
exit /b 0