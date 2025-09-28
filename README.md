
Steps To Run 
1. Create Python UV environment
    - uv venv
    - uv --version to check version

2. Activate the environment
    - copy paste the output comes with Activate with:

3. install requirements
    - uv pip install -r requirements.txt

4. Run pipelines seperately
    - python pipelines/data_pipeline.py


Steps to Access MLflow Interface

1. Mlflow is already installed with requirements
2. To run Mlflow ui paste following command into terminal 
    - mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)
    - UI will be available in localhost:MLFLOW_PORT
    - open another terminal and run the pipeline to see the result in UI

