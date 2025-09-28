
### Steps To Run 
1. Create Python UV environment
    - uv venv
    - uv --version to check version

2. Activate the environment
    - copy paste the output comes with Activate with:

3. install requirements
    - uv pip install -r requirements.txt

4. Run pipelines seperately
    - python pipelines/data_pipeline.py


### Steps to Access MLflow Interface

1. Mlflow is already installed with requirements

2. To run Mlflow ui paste following command into terminal 
    - mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)
    - UI will be available in localhost:MLFLOW_PORT
    - open another terminal and run the pipeline to see the result in UI

<img width="500" height="500" alt="2025-09-28 (1)" src="https://github.com/user-attachments/assets/c79e0dbe-512a-4fff-b173-2dc8951c92ae" />
<img width="500" height="500" alt="2025-09-28 (3)" src="https://github.com/user-attachments/assets/32edca49-6b98-4158-a256-1c21052ce076" />
