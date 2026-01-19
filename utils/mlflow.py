import mlflow
import os
from utils.helper import find_project_root

MLFLOW_PID = -1

def start_mlflow_server():
    """
    Start the MLflow server if it is not already running.
    :return:
    """
    # Navigate to root directory and start MLflow server
    global MLFLOW_PID
    if MLFLOW_PID != -1 or is_mlflow_server_running():
        return  # Server already running
    os.chdir(find_project_root())
    mlflow_cmd = "mlflow server --port 5000"
    MLFLOW_PID = os.system(mlflow_cmd)

def is_mlflow_server_running()->bool:
    """
    Check if the MLflow server is running by attempting to connect to it.
    :return:
    bool: True if the server is running, False otherwise.
    """
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.get_experiment_by_name("test_experiment")
        return True
    except Exception:
        return False

def stop_mlflow_server():
    """
    Stop the MLflow server if it is running.
    :return:
    """
    global MLFLOW_PID
    if MLFLOW_PID != -1:
        os.kill(MLFLOW_PID, 9)
        MLFLOW_PID = -1

def set_mlflow_tracking_uri():
    # Set MLflow tracking URI to local server
    mlflow.set_tracking_uri("http://localhost:5000")

if __name__ == "__main__":
    start_mlflow_server()