import os

import mlflow
import requests
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNet
from skorch.helper import SliceDataset
from torch.utils.data.dataset import Dataset

from utils.helper import find_project_root

PORT = 5000
MLFLOW_TRACKING_URI = "http://localhost:{}".format(PORT)


def start_mlflow_server():
    """
    Start the MLflow server if it is not already running.
    :return:
    """
    # Navigate to root directory and start MLflow serve
    if is_mlflow_server_running():
        return  # Server already running
    os.chdir(find_project_root())
    mlflow_cmd = "mlflow server --port {}".format(PORT)
    os.system(mlflow_cmd)


def is_mlflow_server_running() -> bool:
    try:
        r = requests.get(f"{MLFLOW_TRACKING_URI}/", timeout=2)
    except requests.ConnectTimeout:
        return False
    except requests.ConnectionError:
        return False
    if r.status_code != 200:
        return False
    return True


def set_mlflow_tracking_uri():
    # Set MLflow tracking URI to local server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def log_run(model: BaseEstimator | GridSearchCV, X_train: Dataset | SliceDataset, y_train,
            X_test: Dataset | SliceDataset, y_test, params=None, run_name=None) -> NeuralNet:
    """
    Log a training run to MLflow.
    :param model:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param params:
    :return: The trained model
    """
    run_name = run_name if run_name else model.__class__.__name__
    with mlflow.start_run(run_name=run_name):
        if isinstance(params, dict):
            mlflow.log_params(params)
        else:
            mlflow.log_param("params", str(params))

        model.fit(X_train, y_train)
        if isinstance(model, GridSearchCV):
            mlflow.log_param('best_params', model.best_params_)
            mlflow.log_metric('best_cv_accuracy', float(model.best_score_) * 100)

            print("Best parameters found: ", model.best_params_)
            print("Best cross-validation accuracy: ", model.best_score_ * 100)

            model = model.best_estimator_
        else:
            pass

        mlflow.pytorch.log_model(model.module_, name=run_name)

        test_accuracy = model.score(X_test, y_test) * 100
        print("Test set accuracy: ", test_accuracy)

        mlflow.log_metric('test_accuracy', float(test_accuracy))

        # log history metrics
        for epoch, row in enumerate(model.history):
            if 'train_loss' in row:
                mlflow.log_metric('train_loss', float(row['train_loss']), step=epoch)
            if 'valid_loss' in row:
                mlflow.log_metric('val_loss', float(row['valid_loss']), step=epoch)

        return model


if __name__ == "__main__":
    start_mlflow_server()
