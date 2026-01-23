from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

from computer_vision.data.transforms import get_transform
from computer_vision.models.BaseCNN import BaseCNN
from computer_vision.utils.helper import load_config, Config
from utils.helper import find_project_root

DATASET_PATH = find_project_root() / Path("datasets/computer_vision/Emotions")
TEST_SIZE = 0.2

if __name__ == '__main__':
    API_FOLDER = Path("../api/")
    config: Config = load_config(API_FOLDER / "config.yaml")

    # Load dataset
    transform = get_transform(config.img_size, config.normalization["mean"], config.normalization["std"])

    ds = ImageFolder(root=DATASET_PATH, transform=transform)

    # Split dataset into train and test sets with stratification
    train_indices, test_indices, _, _ = train_test_split(range(len(ds)),
                                                         ds.targets,
                                                         stratify=ds.targets,
                                                         test_size=TEST_SIZE, )

    print("Slicing dataset into train and test sets...")
    train_dataset = Subset(ds, train_indices)
    test_dataset = Subset(ds, test_indices)

    # Define the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = NeuralNetClassifier(BaseCNN,
                              max_epochs=config.num_epochs,
                              lr=config.learning_rate,
                              batch_size=config.batch_size,
                              optimizer=config.optimizer,
                              device=device,
                              callbacks=[EarlyStopping(patience=5)],
                              module__num_classes=5,
                              module__img_size=config.img_size,
                              module__nb_conv_layers=config.nb_conv_layers,
                              module__nb_layers=config.nb_layers,
                              module__net_width=config.net_width,
                              module__dropout_rates=config.dropout_rates, )

    # Train the model
    y_train = np.array([y for x, y in iter(train_dataset)])
    y_test = np.array([y for x, y in iter(test_dataset)])

    print("Starting training...")
    cnn.fit(train_dataset, y_train)

    # Evaluate the model
    print("Starting evaluation...")
    train_acc = cnn.score(train_dataset, y_train)
    test_acc = cnn.score(test_dataset, y_test)

    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Save the model
    MODEL_PATH = Path(API_FOLDER / Path("model/emotion_cnn_model.pkl"))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cnn.save_params(f_params=MODEL_PATH)

    CLASS_NAME_PATH = Path(API_FOLDER / Path("model/emotion_class_names.npy"))
    CLASS_NAME_PATH.mkdir(parents=True, exist_ok=True)
    class_to_idx = ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    np.save(CLASS_NAME_PATH, idx_to_class)

    print("Training completed and model saved.")
