# load_model.py
import torch
from computer_vision.src.BaseCNN import BaseCNN
from skorch import NeuralNetClassifier
import numpy as np

from pathlib import Path

from utils.helper import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(config: Config) -> NeuralNetClassifier:
    model = NeuralNetClassifier(
        module=BaseCNN,
        criterion=torch.nn.NLLLoss,
        module__num_classes=5,
        module__img_height=config.img_size,
        module__img_width=config.img_size,
        module__nb_conv_layers=config.nb_conv_layers,
        module__nb_layers=config.nb_layers,
        module__net_width=config.net_width,
        module__dropout_rates=config.dropout_rates,
    )
    model.initialize()  # This is important!
    model.load_params(f_params=Path("models/emotion_cnn_model.pkl"))
    return model

def pred_to_name(pred) -> str:
    path = Path("models/emotion_class_names.npy")
    idx_to_class = np.load(path, allow_pickle=True).item()
    return idx_to_class[pred]