from pathlib import Path

import numpy as np
import torch
from skorch import NeuralNetClassifier

from computer_vision.models.BaseCNN import BaseCNN
from computer_vision.utils.helper import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(config: Config) -> NeuralNetClassifier:
    model = NeuralNetClassifier(module=BaseCNN,
                                criterion=torch.nn.NLLLoss,
                                module__num_classes=5,
                                module__img_size=config.img_size,
                                module__nb_conv_layers=config.nb_conv_layers,
                                module__nb_layers=config.nb_layers,
                                module__net_width=config.net_width,
                                module__dropout_rates=config.dropout_rates, )

    model.initialize()  # This is important!
    model.load_params(f_params=Path(__file__).parent / "model/emotion_cnn_model.pkl")
    return model


def pred_to_name(pred) -> str:
    path = Path(__file__).parent / "model/emotion_class_names.npy"
    idx_to_class = np.load(path, allow_pickle=True).item()
    return idx_to_class[pred]
