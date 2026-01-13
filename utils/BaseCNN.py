import torch
from torch import nn
from typing import List, Optional

from utils.BaseModel import MyDropout


class BaseCNN(nn.Module):
    def __init__(self, num_classes: int = 1,
                 img_height: int = 32,
                 img_width: int = 32,
                 nb_conv_layers: int = 2,
                 nb_layers: int = 2,
                 net_width: int = 512,
                 dropout_rates: Optional[List[float]] = None,
                 loss_fn=nn.CrossEntropyLoss,
                 activation=nn.ReLU):

        super().__init__()
        dropout = MyDropout(dropout_rates)
        self.net = nn.Sequential()

        # Convolutional layers
        in_channels = 3  # Assuming RGB images
        for i in range(nb_conv_layers):
            out_channels = 32 * (2 ** i)  # Double the number of channels with each layer
            self.net.add_module(f"conv_layer_{i + 1}", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.net.add_module(f"relu_conv_{i + 1}", activation())
            self.net.add_module(f"maxpool_{i + 1}", nn.MaxPool2d(2, 2))  # Downsample by a factor of 2
            in_channels = out_channels

        self.net.add_module("flatten", nn.Flatten())

        # Calculate the size of the feature map after convolutional layers
        conv_output_height = img_height // (2 ** nb_conv_layers)
        conv_output_width = img_width // (2 ** nb_conv_layers)
        nb_features = in_channels * conv_output_height * conv_output_width

        # Fully connected layers
        self.net.add_module("input_layer", nn.Linear(nb_features, net_width))
        for i in range(nb_layers):
            self.net.add_module(f"hidden_layer_{i + 1}", nn.Linear(net_width, net_width))
            self.net.add_module(f"relu_{i + 1}", activation())
            d = dropout.get()
            if d > 0: # Clean the network, only add dropout if rate > 0
                self.net.add_module(f"dropout_{i + 1}", nn.Dropout(d))

        self.net.add_module("output_layer", nn.Linear(net_width, num_classes))
        # Accept either a loss class (callable) or an already-instantiated loss object.
        if callable(loss_fn):
            # instantiate
            try:
                self.loss_fn = loss_fn()
            except Exception:
                # fallback: if instantiation fails, keep the object as-is
                self.loss_fn = loss_fn
        else:
            self.loss_fn = loss_fn

    def forward(self, x):
        return self.net(x)  # .squeeze(1)

    def compute_loss(self, preds, targets):
        return self.loss_fn(preds, targets)

    def compute_metrics(self, preds, targets):
        predicted = preds.argmax(1)
        accuracy = (predicted == targets).float().mean().item()
        return {"accuracy": accuracy}
