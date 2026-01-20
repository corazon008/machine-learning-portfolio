import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional

from utils.helper import MyDropout


class BaseCNN(nn.Module):
    def __init__(self, num_classes: int = 10,
                 img_size: int = 32,
                 nb_conv_layers: int = 3,
                 nb_layers: int = 2,
                 net_width: int = 256,
                 dropout_rates: Optional[List[float]] = None,
                 activation=nn.ReLU):

        super().__init__()
        dropout = MyDropout(dropout_rates)
        self.net = nn.Sequential()

        # Convolutional layers
        in_channels = 3  # Assuming RGB images
        # Cap maximum number of channels to avoid excessive memory use
        max_channels = 512
        for i in range(nb_conv_layers):
            # progressively increase number of channels but cap it
            out_channels = min(max_channels, 32 * (2 ** i))
            self.net.add_module(f"conv_layer_{i + 1}", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.net.add_module(f"relu_conv_{i + 1}", activation())
            self.net.add_module(f"maxpool_{i + 1}", nn.MaxPool2d(2, 2))  # Downsample by a factor of 2
            in_channels = out_channels

        self.net.add_module("flatten", nn.Flatten())

        # Calculate the size of the feature map after convolutional layers
        # Ensure we don't divide down to zero
        conv_output_size = img_size // (2 ** nb_conv_layers)

        if conv_output_size < 1:
            raise ValueError(
                f"Too many pooling operations for given image size: "
                f"img_size={img_size}, nb_conv_layers={nb_conv_layers}. "
                "Reduce `nb_conv_layers` or increase image size."
            )

        nb_features = in_channels * img_size * img_size

        # Fully connected layers
        self.net.add_module("input_layer", nn.Linear(nb_features, net_width))
        for i in range(nb_layers):
            self.net.add_module(f"hidden_layer_{i + 1}", nn.Linear(net_width, net_width))
            self.net.add_module(f"relu_{i + 1}", activation())
            d = dropout.get()
            if d > 0:  # only add dropout if rate > 0
                self.net.add_module(f"dropout_{i + 1}", nn.Dropout(d))

        self.net.add_module("output_layer", nn.Linear(net_width, num_classes))

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)
