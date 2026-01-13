from torch import nn, optim
from utils.BaseCNN import BaseCNN

# Baseline model for CNN
class BaselineModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BaselineModel, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # assuming 3-channel input images (e.g., RGB)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512), # assuming input images are 16x16
            nn.ReLU(),
            nn.Linear(512, num_classes)  # logits
        )

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, preds, targets):
        return self.loss_fn(preds, targets)

    def compute_metrics(self, preds, targets):
        predicted = preds.argmax(1)
        accuracy = (predicted == targets).float().mean().item()
        return {"accuracy": accuracy}