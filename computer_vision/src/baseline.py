from torch import nn
import torch.nn.functional as F

# Baseline model for CNN
class BaselineModel(nn.Module):
    def __init__(
            self,
            input_dim=32*32*3,
            hidden_dim=98,
            output_dim=10,
            dropout=0.5,
    ):
        super(BaselineModel, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = X.reshape(-1, self.hidden.in_features)
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X