import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=1,
        bidirectional=False,
        dropout=0.3,
        padding_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)

        logits = self.fc(hidden)

        return logits.squeeze(1)


# pipeline : tokens → indices → Embedding layer → LSTM → pooling → Dense → sigmoid


def pad_sequence(seq, max_len, pad_idx=0):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))
