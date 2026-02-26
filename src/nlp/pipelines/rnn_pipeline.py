import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset


class RNNPipeline:

    def __init__(self, model, encoder, device="cpu"):
        self.model = model.to(device)
        self.encoder = encoder
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )

    # ---------------------
    # TRAIN
    # ---------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=64, epochs=5):

        X_tensor = self.encoder.batch_encode(X_train)
        y_tensor = torch.tensor(y_train, dtype=torch.float)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):

            self.model.train()
            total_loss = 0

            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                total_loss += loss.item()

            if X_val is not None and y_val is not None:
                val_acc = self.score(X_val, y_val)
                print(f"Epoch {epoch} | Loss {total_loss/len(loader):.4f} | Train Acc {self.score(X_train, y_train):.4f} | Val Acc {val_acc:.4f}")
            else:
                print(f"Epoch {epoch} | Loss {total_loss/len(loader):.4f} | Acc {self.score(X_train, y_train):.4f}")

    # ---------------------
    # PREDICT PROBA
    # ---------------------

    def predict_proba(self, texts, batch_size=64):

        self.model.eval()

        X_tensor = self.encoder.batch_encode(texts)
        loader = DataLoader(X_tensor, batch_size=batch_size)

        probs = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                p = torch.sigmoid(logits)
                probs.extend(p.cpu().numpy())

        return np.array(probs)

    # ---------------------
    # PREDICT
    # ---------------------

    def predict(self, texts, threshold=0.5):
        probs = self.predict_proba(texts)
        return (probs > threshold).astype(int)

    # ---------------------
    # SCORE
    # ---------------------

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)