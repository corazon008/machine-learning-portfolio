import torch
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, model, train_loader, val_loader, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self)->float:
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)

                logits = self.model(x)
                preds = torch.sigmoid(logits) > 0.5

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())

        return accuracy_score(all_labels, all_preds)

