from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.1, hidden_dim=512, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, input):
        return self.net(input)
    
    def predict(self, X: torch.Tensor, threshold: float = 0.5):
        self.eval()
        with torch.no_grad():
            output = torch.sigmoid(self.forward(X))
            predicted_class = (output >= threshold).float()
        return predicted_class
    
    def predict_proba(self, X: torch.Tensor):
        self.eval()
        with torch.no_grad():
            output = torch.sigmoid(self.forward(X)) 
        return output
    
class Trainer(object):
    def __init__(self, model, criterion, optimizer, device, scheduler=None, num_workers=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        # Default LR scheduler (no checkpointing)
        self.scheduler = scheduler or ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
        # Auto-determine number of workers if not specified
        self.num_workers = num_workers if num_workers is not None else min(os.cpu_count(), 4)

    def train(self, X_train:np.ndarray, y_train:np.ndarray, X_val:np.ndarray, y_val:np.ndarray, epochs:int, batch_size:int, lr:float, verbose:bool=True, best_checkpoint_path:str=None, checkpoint_payload:dict=None):
        # Move model to device
        self.model.to(self.device)

        # Convert numpy -> torch and ensure dtypes/shapes
        if isinstance(X_train, np.ndarray): X_train = torch.from_numpy(X_train)
        if isinstance(y_train, np.ndarray): y_train = torch.from_numpy(y_train)
        if isinstance(X_val,   np.ndarray): X_val   = torch.from_numpy(X_val)
        if isinstance(y_val,   np.ndarray): y_val   = torch.from_numpy(y_val)

        X_train = X_train.float()
        X_val   = X_val.float()
        y_train = y_train.float().view(-1, 1)
        y_val   = y_val.float().view(-1, 1)
        # Update optimizer lr if provided
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.patience = max(1, epochs // 10)
            self.scheduler.factor = 0.5

        # Dataloaders with multi-CPU support
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device.startswith('cuda') else False
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.startswith('cuda') else False
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(xb)  # logits shape: (N, 1)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * xb.size(0)

                with torch.no_grad():
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    correct += (preds == (yb >= 0.5)).sum().item()
                    total += xb.size(0)

            train_loss = running_loss / total
            train_acc = correct / total if total > 0 else 0.0

            # Validate
            val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)

            # Scheduler step on validation loss
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            # Track LR
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Store a CPU copy to avoid GPU memory leak
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

                if best_checkpoint_path:
                    checkpoint = {
                        "epoch": epoch,
                        "best_val_loss": float(best_val_loss),
                        "state_dict": best_state,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    }
                    if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
                        checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
                    if checkpoint_payload:
                        checkpoint.update(checkpoint_payload)

                    checkpoint_dir = os.path.dirname(best_checkpoint_path)
                    if checkpoint_dir:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(checkpoint, best_checkpoint_path)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if verbose:
                print(f"Epoch {epoch:03d}/{epochs:03d} | "
                      f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                      f"train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | "
                      f"lr: {self.optimizer.param_groups[0]['lr']:.2e}")

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        history["best_val_loss"] = float(best_val_loss)

        return history

    def calculate_loss(self, X:np.ndarray, y:np.ndarray, batch_size:int=256):
        self.model.eval()
        # Convert numpy -> torch and ensure dtypes/shapes
        if isinstance(X, np.ndarray): X = torch.from_numpy(X)
        if isinstance(y, np.ndarray): y = torch.from_numpy(y)

        X = X.float()
        y = y.float().view(-1, 1)

        loader = DataLoader(
            TensorDataset(X, y), 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.startswith('cuda') else False
        )

        running_loss = 0.0
        total = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                running_loss += loss.item() * xb.size(0)
                total += xb.size(0)

        loss = running_loss / total if total > 0 else float("inf")
        return loss

    def predict_proba(self, X:np.ndarray, batch_size:int=256):
        self.model.eval()
        # Convert numpy -> torch and ensure dtypes/shapes
        if isinstance(X, np.ndarray): X = torch.from_numpy(X)
        X = X.float()

        loader = DataLoader(
            TensorDataset(X), 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.startswith('cuda') else False
        )

        all_preds = []

        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                preds = self.model.predict_proba(xb)
                all_preds.append(preds.cpu().numpy())

        return np.vstack(all_preds).flatten()

    def predict(self, X:np.ndarray, batch_size:int=256, threshold: float = 0.5):
        self.model.eval()
        # Convert numpy -> torch and ensure dtypes/shapes
        if isinstance(X, np.ndarray): X = torch.from_numpy(X)
        X = X.float()

        loader = DataLoader(
            TensorDataset(X), 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.startswith('cuda') else False
        )

        all_preds = []

        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                preds = self.model.predict(xb, threshold=threshold)
                all_preds.append(preds.cpu().numpy())

        return np.vstack(all_preds).flatten()
    
    def evaluate(self, X:np.ndarray, y:np.ndarray, batch_size:int=256):
        self.model.eval()
        # Convert numpy -> torch and ensure dtypes/shapes
        if isinstance(X, np.ndarray): X = torch.from_numpy(X)
        if isinstance(y, np.ndarray): y = torch.from_numpy(y)
        X = X.float()
        y = y.float().view(-1, 1)

        loader = DataLoader(
            TensorDataset(X, y), 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.startswith('cuda') else False
        )

        total = 0
        running_loss = 0.0
        correct = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                running_loss += loss.item() * xb.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == (yb >= 0.5)).sum().item()
                total += xb.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc