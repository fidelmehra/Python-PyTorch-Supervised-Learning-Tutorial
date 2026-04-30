"""
PyTorch Supervised Learning Tutorial
======================================

A hands-on tutorial for supervised machine learning using Python and PyTorch.
Covers binary classification and regression with full training loops, evaluation, and visualizations.

Requirements:
    pip install torch torchvision scikit-learn matplotlib numpy

Author: Fidel Mehra
Date: April 2026
"""

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import make_moons, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


# ============================================================================
# PART 1: BINARY CLASSIFICATION WITH PYTORCH
# ============================================================================

print("="*70)
print("PART 1: Binary Classification (make_moons dataset)")
print("="*70)

set_seed(42)

# ----- Create toy classification data -----
X_clf, y_clf = make_moons(n_samples=1000, noise=0.2, random_state=42)
scaler_clf = StandardScaler()
X_clf_scaled = scaler_clf.fit_transform(X_clf)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf_scaled, y_clf, test_size=0.2, random_state=42
)

X_train_t_clf = torch.tensor(X_train_clf, dtype=torch.float32)
X_test_t_clf  = torch.tensor(X_test_clf,  dtype=torch.float32)
y_train_t_clf = torch.tensor(y_train_clf, dtype=torch.float32).view(-1, 1)
y_test_t_clf  = torch.tensor(y_test_clf,  dtype=torch.float32).view(-1, 1)

train_ds_clf = TensorDataset(X_train_t_clf, y_train_t_clf)
test_ds_clf  = TensorDataset(X_test_t_clf,  y_test_t_clf)

train_loader_clf = DataLoader(train_ds_clf, batch_size=64, shuffle=True)
test_loader_clf  = DataLoader(test_ds_clf,  batch_size=256, shuffle=False)

print(f"Training samples: {len(X_train_t_clf)}")
print(f"Test samples: {len(X_test_t_clf)}")
print(f"Input features: {X_train_t_clf.shape[1]}")
print()


class BinaryClassifier(nn.Module):
    """Simple feed-forward network for binary classification."""
    
    def __init__(self, in_features=2, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model_clf = BinaryClassifier(in_features=2, hidden_dim=16)
criterion_clf = nn.BCELoss()
optimizer_clf = torch.optim.Adam(model_clf.parameters(), lr=1e-2)

print("Binary Classifier Architecture:")
print(model_clf)
print()


def train_classifier(model, train_loader, test_loader, criterion, optimizer, epochs=50):
    """Train the binary classifier."""
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        # Evaluation
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for xb, yb in test_loader:
                preds = model(xb)
                predicted = (preds >= 0.5).float()
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss = {running_loss/len(train_loader.dataset):.4f} | "
                f"test_acc = {correct/total:.3f}"
            )


print("Training binary classifier...")
train_classifier(model_clf, train_loader_clf, test_loader_clf, 
                criterion_clf, optimizer_clf, epochs=50)
print()


def plot_decision_boundary(model, X, y, scaler, title="Decision Boundary"):
    """Plot decision boundary for binary classification."""
    model.eval()
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        probs = model(grid_t).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, probs.numpy(), levels=50, cmap="RdBu", alpha=0.6)
    plt.colorbar(label="P(class=1)")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k", s=30)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("classification_decision_boundary.png", dpi=150)
    print("Saved: classification_decision_boundary.png")
    # plt.show()


plot_decision_boundary(model_clf, X_clf_scaled, y_clf, scaler_clf, 
                       title="Binary Classification Decision Boundary")
print()


# ============================================================================
# PART 2: REGRESSION WITH PYTORCH
# ============================================================================

print("="*70)
print("PART 2: Regression (make_regression dataset)")
print("="*70)

set_seed(42)

# ----- Create toy regression data -----
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=5,
    noise=15.0,
    random_state=42
)

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

Xr_train_t = torch.tensor(Xr_train, dtype=torch.float32)
Xr_test_t  = torch.tensor(Xr_test,  dtype=torch.float32)
yr_train_t = torch.tensor(yr_train, dtype=torch.float32).view(-1, 1)
yr_test_t  = torch.tensor(yr_test,  dtype=torch.float32).view(-1, 1)

train_ds_reg = TensorDataset(Xr_train_t, yr_train_t)
test_ds_reg  = TensorDataset(Xr_test_t,  yr_test_t)

train_loader_reg = DataLoader(train_ds_reg, batch_size=64, shuffle=True)
test_loader_reg  = DataLoader(test_ds_reg,  batch_size=256, shuffle=False)

print(f"Training samples: {len(Xr_train_t)}")
print(f"Test samples: {len(Xr_test_t)}")
print(f"Input features: {Xr_train_t.shape[1]}")
print()


class Regressor(nn.Module):
    """Feed-forward network for regression."""
    
    def __init__(self, in_features=5, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


model_reg = Regressor(in_features=5, hidden_dim=32)
criterion_reg = nn.MSELoss()
optimizer_reg = torch.optim.Adam(model_reg.parameters(), lr=1e-2)

print("Regressor Architecture:")
print(model_reg)
print()


def train_regressor(model, train_loader, test_loader, criterion, optimizer, epochs=50):
    """Train the regressor."""
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        # Evaluation
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total = 0
            for xb, yb in test_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                total_loss += loss.item() * xb.size(0)
                total += xb.size(0)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"train_mse = {running_loss/len(train_loader.dataset):.2f} | "
                f"test_mse = {total_loss/total:.2f}"
            )


print("Training regressor...")
train_regressor(model_reg, train_loader_reg, test_loader_reg, 
               criterion_reg, optimizer_reg, epochs=50)
print()


def plot_regression_results(model, X_test, y_test, title="Regression: True vs Predicted"):
    """Plot predicted vs true targets for regression."""
    model.eval()
    with torch.no_grad():
        preds_test = model(X_test).squeeze().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test.squeeze().numpy(), preds_test, alpha=0.5, s=30)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title(title)
    
    # Plot perfect prediction line
    y_min = y_test.min().item()
    y_max = y_test.max().item()
    plt.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=2, label="Perfect prediction")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("regression_predictions.png", dpi=150)
    print("Saved: regression_predictions.png")
    # plt.show()


plot_regression_results(model_reg, Xr_test_t, yr_test_t)
print()


# ============================================================================
# SUMMARY
# ============================================================================

print("="*70)
print("SUMMARY OF THE TWO SUPERVISED MODELS")
print("="*70)

summary_table = """
Task                  | Loss Function       | Output Activation | Dataset
----------------------|---------------------|-------------------|-----------------------
Binary Classification | Binary Cross-Entropy| Sigmoid           | make_moons (non-linear)
Regression            | Mean Squared Error  | Linear (none)     | make_regression (synthetic)
"""

print(summary_table)

print("\nKey Takeaways:")
print("  1. Binary classification uses sigmoid output + BCELoss")
print("  2. Regression uses linear output + MSELoss")
print("  3. Both models use Adam optimizer and mini-batch training")
print("  4. DataLoader handles batching and shuffling")
print("  5. Always split data into train/test sets and standardize features")
print()
print("Tutorial complete! Check the saved plots for visualizations.")
print("="*70)
