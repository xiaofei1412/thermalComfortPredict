"""The historical network architecture with deterministic, validated training."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy
import random

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


class LegacyAttentionMLP(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.tensor([0.6, 0.2, 0.2, 0.1]))
        self.linear1 = nn.Linear(4, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        distribution = torch.softmax(inputs * self.attention_weights, dim=1)
        value = torch.relu(self.linear1(inputs * distribution))
        value = torch.relu(self.linear2(value))
        return self.linear3(value).squeeze(-1)


def fit_neural(x_train, y_train, x_validation, y_validation, seed, config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = StandardScaler().fit(x_train)
    inputs = torch.tensor(scaler.transform(x_train), dtype=torch.float32, device=device)
    labels = torch.tensor(np.asarray(y_train), dtype=torch.float32, device=device)
    validation_inputs = torch.tensor(
        scaler.transform(x_validation), dtype=torch.float32, device=device
    )
    validation_labels = torch.tensor(np.asarray(y_validation), dtype=torch.float32, device=device)
    model = LegacyAttentionMLP(config["hidden_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(inputs), device=device)
        total_loss = 0.0
        for indices in order.split(config["batch_size"]):
            optimizer.zero_grad()
            loss = torch.mean((model(inputs[indices]) - labels[indices]) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.mean((model(validation_inputs) - validation_labels) ** 2))
        history.append({
            "epoch": epoch, "train_mse": total_loss / len(inputs),
            "validation_mse": validation_loss,
        })
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f"  neural seed={seed} epoch={epoch} validation_mse={validation_loss:.6f}", flush=True)
        if epoch - best_epoch >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    return {
        "model": model, "scaler": scaler, "history": history,
        "best_epoch": best_epoch, "device": device,
    }


def predict_neural(fitted, inputs):
    inputs = torch.tensor(fitted["scaler"].transform(inputs), dtype=torch.float32)
    with torch.no_grad():
        return fitted["model"](inputs).numpy()


def save_neural(fitted, path, features):
    torch.save({
        "state_dict": fitted["model"].state_dict(),
        "hidden_size": fitted["model"].linear1.out_features,
        "scaler_mean": fitted["scaler"].mean_.tolist(),
        "scaler_scale": fitted["scaler"].scale_.tolist(),
        "features": features,
        "target": "thermal_sensation [-3,3]",
        "best_epoch": fitted["best_epoch"],
    }, path)


def load_neural(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = LegacyAttentionMLP(state["hidden_size"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["scaler_mean"])
    scaler.scale_ = np.asarray(state["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(state["features"])
    return {"model": model, "scaler": scaler, "features": state["features"]}
