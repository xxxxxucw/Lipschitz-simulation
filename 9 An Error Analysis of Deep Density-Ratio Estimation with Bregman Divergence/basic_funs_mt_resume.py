# -*- coding: utf-8 -*-
import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = __import__('time').time()

    def stop(self):
        t = __import__('time').time() - self.tik
        self.times.append(t)
        return t

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


@dataclass
class TrainRecord:
    best_valid_loss: float
    best_train_loss: float
    best_test_loss: float
    best_lipschitz: float
    best_epoch: int
    checkpoint_path: str


class StageEarlyStopping:
    """Stage-specific early stopping that saves unique checkpoints."""

    def __init__(self, checkpoint_path: str, patience: int = 200, delta: float = 0.0, verbose: bool = False):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_train_loss = np.inf
        self.best_test_loss = np.inf
        self.best_lipschitz = 0.0
        self.best_epoch = -1

    def __call__(self, train_loss: float, val_loss: float, test_loss: float, lipschitz: float, epoch: int, model: nn.Module):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(train_loss, val_loss, test_loss, lipschitz, epoch, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(train_loss, val_loss, test_loss, lipschitz, epoch, model)
            self.counter = 0

    def _save_checkpoint(self, train_loss: float, val_loss: float, test_loss: float, lipschitz: float, epoch: int, model: nn.Module):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss
        self.best_train_loss = train_loss
        self.best_test_loss = test_loss
        self.best_lipschitz = lipschitz
        self.best_epoch = epoch
        if self.verbose:
            print(f"Saved checkpoint to {self.checkpoint_path} at epoch={epoch}, val_loss={val_loss:.6f}")


class MultivariateNormalDataset(torch.utils.data.Dataset):
    def __init__(self, N: int, dim: int, rho: float, device: Optional[torch.device] = None):
        self.N = int(N)
        self.rho = float(rho)
        self.dim = int(dim)
        self.device = device
        self.COV = self.cov_matrix
        self.dist = self.build_dist
        self.x, self.logpdf = self.data_sampling

    def __getitem__(self, ix):
        return self.x[ix, :]

    def __len__(self):
        return self.N

    @property
    def cov_matrix(self):
        cov = torch.eye(2 * self.dim)
        for i in range(self.dim):
            cov[2 * i + 1, 2 * i] = self.rho
            cov[2 * i, 2 * i + 1] = self.rho
        return cov

    @property
    def build_dist(self):
        mu = torch.zeros(2 * self.dim)
        return MultivariateNormal(mu, self.COV)

    @property
    def data_sampling(self):
        mu2 = torch.zeros(2 * self.dim)
        cov2 = torch.eye(2 * self.dim)
        dist2 = MultivariateNormal(mu2, cov2)
        sample = self.dist.sample((self.N,))
        logdr_temp = self.dist.log_prob(sample) - dist2.log_prob(sample)
        if self.device is not None:
            sample = sample.to(self.device)
            logdr_temp = logdr_temp.to(self.device)
        return sample, logdr_temp.view(-1, 1)


class bregmanFNN(nn.Module):
    def __init__(self, dim: int, width_vec: Optional[List[int]] = None):
        super().__init__()
        if width_vec is None:
            width_vec = [2 * dim, 16, 8]
        modules: List[nn.Module] = []
        for i in range(len(width_vec) - 1):
            modules.append(nn.Sequential(nn.Linear(width_vec[i], width_vec[i + 1]), nn.ReLU()))
        self.net = nn.Sequential(*modules, nn.Linear(width_vec[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def lr_bregman(D_hat_q: torch.Tensor, D_hat_p: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.log(torch.exp(-D_hat_q) + 1.0)) + torch.mean(torch.log(torch.exp(D_hat_p) + 1.0))


def accuracy_eval(ldr_est: torch.Tensor, ldr_true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((ldr_est.view(-1, 1) - ldr_true.view(-1, 1)) ** 2))


def compute_lipschitz_constant(model: nn.Module) -> float:
    lipschitz = 1.0
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                w = layer.weight.detach()
                s = torch.linalg.svdvals(w)
                lipschitz *= float(s.max().item())
    return float(lipschitz)


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_density_ratio_sum(model_paths: List[str], dataset_x: torch.Tensor, device: torch.device, width_vec: List[int]) -> torch.Tensor:
    model_sum = torch.zeros(dataset_x.shape[0], 1, device=device)
    for path in model_paths:
        net = bregmanFNN(dim=dataset_x.shape[1] // 2, width_vec=width_vec).to(device)
        state = torch.load(path, map_location=device)
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            model_sum += net(dataset_x)
    return model_sum


def train_single_stage(
    net: nn.Module,
    trainer: torch.optim.Optimizer,
    loader_s: torch.utils.data.DataLoader,
    loader_t: torch.utils.data.DataLoader,
    valid_s: torch.Tensor,
    valid_t: torch.Tensor,
    test_s: torch.Tensor,
    test_t: torch.Tensor,
    checkpoint_path: str,
    num_epochs: int,
    patience: int,
    stage_name: str,
    verbose: bool = False,
) -> TrainRecord:
    stopper = StageEarlyStopping(checkpoint_path=checkpoint_path, patience=patience, verbose=verbose)

    for epoch in range(num_epochs):
        net.train()
        train_losses: List[float] = []
        loader_s_iter = iter(loader_s)
        for batch_t in loader_t:
            try:
                batch_s = next(loader_s_iter)
            except StopIteration:
                loader_s_iter = iter(loader_s)
                batch_s = next(loader_s_iter)
            trainer.zero_grad()
            loss = lr_bregman(net(batch_t), net(batch_s))
            loss.backward()
            trainer.step()
            train_losses.append(float(loss.detach().item()))

        net.eval()
        with torch.no_grad():
            train_loss = float(np.mean(train_losses)) if train_losses else np.inf
            valid_loss = float(lr_bregman(net(valid_t), net(valid_s)).detach().item())
            test_loss = float(lr_bregman(net(test_t), net(test_s)).detach().item())
            lipschitz = compute_lipschitz_constant(net)

        stopper(train_loss, valid_loss, test_loss, lipschitz, epoch, net)
        if verbose and (epoch % 100 == 0 or stopper.early_stop):
            print(
                f"[{stage_name}] epoch={epoch} train={train_loss:.6f} valid={valid_loss:.6f} "
                f"test={test_loss:.6f} lip={lipschitz:.6f}"
            )
        if stopper.early_stop:
            break

    return TrainRecord(
        best_valid_loss=float(stopper.val_loss_min),
        best_train_loss=float(stopper.best_train_loss),
        best_test_loss=float(stopper.best_test_loss),
        best_lipschitz=float(stopper.best_lipschitz),
        best_epoch=int(stopper.best_epoch),
        checkpoint_path=checkpoint_path,
    )
