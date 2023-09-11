import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch import Tensor
import numpy as np


import torch

class CosineSimilarity:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.func = torch.nn.CosineSimilarity(**kwargs)

    def __call__(self, x, y):
        return (self.func(x.squeeze(), y.squeeze()) + 1) / 2


class Loss:
    def __init__(self, model, loss_fns, weights=None, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.loss_fns = loss_fns
        if weights is not None:
            self.loss_weights = weights
        else:
            self.loss_weights = [1] * len(loss_fns)

    def loss_gradient(self, x, y):
        x_grad = x.clone().detach().requires_grad_(True)
        y_c = y.clone().detach()
        pred = self.model.encode_batch(x_grad)  # TODO: remove hard-coded encode_batch

        loss = torch.zeros(1, device=x_grad.device)
        for loss_fn, loss_weight in zip(self.loss_fns, self.loss_weights):
            loss += loss_weight * loss_fn(pred, y_c).squeeze().mean()

        self.model.zero_grad()
        loss.backward()
        grads = x_grad.grad
        return grads, loss.item()



