import torch


def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_tot = torch.sum((y_true - y_mean) ** 2, dim=0)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    return torch.mean(1 - ss_res / (ss_tot + 1e-8))


def mae_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    return torch.mean(torch.abs(y_pred - y_true))


def rmse_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))
