import torch


def l0_distance(x1: torch.tensor, x2: torch.tensor):
    return (x1 != x2).sum().item()


def l1_distance(x1: torch.tensor, x2: torch.tensor):
    return torch.abs(x1 - x2).sum().item()


def l2_distance(x1: torch.tensor, x2: torch.tensor):
    return torch.sqrt(((x1 - x2) ** 2).sum()).item()


def linf_distance(x1: torch.tensor, x2: torch.tensor):
    abs_x = torch.abs(x1 - x2)
    return abs_x.max().item()


def l1_norm(x: torch.tensor):
    return torch.abs(x).sum()


def l2_norm(x: torch.tensor):
    return torch.square(x).sum()


def linf_norm(x: torch.tensor):
    return torch.abs(x).max()
