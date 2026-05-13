"""Per-layer AVGGT K/V budget schedules."""

from __future__ import annotations

import math


SUPPORTED_FACTORS = (1, 2, 4, 6, 9)


def snap_to_supported(values, supported=SUPPORTED_FACTORS):
    """Map arbitrary numeric factors to the nearest supported AVGGT factor."""
    return [min(supported, key=lambda factor: abs(factor - value)) for value in values]


def uniform_budget(n_layers, factor=4):
    return [int(factor)] * int(n_layers)


def linear_pyramid(n_layers, factor_min=2, factor_max=9):
    n_layers = int(n_layers)
    if n_layers <= 1:
        return [int(factor_min)]
    values = [
        factor_min + (factor_max - factor_min) * layer_idx / (n_layers - 1)
        for layer_idx in range(n_layers)
    ]
    return snap_to_supported(values)


def exp_pyramid(n_layers, factor_min=2, factor_max=9, lam=8.0):
    n_layers = int(n_layers)
    values = [
        factor_min + (factor_max - factor_min) * (1.0 - math.exp(-layer_idx / lam))
        for layer_idx in range(n_layers)
    ]
    return snap_to_supported(values)


def make_budget(name, n_layers, base_factor=4, factor_min=2, factor_max=9):
    if name == "uniform":
        return uniform_budget(n_layers, base_factor)
    if name == "linear":
        return linear_pyramid(n_layers, factor_min, factor_max)
    if name == "exp":
        return exp_pyramid(n_layers, factor_min, factor_max)
    raise ValueError(f"Unknown budget function: {name}")
