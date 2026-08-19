from __future__ import annotations

import numpy as np


def build_line_graph_adjacency(n_nodes: int, self_loops: bool = True) -> np.ndarray:
    A = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    if self_loops:
        A += np.eye(n_nodes, dtype=float)
    return A


def normalized_adjacency(A: np.ndarray) -> np.ndarray:
    degree = A.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    D_inv = np.diag(inv_sqrt)
    return D_inv @ A @ D_inv


def node_positions(n_nodes: int) -> np.ndarray:
    if n_nodes == 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, n_nodes)
