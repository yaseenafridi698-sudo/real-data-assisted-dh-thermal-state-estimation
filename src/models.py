from __future__ import annotations

from typing import Any

import torch
from torch import nn


class InterpolationBaseline(nn.Module):
    def __init__(self, n_nodes: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x[..., 0:4]
        masks = x[..., 4:8]
        B, T, N, V = values.shape
        out = torch.zeros_like(values)
        grid = torch.arange(N, device=x.device, dtype=torch.float32)
        for b in range(B):
            for t in range(T):
                for v in range(V):
                    idx = torch.where(masks[b, t, :, v] > 0.5)[0]
                    if idx.numel() == 0:
                        continue
                    if idx.numel() == 1:
                        out[b, t, :, v] = values[b, t, idx[0], v]
                        continue
                    idx_f = idx.to(torch.float32)
                    vals = values[b, t, idx, v]
                    interp = torch.empty(N, device=x.device, dtype=x.dtype)
                    for node in range(N):
                        if node <= idx[0]:
                            interp[node] = vals[0]
                        elif node >= idx[-1]:
                            interp[node] = vals[-1]
                        else:
                            right_pos = torch.searchsorted(idx_f, grid[node])
                            left_pos = right_pos - 1
                            x0 = idx_f[left_pos]
                            x1 = idx_f[right_pos]
                            y0 = vals[left_pos]
                            y1 = vals[right_pos]
                            weight = (grid[node] - x0) / torch.clamp(x1 - x0, min=1.0)
                            interp[node] = y0 * (1 - weight) + y1 * weight
                    out[b, t, :, v] = interp
        return out


class LSTMGlobalBaseline(nn.Module):
    def __init__(self, input_dim: int, n_nodes: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.lstm = nn.LSTM(n_nodes * input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, n_nodes * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape
        h, _ = self.lstm(x.reshape(B, T, N * F))
        y = self.decoder(self.dropout(h)).reshape(B, T, N, 4)
        return y


class GRUGlobalBaseline(nn.Module):
    def __init__(self, input_dim: int, n_nodes: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.gru = nn.GRU(n_nodes * input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, n_nodes * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape
        h, _ = self.gru(x.reshape(B, T, N * F))
        return self.decoder(self.dropout(h)).reshape(B, T, N, 4)


class TransformerGlobalBaseline(nn.Module):
    def __init__(self, input_dim: int, n_nodes: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.input = nn.Linear(n_nodes * input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=max(hidden_dim * 2, 64),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, n_nodes * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape
        h = self.input(x.reshape(B, T, N * F))
        h = self.encoder(h)
        return self.decoder(self.dropout(h)).reshape(B, T, N, 4)


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, a_norm: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("a_norm", a_norm)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("ij,btjf->btif", self.a_norm, x)
        return self.linear(agg)


class ResidualGraphBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, a_norm: torch.Tensor, dropout: float = 0.0) -> None:
        super().__init__()
        self.graph = GraphConvolution(in_dim, hidden_dim, a_norm)
        self.skip = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.graph(x)
        h = self.dropout(self.activation(h))
        return self.norm(h + self.skip(x))


class PureGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, a_norm: torch.Tensor, layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        modules = []
        in_dim = input_dim
        for _ in range(max(1, layers)):
            modules.append(GraphConvolution(in_dim, hidden_dim, a_norm))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(modules)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.dropout(self.activation(layer(h)))
        return self.decoder(h)


class PIGNNDigitalTwin(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        a_norm: torch.Tensor,
        gnn_layers: int = 2,
        gru_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        modules = []
        in_dim = input_dim
        for _ in range(max(1, gnn_layers)):
            modules.append(GraphConvolution(in_dim, hidden_dim, a_norm))
            in_dim = hidden_dim
        self.gnn_layers = nn.ModuleList(modules)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_layers, batch_first=True, dropout=dropout if gru_layers > 1 else 0.0)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.gnn_layers:
            h = self.dropout(self.activation(layer(h)))
        B, T, N, H = h.shape
        node_sequences = h.permute(0, 2, 1, 3).reshape(B * N, T, H)
        temporal_out, _ = self.temporal(node_sequences)
        decoded = self.decoder(temporal_out)
        return decoded.reshape(B, N, T, 4).permute(0, 2, 1, 3)


class ImprovedPIGNNDigitalTwin(nn.Module):
    """Residual graph-temporal estimator with separate physical-state heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        a_norm: torch.Tensor,
        gnn_layers: int = 2,
        gru_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks = []
        in_dim = input_dim
        for _ in range(max(1, gnn_layers)):
            blocks.append(ResidualGraphBlock(in_dim, hidden_dim, a_norm, dropout=dropout))
            in_dim = hidden_dim
        self.graph_blocks = nn.ModuleList(blocks)
        self.temporal = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
                for _ in range(4)
            ]
        )
        self.variance_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4), nn.Softplus())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.graph_blocks:
            h = block(h)
        B, T, N, H = h.shape
        node_sequences = h.permute(0, 2, 1, 3).reshape(B * N, T, H)
        temporal_out, _ = self.temporal(node_sequences)
        temporal_out = self.temporal_norm(temporal_out)
        temporal_out = self.dropout(temporal_out)
        decoded = torch.cat([head(temporal_out) for head in self.heads], dim=-1)
        return decoded.reshape(B, N, T, 4).permute(0, 2, 1, 3)


class ProposedPIGNNGRUV2(nn.Module):
    """Graph-temporal sparse-sensor estimator with gated sensor/topology fusion."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        a_norm: torch.Tensor,
        gnn_layers: int = 2,
        gru_layers: int = 1,
        dropout: float = 0.0,
        pipe_length_m: float = 20000.0,
        heat_loss_u: float = 0.75,
        friction_factor: float = 0.02,
    ) -> None:
        super().__init__()
        self.register_buffer("a_norm", a_norm)
        self.pipe_length_norm = float(max(pipe_length_m, 1.0)) / 20000.0
        self.heat_loss_norm = float(heat_loss_u) / 1.0
        self.friction_norm = float(friction_factor) / 0.02
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.sensor_proj = nn.Sequential(nn.Linear(8, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.static_proj = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        blocks = [ResidualGraphBlock(hidden_dim, hidden_dim, a_norm, dropout=dropout) for _ in range(max(1, gnn_layers))]
        self.graph_blocks = nn.ModuleList(blocks)
        self.post_graph_norm = nn.LayerNorm(hidden_dim)
        self.temporal = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.state_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
                for _ in range(4)
            ]
        )
        self.heat_loss_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.energy_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.aux_outputs: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.feature_proj(x)
        sensor_inputs = x[..., :8] if x.shape[-1] >= 8 else torch.nn.functional.pad(x, (0, max(0, 8 - x.shape[-1])))[..., :8]
        sensor_embedding = self.sensor_proj(sensor_inputs)
        x_norm = x[..., 8:9] if x.shape[-1] > 8 else torch.zeros_like(x[..., :1])
        static = torch.cat(
            [
                x_norm,
                torch.zeros_like(x_norm) + self.pipe_length_norm,
                torch.zeros_like(x_norm) + self.heat_loss_norm,
                torch.zeros_like(x_norm) + self.friction_norm,
            ],
            dim=-1,
        )
        static_embedding = self.static_proj(static)
        gate = self.fusion_gate(torch.cat([base, sensor_embedding, static_embedding], dim=-1))
        h = base + gate * sensor_embedding + (1.0 - gate) * static_embedding
        for block in self.graph_blocks:
            h = block(h)
        h = self.post_graph_norm(h)
        B, T, N, H = h.shape
        node_sequences = h.permute(0, 2, 1, 3).reshape(B * N, T, H)
        temporal_out, _ = self.temporal(node_sequences)
        temporal_out = self.dropout(self.temporal_norm(temporal_out))
        decoded = torch.cat([head(temporal_out) for head in self.state_heads], dim=-1)
        heat_loss = self.heat_loss_head(temporal_out).reshape(B, N, T, 1).permute(0, 2, 1, 3)
        energy = self.energy_head(temporal_out).reshape(B, N, T, 1).permute(0, 2, 1, 3)
        self.aux_outputs = {"heat_loss": heat_loss, "energy_residual": energy}
        return decoded.reshape(B, N, T, 4).permute(0, 2, 1, 3)


def _interpolate_normalized_sensor_state(x: torch.Tensor) -> torch.Tensor:
    values = x[..., 0:4]
    masks = x[..., 4:8]
    N = values.shape[-2]
    pos = torch.linspace(0.0, 1.0, N, device=x.device, dtype=x.dtype)
    distance = torch.abs(pos[:, None] - pos[None, :])
    kernel = 1.0 / (distance + 1.0 / max(float(N), 1.0))
    weights = masks.unsqueeze(-3) * kernel.view(1, 1, N, N, 1)
    numerator = torch.sum(weights * values.unsqueeze(-3), dim=-2)
    denominator = torch.sum(weights, dim=-2)
    return numerator / torch.clamp(denominator, min=1e-6)


class ProposedPIGNNGRUV3(nn.Module):
    """Interpolation-residual graph-temporal estimator for sparse district-heating sensing."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        a_norm: torch.Tensor,
        gnn_layers: int = 2,
        gru_layers: int = 1,
        dropout: float = 0.0,
        pipe_length_m: float = 20000.0,
        heat_loss_u: float = 0.75,
        friction_factor: float = 0.02,
        use_graph: bool = True,
        use_temporal: bool = True,
        use_interpolation_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_graph = use_graph
        self.use_temporal = use_temporal
        self.use_interpolation_residual = use_interpolation_residual
        self.pipe_length_norm = float(max(pipe_length_m, 1.0)) / 20000.0
        self.heat_loss_norm = float(heat_loss_u) / 1.0
        self.friction_norm = float(friction_factor) / 0.02
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.sensor_proj = nn.Sequential(nn.Linear(8, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.edge_position_proj = nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.graph_blocks = nn.ModuleList(
            [ResidualGraphBlock(hidden_dim, hidden_dim, a_norm, dropout=dropout) for _ in range(max(1, gnn_layers))]
        )
        self.no_graph_blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
                for _ in range(max(1, gnn_layers))
            ]
        )
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.temporal = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.state_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
                for _ in range(4)
            ]
        )
        self.heat_loss_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.boundary_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        self.correction_scale = nn.Parameter(torch.tensor(0.25))
        self.aux_outputs: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.input_proj(x)
        sensor_inputs = x[..., :8] if x.shape[-1] >= 8 else torch.nn.functional.pad(x, (0, max(0, 8 - x.shape[-1])))[..., :8]
        sensor_embedding = self.sensor_proj(sensor_inputs)
        x_norm = x[..., 8:9] if x.shape[-1] > 8 else torch.zeros_like(x[..., :1])
        sensor_density = x[..., 4:8].mean(dim=-1, keepdim=True) if x.shape[-1] >= 8 else torch.zeros_like(x_norm)
        segment_norm = torch.zeros_like(x_norm) + self.pipe_length_norm / max(x.shape[-2] - 1, 1)
        edge_position = torch.cat(
            [
                x_norm,
                segment_norm,
                torch.zeros_like(x_norm) + self.heat_loss_norm,
                torch.zeros_like(x_norm) + self.friction_norm,
                sensor_density,
                torch.abs(x_norm - 0.5),
            ],
            dim=-1,
        )
        edge_embedding = self.edge_position_proj(edge_position)
        gate = self.fusion_gate(torch.cat([base, sensor_embedding, edge_embedding], dim=-1))
        h = base + gate * sensor_embedding + (1.0 - gate) * edge_embedding
        if self.use_graph:
            for block in self.graph_blocks:
                h = block(h)
        else:
            for block in self.no_graph_blocks:
                h = block(h) + h
        h = self.graph_norm(h)
        B, T, N, H = h.shape
        if self.use_temporal:
            node_sequences = h.permute(0, 2, 1, 3).reshape(B * N, T, H)
            temporal_out, _ = self.temporal(node_sequences)
            temporal_out = self.dropout(self.temporal_norm(temporal_out))
        else:
            temporal_out = self.dropout(self.temporal_norm(h.permute(0, 2, 1, 3).reshape(B * N, T, H)))
        correction = torch.cat([head(temporal_out) for head in self.state_heads], dim=-1).reshape(B, N, T, 4).permute(0, 2, 1, 3)
        if self.use_interpolation_residual:
            interp = _interpolate_normalized_sensor_state(x)
            decoded = interp + torch.tanh(self.correction_scale) * correction
        else:
            decoded = correction
        heat_loss = self.heat_loss_head(temporal_out).reshape(B, N, T, 1).permute(0, 2, 1, 3)
        boundary = self.boundary_head(temporal_out).reshape(B, N, T, 2).permute(0, 2, 1, 3)
        self.aux_outputs = {"heat_loss": heat_loss, "boundary": boundary}
        return decoded


def build_model(model_name: str, input_dim: int, n_nodes: int, a_norm: torch.Tensor, config: dict[str, Any]) -> nn.Module:
    hidden = int(config["model"]["hidden_dim"])
    dropout = float(config["model"]["dropout"])
    if model_name == "lstm":
        return LSTMGlobalBaseline(input_dim, n_nodes, hidden, dropout)
    if model_name == "pi_lstm":
        return LSTMGlobalBaseline(input_dim, n_nodes, hidden, dropout)
    if model_name == "gru":
        return GRUGlobalBaseline(input_dim, n_nodes, hidden, dropout)
    if model_name == "transformer":
        return TransformerGlobalBaseline(input_dim, n_nodes, hidden, dropout)
    if model_name == "pure_gnn":
        return PureGNN(input_dim, hidden, a_norm, layers=int(config["model"]["gnn_layers"]), dropout=dropout)
    if model_name == "pignn_no_temporal":
        return PureGNN(input_dim, hidden, a_norm, layers=int(config["model"]["gnn_layers"]), dropout=dropout)
    if model_name == "pignn":
        return PIGNNDigitalTwin(
            input_dim,
            hidden,
            a_norm,
            gnn_layers=int(config["model"]["gnn_layers"]),
            gru_layers=int(config["model"]["gru_layers"]),
            dropout=dropout,
        )
    if model_name == "pignn_improved":
        return ImprovedPIGNNDigitalTwin(
            input_dim,
            hidden,
            a_norm,
            gnn_layers=int(config["model"]["gnn_layers"]),
            gru_layers=int(config["model"]["gru_layers"]),
            dropout=dropout,
        )
    if model_name == "pignn_v2":
        sys = config.get("system", {})
        return ProposedPIGNNGRUV2(
            input_dim,
            hidden,
            a_norm,
            gnn_layers=int(config["model"]["gnn_layers"]),
            gru_layers=int(config["model"]["gru_layers"]),
            dropout=dropout,
            pipe_length_m=float(sys.get("length_m", 20000.0)),
            heat_loss_u=float(sys.get("heat_loss_U_W_m2K", 0.75)),
            friction_factor=float(sys.get("friction_factor", 0.02)),
        )
    if model_name in {"pignn_v3", "pignn_v3_no_graph", "pignn_v3_no_temporal", "pignn_v3_no_interp"}:
        sys = config.get("system", {})
        return ProposedPIGNNGRUV3(
            input_dim,
            hidden,
            a_norm,
            gnn_layers=int(config["model"]["gnn_layers"]),
            gru_layers=int(config["model"].get("gru_layers", 1)),
            dropout=dropout,
            pipe_length_m=float(sys.get("length_m", 20000.0)),
            heat_loss_u=float(sys.get("heat_loss_U_W_m2K", 0.75)),
            friction_factor=float(sys.get("friction_factor", 0.02)),
            use_graph=model_name != "pignn_v3_no_graph",
            use_temporal=model_name != "pignn_v3_no_temporal",
            use_interpolation_residual=model_name != "pignn_v3_no_interp",
        )
    if model_name == "interpolation":
        return InterpolationBaseline(n_nodes)
    raise KeyError(f"Unknown model: {model_name}")
