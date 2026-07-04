import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, ChebConv
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv


class MarasiGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MarasiGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MarasiGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MarasiChebNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,
        k1: int = 1,
        k2: int = 2,
    ):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=k1)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=k2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MarasiGATv2(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def build_marasi_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int = 2,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "gcn":
        return MarasiGCN(in_channels, hidden_channels, out_channels)

    if model_name == "gat":
        return MarasiGAT(in_channels, hidden_channels, out_channels)

    if model_name in {"graphsage", "sage"}:
        return MarasiGraphSAGE(in_channels, hidden_channels, out_channels)

    if model_name in {"chebnet", "cheb"}:
        return MarasiChebNet(in_channels, hidden_channels, out_channels)

    if model_name == "gatv2":
        return MarasiGATv2(in_channels, hidden_channels, out_channels)

    raise ValueError(f"Unknown model_name: {model_name}")