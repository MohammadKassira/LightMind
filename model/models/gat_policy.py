"""GAT backbone for traffic signal control (R4 = 1 layer, R5 = 2 layers, R6 = typed edges, R7 = neighbor masking)."""

import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import GATConv


def _node_valid_from_validity(node_validity: Tensor) -> Tensor:
    """[N, obs_dim] → [N] bool: True if >= 75% of sensors in the node are valid.

    All-fail alternative (sum > 0) is broken: at severity=0.1 with n=17 sensors,
    P(all fail) = 0.1^17 ≈ 1e-17 — masking never fires during training.
    Threshold 0.75 fires when >25% of sensors fail; tests use synthetic inputs
    for full code-path coverage.
    """
    return node_validity.mean(dim=-1) >= 0.75


def _mask_edge_index(
    edge_index: Tensor,
    edge_type: Tensor | None,
    node_valid: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """Remove edges whose source (edge_index[0]) is a failed node.

    Failed nodes are sources of suppression, not sinks: only outgoing edges from
    failed nodes are removed. Incoming edges to a failed node from valid neighbors
    survive — this is the graph-based imputation path.
    """
    if edge_index.shape[1] == 0:
        return edge_index, edge_type
    src  = edge_index[0]
    keep = node_valid[src]
    return edge_index[:, keep], (edge_type[keep] if edge_type is not None else None)


class GATPolicy(nn.Module):
    """One- or two-layer GAT for neighborhood embedding aggregation.

    Args:
        in_channels:     Input embedding dimension (NodeEncoder output, default 64).
        num_heads:       Number of attention heads for both layers (default 4).
        out_per_head:    Output features per head for layer 1 (default 32).
                         Layer 1 output = num_heads * out_per_head = 128.
        num_layers:      1 (R4) or 2 (R5/R6). Default 1 is identical to R4 — same
                         weights, same state dict keys, same out_channels=128.
        l2_out_per_head: Output features per head for layer 2 (default 16).
                         Layer 2 output = num_heads * l2_out_per_head = 64.
                         Ignored when num_layers == 1.
        zero_hop:        If True, skip neighbor messages at every layer — each node
                         only attends to itself.  Use for the 0-hop ablation baseline.
        typed_edges:     If True (R6+), use separate GATConv modules per edge type.
                         gat_flow (type 0, add_self_loops=True) handles upstream flow
                         + self; gat_coord (type 1, add_self_loops=False) handles
                         downstream coordination. Outputs are summed per layer.
                         If False (default), identical to R4/R5 — single GATConv,
                         edge_type argument to forward() is silently ignored.
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_heads: int = 4,
        out_per_head: int = 32,
        num_layers: int = 1,
        l2_out_per_head: int = 16,
        zero_hop: bool = False,
        typed_edges: bool = False,
    ) -> None:
        super().__init__()
        self.zero_hop    = zero_hop
        self.num_layers  = num_layers
        self.typed_edges = typed_edges

        if typed_edges:
            # R6 path: separate weight matrices per edge type.
            # Self-loops are part of the flow stream (gat_flow, add_self_loops=True) —
            # deliberate architectural choice so self-loops appear exactly once.
            # gat_coord (add_self_loops=False) handles downstream coord edges only.
            self.gat_flow  = GATConv(in_channels, out_per_head, heads=num_heads,
                                     concat=True, add_self_loops=True)
            self.gat_coord = GATConv(in_channels, out_per_head, heads=num_heads,
                                     concat=True, add_self_loops=False)
            if num_layers >= 2:
                self.gat2_flow  = GATConv(num_heads * out_per_head, l2_out_per_head,
                                          heads=num_heads, concat=True, add_self_loops=True)
                self.gat2_coord = GATConv(num_heads * out_per_head, l2_out_per_head,
                                          heads=num_heads, concat=True, add_self_loops=False)
        else:
            # R4/R5 path: single GATConv.
            # Named `gat` (not `gat1`) so num_layers=1 state dict is identical to R4 —
            # R4 checkpoints load directly into a num_layers=1 trainer without key remapping.
            # add_self_loops=True (default): GATConv adds missing self-loops automatically.
            # For zero_hop, we pass an empty edge_index; GATConv infers num_nodes from
            # x.size(0) and adds all N self-loops, so each node attends only to itself.
            self.gat = GATConv(in_channels, out_per_head, heads=num_heads, concat=True)
            if num_layers >= 2:
                self.gat2 = GATConv(
                    num_heads * out_per_head, l2_out_per_head, heads=num_heads, concat=True
                )

    @property
    def out_channels(self) -> int:
        if self.num_layers >= 2:
            if self.typed_edges:
                return self.gat2_flow.heads * self.gat2_flow.out_channels   # 4*16 = 64
            return self.gat2.heads * self.gat2.out_channels                  # 4*16 = 64
        if self.typed_edges:
            return self.gat_flow.heads * self.gat_flow.out_channels          # 4*32 = 128
        return self.gat.heads * self.gat.out_channels                        # 4*32 = 128

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor = None,
        node_validity: Tensor = None,
    ) -> Tensor:
        """
        Args:
            x:             [N, in_channels]   per-node embeddings from NodeEncoder
            edge_index:    [2, E]             graph edges (ignored when zero_hop=True)
            edge_type:     [E]                0=flow, 1=coord; required when typed_edges=True,
                                              silently ignored when typed_edges=False
            node_validity: [N, obs_dim]       R7: per-sensor validity from obs_dict;
                                              nodes with mean validity < 0.75 are treated
                                              as failed — their outgoing edges are removed.
                                              None → no masking (identical to R6).
        Returns:
            [N, out_channels]  (128 for 1 layer, 64 for 2 layers)
        """
        if node_validity is not None and not self.zero_hop:
            node_valid            = _node_valid_from_validity(node_validity)
            edge_index, edge_type = _mask_edge_index(edge_index, edge_type, node_valid)

        if not self.typed_edges:
            # R4/R5 path: single GATConv
            ei = edge_index.new_zeros(2, 0) if self.zero_hop else edge_index
            x = self.gat(x, ei)
            if self.num_layers >= 2:
                x = F.elu(x)
                x = self.gat2(x, ei)
            return x

        # R6 typed path: separate flow and coord streams, outputs summed
        empty = edge_index.new_zeros(2, 0)
        if self.zero_hop:
            # gat_flow with empty ei → GATConv adds N self-loops (each node attends to self)
            # gat_coord with empty ei and no self-loops → contributes only bias (x-independent)
            ei_flow  = empty
            ei_coord = empty
        else:
            ei_flow  = edge_index[:, edge_type == 0]
            ei_coord = edge_index[:, edge_type == 1]

        x = self.gat_flow(x, ei_flow) + self.gat_coord(x, ei_coord)
        if self.num_layers >= 2:
            x = F.elu(x)
            x = self.gat2_flow(x, ei_flow) + self.gat2_coord(x, ei_coord)
        return x
