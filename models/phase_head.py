import torch
import torch.nn as nn
from torch import Tensor


def pad_phase_features(graph: dict, target_dim: int | None = None) -> tuple[int, list[list[Tensor]]]:
    """Pad all phase feature vectors in the graph to the same length.

    phase_feat_dim (= num_incoming_lanes) varies per intersection. This
    function finds the maximum across all nodes and pads shorter vectors with
    0.0 (= no-green signal, same as red in the §3.4 feature convention).

    If target_dim is given, phase_feat_dim = max(local_max, target_dim) so
    that phase features can be padded to a global ceiling across multiple networks.

    Returns (phase_feat_dim, padded_phase_features) where padded_phase_features
    has the same [node_idx][phase_idx] indexing as graph["phase_features"].
    The original graph dict is not mutated.
    """
    all_feats = graph["phase_features"]
    local_max      = max(feats[0].shape[0] for feats in all_feats if feats)
    phase_feat_dim = max(local_max, target_dim) if target_dim is not None else local_max
    padded = []
    for node_feats in all_feats:
        node_padded = []
        for feat in node_feats:
            n = feat.shape[0]
            if n == phase_feat_dim:
                node_padded.append(feat)
            else:
                pf = torch.zeros(phase_feat_dim, dtype=feat.dtype)
                pf[:n] = feat
                node_padded.append(pf)
        padded.append(node_padded)
    return phase_feat_dim, padded


class PhaseHead(nn.Module):
    """FRAP-style phase-scoring head (§6.2).

    Scores each candidate phase by concatenating the node embedding with the
    phase's feature vector and passing the pair through a shared MLP.  The
    scalar output is the Q-value for that phase in the DQN loop (R2).

    Phases flagged False in valid_transition_mask are set to -inf so argmax
    never selects them.
    """

    def __init__(self, embed_dim: int, phase_feat_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim + phase_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embedding: Tensor,
        phase_features: list[Tensor],
        valid_transition_mask: Tensor,
    ) -> Tensor:
        """Score all candidate phases for one node.

        Args:
            node_embedding:       (embed_dim,) node embedding from NodeEncoder
            phase_features:       list of (phase_feat_dim,) tensors, one per phase
            valid_transition_mask: (num_phases,) bool tensor; False → -inf

        Returns:
            (num_phases,) Q-value scores; invalid phases = -inf
        """
        scores = torch.stack(
            [self.scorer(torch.cat([node_embedding, pf])).squeeze(-1) for pf in phase_features]
        )
        scores = scores.masked_fill(~valid_transition_mask, float("-inf"))
        return scores

    def select_action(
        self,
        node_embedding: Tensor,
        phase_features: list[Tensor],
        valid_transition_mask: Tensor,
    ) -> int:
        """Return the index of the highest-scoring valid phase."""
        with torch.no_grad():
            return int(self.forward(node_embedding, phase_features, valid_transition_mask).argmax().item())

    def forward_batch(
        self,
        node_embeddings: Tensor,
        phase_features: list[Tensor],
        valid_transition_mask: Tensor,
    ) -> Tensor:
        """Score all phases for a batch of B samples for one node.

        Args:
            node_embeddings:       [B, embed_dim]
            phase_features:        list of P tensors, each [phase_feat_dim]
            valid_transition_mask: [P] bool; False → -inf

        Returns:
            [B, P] Q-value scores; invalid phases = -inf
        """
        B = node_embeddings.shape[0]
        P = len(phase_features)
        pf      = torch.stack(phase_features, dim=0)                   # [P, phase_feat_dim]
        emb_exp = node_embeddings.unsqueeze(1).expand(B, P, -1)        # [B, P, embed_dim]
        pf_exp  = pf.unsqueeze(0).expand(B, -1, -1)                    # [B, P, phase_feat_dim]
        pairs   = torch.cat([emb_exp, pf_exp], dim=-1)                 # [B, P, embed_dim+pf_dim]
        scores  = self.scorer(pairs).squeeze(-1)                        # [B, P]
        return scores.masked_fill(~valid_transition_mask.unsqueeze(0), float("-inf"))
