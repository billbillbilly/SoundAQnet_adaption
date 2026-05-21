"""
ResGatedGCN: Residual Gated Graph ConvNets — pure-PyTorch implementation.
An Experimental Study of Neural Networks for Variable Graphs (Bresson & Laurent, ICLR 2018)
https://arxiv.org/pdf/1711.07553v2.pdf

This version removes the DGL dependency entirely.  SoundAQnet always uses a
fixed fully-connected graph with self-loops (8 nodes → 8² = 64 directed edges).
Because the topology never changes, edge indices are precomputed as module
buffers and message-passing is done with torch.scatter_add_, which is built into
PyTorch and produces mathematically identical outputs to the original DGL code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedGCNLayer(nn.Module):
    """Gated GCN layer — pure PyTorch, no DGL required.

    Args:
        input_dim:  Node/edge input feature dimension.
        output_dim: Node/edge output feature dimension.
        dropout:    Dropout probability applied after activation.
        batch_norm: If True, apply BatchNorm1d after the linear transforms.
        residual:   Skip connection (only active when input_dim == output_dim).
        num_nodes:  Nodes in the fixed graph (default 8).  Self-loops are
                    included so there are num_nodes² directed edges.
    """

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, num_nodes=8):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual if input_dim == output_dim else False
        self.num_nodes = num_nodes

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

        # Pre-compute the fixed N×N edge index (with self-loops).
        # Registered as buffers so they are automatically moved by .to(device).
        N = num_nodes
        src = torch.tensor([i for i in range(N) for j in range(N)], dtype=torch.long)
        dst = torch.tensor([j for i in range(N) for j in range(N)], dtype=torch.long)
        # persistent=False: buffers move with .to(device) but are NOT saved in
        # state_dict, so pre-trained .pth files (which lack these keys) load
        # cleanly with the default strict=True.
        self.register_buffer("_src_tmpl", src, persistent=False)  # [N²]
        self.register_buffer("_dst_tmpl", dst, persistent=False)  # [N²]

    def forward(self, h, e):
        """
        Args:
            h: Node features  ``[B*N, in_dim]``  (B graphs stacked).
            e: Edge features  ``[B*N², in_dim]``.
        Returns:
            Tuple ``(h, e)`` with shapes ``[B*N, out_dim]`` and ``[B*N², out_dim]``.
        """
        h_in = h
        e_in = e

        N = self.num_nodes
        E = N * N  # edges per graph
        B = h.shape[0] // N  # batch size

        # ── Batched edge indices ──────────────────────────────────────────────
        offsets = torch.arange(B, device=h.device).repeat_interleave(E) * N  # [B*E]
        src_idx = self._src_tmpl.repeat(B) + offsets  # [B*E]
        dst_idx = self._dst_tmpl.repeat(B) + offsets  # [B*E]

        # ── Linear projections ────────────────────────────────────────────────
        Ah = self.A(h)  # [B*N, out]
        Bh = self.B(h)  # [B*N, out]
        Dh = self.D(h)  # [B*N, out]
        Eh = self.E(h)  # [B*N, out]
        Ce = self.C(e)  # [B*E, out]

        # ── Edge update: e_ij = D(h_i) + E(h_j) + C(e_ij) ───────────────────
        # Matches DGL: apply_edges(fn.u_add_v('Dh','Eh','DEh')); edata['e']=DEh+Ce
        e_new = Dh[src_idx] + Eh[dst_idx] + Ce  # [B*E, out]
        sigma = torch.sigmoid(e_new)  # gate

        # ── Node aggregation ─────────────────────────────────────────────────
        # h_i = A(h_i) + Σ_j σ_ij · B(h_j)  /  (Σ_j σ_ij + ε)
        # Matches DGL: update_all(fn.u_mul_e('Bh','sigma','m'), fn.sum('m',...))
        msg = Bh[src_idx] * sigma  # [B*E, out]
        dst_exp = dst_idx.unsqueeze(1).expand_as(msg)  # [B*E, out]
        sum_sh = torch.zeros_like(Ah)
        sum_s = torch.zeros_like(Ah)
        sum_sh.scatter_add_(0, dst_exp, msg)
        sum_s.scatter_add_(0, dst_exp, sigma)
        h_new = Ah + sum_sh / (sum_s + 1e-6)

        # ── Post-processing ───────────────────────────────────────────────────
        if self.batch_norm:
            h_new = self.bn_node_h(h_new)
            e_new = self.bn_node_e(e_new)

        h_new = F.relu(h_new)
        e_new = F.relu(e_new)

        if self.residual:
            h_new = h_in + h_new
            e_new = e_in + e_new

        h_new = F.dropout(h_new, self.dropout, training=self.training)
        e_new = F.dropout(e_new, self.dropout, training=self.training)

        return h_new, e_new

    def __repr__(self):
        return "{}(in={}, out={}, nodes={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_nodes
        )


##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################


class GatedGCNLayerEdgeFeatOnly(nn.Module):
    """Variant that updates node features only (no edge feature output).

    Same pure-PyTorch approach as GatedGCNLayer; edge features are passed
    through unchanged.
    """

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, num_nodes=8):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual if input_dim == output_dim else False
        self.num_nodes = num_nodes

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

        N = num_nodes
        src = torch.tensor([i for i in range(N) for j in range(N)], dtype=torch.long)
        dst = torch.tensor([j for i in range(N) for j in range(N)], dtype=torch.long)
        self.register_buffer("_src_tmpl", src, persistent=False)
        self.register_buffer("_dst_tmpl", dst, persistent=False)

    def forward(self, h, e):
        h_in = h
        N = self.num_nodes
        E = N * N
        B = h.shape[0] // N
        offsets = torch.arange(B, device=h.device).repeat_interleave(E) * N
        src_idx = self._src_tmpl.repeat(B) + offsets
        dst_idx = self._dst_tmpl.repeat(B) + offsets

        Ah = self.A(h)
        Bh = self.B(h)
        Dh = self.D(h)
        Eh = self.E(h)

        e_gate = Dh[src_idx] + Eh[dst_idx]
        sigma = torch.sigmoid(e_gate)

        msg = Bh[src_idx] * sigma
        dst_exp = dst_idx.unsqueeze(1).expand_as(msg)
        sum_sh = torch.zeros_like(Ah)
        sum_s = torch.zeros_like(Ah)
        sum_sh.scatter_add_(0, dst_exp, msg)
        sum_s.scatter_add_(0, dst_exp, sigma)
        h_new = Ah + sum_sh / (sum_s + 1e-6)

        if self.batch_norm:
            h_new = self.bn_node_h(h_new)
        h_new = F.relu(h_new)
        if self.residual:
            h_new = h_in + h_new
        h_new = F.dropout(h_new, self.dropout, training=self.training)
        return h_new, e

    def __repr__(self):
        return "{}(in={}, out={}, nodes={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_nodes
        )


##############################################################


class GatedGCNLayerIsotropic(nn.Module):
    """Isotropic variant — no edge gating, simple neighbourhood sum."""

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, num_nodes=8):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual if input_dim == output_dim else False
        self.num_nodes = num_nodes

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

        N = num_nodes
        src = torch.tensor([i for i in range(N) for j in range(N)], dtype=torch.long)
        dst = torch.tensor([j for i in range(N) for j in range(N)], dtype=torch.long)
        self.register_buffer("_src_tmpl", src, persistent=False)
        self.register_buffer("_dst_tmpl", dst, persistent=False)

    def forward(self, h, e):
        h_in = h
        N = self.num_nodes
        E = N * N
        B = h.shape[0] // N
        offsets = torch.arange(B, device=h.device).repeat_interleave(E) * N
        src_idx = self._src_tmpl.repeat(B) + offsets
        dst_idx = self._dst_tmpl.repeat(B) + offsets

        Ah = self.A(h)
        Bh = self.B(h)
        msg = Bh[src_idx]
        dst_exp = dst_idx.unsqueeze(1).expand_as(msg)
        sum_h = torch.zeros_like(Ah)
        sum_h.scatter_add_(0, dst_exp, msg)
        h_new = Ah + sum_h

        if self.batch_norm:
            h_new = self.bn_node_h(h_new)
        h_new = F.relu(h_new)
        if self.residual:
            h_new = h_in + h_new
        h_new = F.dropout(h_new, self.dropout, training=self.training)
        return h_new, e

    def __repr__(self):
        return "{}(in={}, out={}, nodes={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_nodes
        )
