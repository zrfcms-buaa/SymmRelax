import math
import warnings
import torch
from torch import nn
from torch_scatter import scatter
import torch.nn.functional as F

from symm_graph import (
    ScaledSiLU,
    AtomEmbedding,
    RadialBasis,
    cell_offsets_to_num,
    sinusoidal_positional_encoding,
    vector_norm,
)



# ============================================================
# Helpers
# ============================================================

def _get_atomic_numbers(data: object) -> torch.Tensor:
    for k in ("atomic_numbers", "z"):
        if hasattr(data, k) and getattr(data, k) is not None:
            z = getattr(data, k).view(-1)
            if torch.is_floating_point(z):
                z = torch.round(z)
            return z.long()

    if hasattr(data, "x") and data.x is not None:
        x = data.x
        z = x if x.dim() == 1 else x[:, 0]
        if torch.is_floating_point(z):
            z = torch.round(z)
        return z.long()

    raise AttributeError("Cannot find atomic numbers: tried atomic_numbers/z/x")


def _wrap01(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)


def gaussian_nll_1d(
    mu: torch.Tensor,
    raw_var: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    var = F.softplus(raw_var) + eps
    return 0.5 * (((target - mu) ** 2) / var + torch.log(var)).mean()


def build_edge_distances_pbc(
    pos: torch.Tensor,
    cell: torch.Tensor,
    cell_offsets: torch.Tensor,
    edge_index: torch.Tensor,
    neighbors: torch.Tensor,
) -> torch.Tensor:
    j, i = edge_index
    cell_offsets_unsqueeze = cell_offsets.unsqueeze(1).to(dtype=cell.dtype)
    abc_unsqueeze = cell.repeat_interleave(neighbors, dim=0)
    vecs = (pos[j] + (cell_offsets_unsqueeze @ abc_unsqueeze).squeeze(1)) - pos[i]
    return vector_norm(vecs, dim=-1)


def build_cell_from_params(
    cell_u: torch.Tensor,           # [B,3,3], kept for API compatibility
    delta_params: torch.Tensor,     # [B,<=6]
    lattice_type: torch.Tensor,     # [B] 0..5
    cell_param_u: torch.Tensor,     # [B,6] a,b,c,alpha,beta,gamma (radians)
    eps: float = 1e-8,
    delta_clip: float = 2.0,
):
    """
    Hard crystal-system parameterization:
      lt=0 triclinic: a,b,c, alpha,beta,gamma (6 dof)
      lt=1 monoclinic: a,b,c,beta (4 dof), alpha=gamma=90
      lt=2 orthorhombic: a,b,c (3 dof), angles 90
      lt=3 tetragonal: a,c (2 dof), b=a, angles 90
      lt=4 hex/trig: a,c (2 dof), b=a, alpha=beta=90, gamma=120
      lt=5 cubic: a (1 dof), b=c=a, angles 90
    """
    delta_params = torch.nan_to_num(delta_params, nan=0.0, posinf=0.0, neginf=0.0)
    cell_param_u = torch.nan_to_num(cell_param_u, nan=0.0, posinf=0.0, neginf=0.0)
    lattice_type = lattice_type.long()

    B = cell_param_u.size(0)
    device = cell_param_u.device
    dtype = delta_params.dtype

    a0, b0, c0, alpha0, beta0, gamma0 = torch.split(cell_param_u, 1, dim=-1)
    a0 = a0.squeeze(-1)
    b0 = b0.squeeze(-1)
    c0 = c0.squeeze(-1)
    alpha0 = a0.new_tensor(alpha0.squeeze(-1))
    beta0 = a0.new_tensor(beta0.squeeze(-1))
    gamma0 = a0.new_tensor(gamma0.squeeze(-1))

    alpha0 = torch.clamp(alpha0, min=1e-2, max=math.pi - 1e-2)
    beta0 = torch.clamp(beta0, min=1e-2, max=math.pi - 1e-2)
    gamma0 = torch.clamp(gamma0, min=1e-2, max=math.pi - 1e-2)

    a = a0.clone()
    b = b0.clone()
    c = c0.clone()
    alpha = alpha0.clone()
    beta = beta0.clone()
    gamma = gamma0.clone()

    def clamp_angle(x):
        return torch.clamp(x, min=1e-2, max=math.pi - 1e-2)

    def pos_len_mul(base, delta):
        delta = torch.clamp(delta, min=-float(delta_clip), max=float(delta_clip))
        return torch.clamp(base, min=eps) * torch.exp(delta)

    def dcol(k: int):
        if delta_params.size(1) > k:
            return delta_params[:, k]
        return torch.zeros(B, device=device, dtype=dtype)

    d0, d1, d2, d3, d4, d5 = dcol(0), dcol(1), dcol(2), dcol(3), dcol(4), dcol(5)
    lt = lattice_type.long()

    # triclinic
    m = (lt == 0)
    if m.any():
        a[m] = pos_len_mul(a0[m], d0[m])
        b[m] = pos_len_mul(b0[m], d1[m])
        c[m] = pos_len_mul(c0[m], d2[m])
        alpha[m] = clamp_angle(alpha0[m] + d3[m])
        beta[m] = clamp_angle(beta0[m] + d4[m])
        gamma[m] = clamp_angle(gamma0[m] + d5[m])

    # monoclinic
    m = (lt == 1)
    if m.any():
        a[m] = pos_len_mul(a0[m], d0[m])
        b[m] = pos_len_mul(b0[m], d1[m])
        c[m] = pos_len_mul(c0[m], d2[m])
        alpha[m] = math.pi / 2
        gamma[m] = math.pi / 2
        beta[m] = clamp_angle(beta0[m] + d3[m])

    # orthorhombic
    m = (lt == 2)
    if m.any():
        a[m] = pos_len_mul(a0[m], d0[m])
        b[m] = pos_len_mul(b0[m], d1[m])
        c[m] = pos_len_mul(c0[m], d2[m])
        alpha[m] = math.pi / 2
        beta[m] = math.pi / 2
        gamma[m] = math.pi / 2

    # tetragonal
    m = (lt == 3)
    if m.any():
        a_new = pos_len_mul(a0[m], d0[m])
        c_new = pos_len_mul(c0[m], d1[m])
        a[m] = a_new
        b[m] = a_new
        c[m] = c_new
        alpha[m] = math.pi / 2
        beta[m] = math.pi / 2
        gamma[m] = math.pi / 2

    # hex / trig
    m = (lt == 4)
    if m.any():
        a_new = pos_len_mul(a0[m], d0[m])
        c_new = pos_len_mul(c0[m], d1[m])
        a[m] = a_new
        b[m] = a_new
        c[m] = c_new
        alpha[m] = math.pi / 2
        beta[m] = math.pi / 2
        gamma[m] = 2 * math.pi / 3

    # cubic
    m = (lt == 5)
    if m.any():
        a_new = pos_len_mul(a0[m], d0[m])
        a[m] = a_new
        b[m] = a_new
        c[m] = a_new
        alpha[m] = math.pi / 2
        beta[m] = math.pi / 2
        gamma[m] = math.pi / 2

    ca = torch.cos(alpha)
    cb = torch.cos(beta)
    cg = torch.cos(gamma)
    sg = torch.sin(gamma).clamp(min=1e-4)

    ax = a
    ay = torch.zeros_like(a)
    az = torch.zeros_like(a)

    bx = b * cg
    by = b * sg
    bz = torch.zeros_like(b)

    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = torch.clamp(c * c - cx * cx - cy * cy, min=eps)
    cz = torch.sqrt(cz_sq)

    cell_pred = torch.stack(
        [
            torch.stack([ax, ay, az], dim=-1),
            torch.stack([bx, by, bz], dim=-1),
            torch.stack([cx, cy, cz], dim=-1),
        ],
        dim=1,
    )

    cell_param_pred = torch.stack([a, b, c, alpha, beta, gamma], dim=-1)
    return cell_pred, cell_param_pred


# ============================================================
# Model blocks
# ============================================================

class MessagePassing(nn.Module):
    def __init__(self, hidden_channels: int, edge_feat_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels * 3),
        )
        self.edge_proj = nn.Linear(edge_feat_channels, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

    def forward(self, x, vec, edge_index, edge_feat, edge_vector):
        j, i = edge_index
        rbf_h = self.edge_proj(edge_feat)
        x_h = self.x_proj(x)
        x_ji1, x_ji2, x_ji3 = torch.split(
            x_h[j] * rbf_h * self.inv_sqrt_3, self.hidden_channels, dim=-1
        )

        vec_ji = x_ji1.unsqueeze(1) * vec[j] + x_ji2.unsqueeze(1) * edge_vector.unsqueeze(2)
        vec_ji = vec_ji * self.inv_sqrt_h

        d_vec = scatter(vec_ji, index=i, dim=0, dim_size=x.size(0), reduce="sum")
        d_x = scatter(x_ji3, index=i, dim=0, dim_size=x.size(0), reduce="sum")
        return d_x, d_vec


class MessageUpdating(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        x_vec_h = self.xvec_proj(
            torch.cat([x, torch.sqrt(torch.sum(vec2 ** 2, dim=-2) + 1e-8)], dim=-1)
        )
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = (xvec1 + xvec2 * vec_dot) * self.inv_sqrt_2
        dvec = xvec3.unsqueeze(1) * vec1
        return dx, dvec


class ChargeTransferLayer(nn.Module):
    """
    Electronegative-gated local transfer.
    """
    def __init__(
        self,
        hidden_channels: int,
        edge_feat_channels: int,
        num_elements: int = 118,
        chi0: float = 0.4,
        tau: float = 0.15,
        q_clip: float = 2.5,
        learnable_chi: bool = True,
        z_index_mode: str = "atomic_number",
    ):
        super().__init__()
        self.num_elements = int(num_elements)
        self.q_clip = float(q_clip)
        self.z_index_mode = str(z_index_mode).lower()

        chi_init, chi_source = self._build_pauling_table(self.num_elements)

        self.chi = nn.Embedding(self.num_elements + 1, 1, padding_idx=0)
        with torch.no_grad():
            self.chi.weight[:, 0].copy_(chi_init)
        self.chi.weight.requires_grad_(bool(learnable_chi))

        if (chi_source == "fallback") and (not learnable_chi):
            warnings.warn(
                "ChargeTransferLayer: using fallback electronegativity table with "
                "learnable_chi=False. Missing elements keep default chi and may bias results.",
                stacklevel=2,
            )

        self.register_buffer("chi0", torch.tensor(float(chi0)))
        self.register_buffer("tau", torch.tensor(float(tau)))

        in_dim = hidden_channels * 2 + edge_feat_channels + 3
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.node_fuse = nn.Sequential(
            nn.Linear(hidden_channels + 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)
        nn.init.zeros_(self.node_fuse[-1].weight)
        nn.init.zeros_(self.node_fuse[-1].bias)

    @staticmethod
    def _build_pauling_table(num_elements: int):
        tab = torch.full((num_elements + 1,), 1.70, dtype=torch.float32)
        tab[0] = 0.0

        try:
            from pymatgen.core import Element
            for Z in range(1, num_elements + 1):
                x = Element.from_Z(Z).X
                if x is not None:
                    tab[Z] = float(x)
            return tab, "pymatgen"
        except Exception:
            pass

        known = {
            1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
            11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
            19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83,
            27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55,
            35: 2.96, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90,
            44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96, 51: 2.05,
            52: 2.10, 53: 2.66, 55: 0.79, 56: 0.89, 57: 1.10,
            58: 1.12, 59: 1.13, 60: 1.14, 61: 1.13, 62: 1.17, 63: 1.20, 64: 1.20,
            65: 1.10, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.10, 71: 1.27,
            72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28,
            79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.00, 85: 2.20,
            89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38, 93: 1.36, 94: 1.28, 95: 1.13, 96: 1.28,
        }
        for z, v in known.items():
            if z <= num_elements:
                tab[z] = float(v)
        return tab, "fallback"

    def _z_to_index(self, z: torch.Tensor) -> torch.Tensor:
        z = z.long().view(-1)
        if self.z_index_mode == "atomic_number":
            return z.clamp(1, self.num_elements)
        if self.z_index_mode == "zero_based":
            return (z + 1).clamp(1, self.num_elements)
        raise ValueError(
            f"Invalid z_index_mode={self.z_index_mode}, expected 'atomic_number' or 'zero_based'."
        )

    def forward(self, x, z, edge_index, edge_feat, batch):
        j, i = edge_index.long()
        batch = batch.long()
        edge_feat = edge_feat.to(dtype=x.dtype)

        z_idx = self._z_to_index(z)
        chi = self.chi(z_idx).squeeze(-1).to(dtype=x.dtype)

        dchi = chi[j] - chi[i]
        gate = torch.sigmoid(
            (dchi.abs() - self.chi0.to(dchi.dtype)) / (self.tau.to(dchi.dtype) + 1e-8)
        )

        edge_in = torch.cat(
            [
                x[i], x[j], edge_feat,
                dchi.unsqueeze(-1),
                chi[i].unsqueeze(-1),
                chi[j].unsqueeze(-1),
            ],
            dim=-1,
        )
        dq_ji = self.edge_mlp(edge_in).squeeze(-1)
        q_raw = scatter(gate * dq_ji, i, dim=0, dim_size=x.size(0), reduce="sum")

        q = q_raw - scatter(q_raw, batch, dim=0, reduce="mean")[batch]
        if self.q_clip > 0:
            q = self.q_clip * torch.tanh(q / self.q_clip)
        q = q - scatter(q, batch, dim=0, reduce="mean")[batch]

        x_delta = self.node_fuse(torch.cat([x, q.unsqueeze(-1), chi.unsqueeze(-1)], dim=-1))
        x_ct = x + x_delta

        q_sum = scatter(q, batch, dim=0, reduce="sum")
        stats = {
            "chi_atom": chi,
            "q_sum_per_graph": q_sum,
            "gate_mean": gate.mean().detach(),
            "gate_saturated_frac": ((gate > 0.99) | (gate < 0.01)).float().mean().detach(),
            "q_abs_mean": q.abs().mean().detach(),
            "q_neutral_l1": q_sum.abs().mean().detach(),
            "q_neutral_max": q_sum.abs().max().detach(),
        }
        return x_ct, q, stats


class StructuredCellHead(nn.Module):
    """
    Bounded structured cell head:
      - isotropic log-volume-like term
      - anisotropic length residual
      - angle residual
    """
    def __init__(self, g_dim: int, hidden_channels: int, max_cell_dof: int):
        super().__init__()
        self.max_cell_dof = int(max_cell_dof)

        trunk_h = max(hidden_channels // 2, 128)
        self.trunk = nn.Sequential(
            nn.Linear(g_dim, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, trunk_h),
            ScaledSiLU(),
        )

        self.vol_head = nn.Linear(trunk_h, 1)
        self.shape_head = nn.Linear(trunk_h, 3)
        self.ang_head = nn.Linear(trunk_h, 3)

        nn.init.zeros_(self.vol_head.weight)
        nn.init.zeros_(self.vol_head.bias)
        nn.init.zeros_(self.shape_head.weight)
        nn.init.zeros_(self.shape_head.bias)
        nn.init.zeros_(self.ang_head.weight)
        nn.init.zeros_(self.ang_head.bias)

        self.max_d_logV = 0.35
        self.max_d_shape = 0.25
        self.max_d_ang = 0.20

    def forward(self, g: torch.Tensor):
        h = self.trunk(g)

        raw_logV = self.vol_head(h)
        raw_shape = self.shape_head(h)
        raw_ang = self.ang_head(h)

        d_logV = self.max_d_logV * torch.tanh(raw_logV)
        d_shape = self.max_d_shape * torch.tanh(raw_shape)
        d_ang = self.max_d_ang * torch.tanh(raw_ang)

        d_len = d_shape + d_logV / 3.0
        delta = torch.cat([d_len, d_ang], dim=-1)

        if self.max_cell_dof < 6:
            delta = delta[:, :self.max_cell_dof]
        return delta


class ThetaHead(nn.Module):
    def __init__(self, in_dim: int, theta_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, theta_hidden),
            ScaledSiLU(),
            nn.Linear(theta_hidden, theta_hidden),
            ScaledSiLU(),
            nn.Linear(theta_hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Main Model (KEEP-ONLY)
# ============================================================

class DeepRelaxSymmEnergy(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 512,
        num_layers: int = 4,
        num_rbf: int = 128,
        cutoff: float = 30.0,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        num_elements: int = 118,
        d_model: int = 128,
        max_cell_dof: int = 6,
        theta_hidden: int = 256,
        use_aux_dist: bool = True,

        # ChargeTransfer
        use_charge_transfer: bool = True,
        ct_insert_after: int = 2,
        ct_chi0: float = 0.4,
        ct_tau: float = 0.15,
        ct_q_clip: float = 2.5,
        ct_learnable_chi: bool = True,
        ct_z_index_mode: str = "atomic_number",

        # Graph pooling
        graph_pool: str = "meanmax",
        use_lattice_embed: bool = True,
        lattice_emb_dim: int = 32,
        std_min_count: int = 3,

        # Theta conditioning
        theta_use_q: bool = False,

        # Cell delta stability
        cell_delta_clip: float = 2.0,

        # High-ROI additions
        use_refinement: bool = True,
        refinement_layers: int = 1,
        theta_free_scale: float = 0.25,
        theta_ctx_dim: int = 64,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.d_model = d_model
        self.max_cell_dof = max_cell_dof
        self.use_aux_dist = bool(use_aux_dist)
        self.cell_delta_clip = float(cell_delta_clip)
        self.use_refinement = bool(use_refinement)
        self.refinement_layers = int(refinement_layers)
        self.theta_free_scale = float(theta_free_scale)

        self.edge_feat_channels = num_rbf + d_model

        self.use_charge_transfer = bool(use_charge_transfer)
        self.theta_use_q = bool(theta_use_q)
        if self.theta_use_q and (not self.use_charge_transfer):
            warnings.warn(
                "theta_use_q=True but use_charge_transfer=False; theta will fallback to x-only.",
                stacklevel=2,
            )

        ct_insert_after = int(ct_insert_after)
        self.ct_insert_after = max(1, min(max(1, num_layers), ct_insert_after))

        self.graph_pool = str(graph_pool).lower()
        if self.graph_pool not in ("mean", "meanmax", "meanmaxstd"):
            raise ValueError("graph_pool must be one of: mean | meanmax | meanmaxstd")
        self.std_min_count = int(std_min_count)

        self.use_lattice_embed = bool(use_lattice_embed)

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        # backbone
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(MessagePassing(hidden_channels, self.edge_feat_channels))
            self.update_layers.append(MessageUpdating(hidden_channels))

        # small refinement stack
        if self.use_refinement:
            self.refine_message_layers = nn.ModuleList()
            self.refine_update_layers = nn.ModuleList()
            for _ in range(self.refinement_layers):
                self.refine_message_layers.append(MessagePassing(hidden_channels, self.edge_feat_channels))
                self.refine_update_layers.append(MessageUpdating(hidden_channels))

        # CT block
        if self.use_charge_transfer:
            self.charge_transfer = ChargeTransferLayer(
                hidden_channels=hidden_channels,
                edge_feat_channels=self.edge_feat_channels,
                num_elements=num_elements,
                chi0=ct_chi0,
                tau=ct_tau,
                q_clip=ct_q_clip,
                learnable_chi=ct_learnable_chi,
                z_index_mode=ct_z_index_mode,
            )

        # graph pooled feature dim from scalar x
        if self.graph_pool == "mean":
            g_x_dim = hidden_channels
        elif self.graph_pool == "meanmax":
            g_x_dim = hidden_channels * 2
        else:
            g_x_dim = hidden_channels * 3

        # graph pooled feature dim from vec norm summary
        if self.graph_pool == "mean":
            g_v_dim = hidden_channels
        elif self.graph_pool == "meanmax":
            g_v_dim = hidden_channels * 2
        else:
            g_v_dim = hidden_channels * 3

        g_core_dim = g_x_dim + g_v_dim

        if self.use_lattice_embed:
            self.lattice_emb = nn.Embedding(6, lattice_emb_dim)
            g_dim = g_core_dim + lattice_emb_dim
        else:
            g_dim = g_core_dim

        # heads
        self.cell_head = StructuredCellHead(
            g_dim=g_dim,
            hidden_channels=hidden_channels,
            max_cell_dof=max_cell_dof,
        )
        if self.use_refinement:
            self.cell_head_refine = StructuredCellHead(
                g_dim=g_dim,
                hidden_channels=hidden_channels,
                max_cell_dof=max_cell_dof,
            )

        # theta context projections
        self.cell_ctx_proj = nn.Sequential(
            nn.Linear(max_cell_dof, theta_ctx_dim),
            ScaledSiLU(),
            nn.Linear(theta_ctx_dim, theta_ctx_dim),
        )
        if self.use_refinement:
            self.cell_ctx_proj_refine = nn.Sequential(
                nn.Linear(max_cell_dof, theta_ctx_dim),
                ScaledSiLU(),
                nn.Linear(theta_ctx_dim, theta_ctx_dim),
            )

        wyc_scalar_dim = hidden_channels * 2         # mean + max of x
        wyc_vec_dim = hidden_channels * 2            # mean + max of vec_norm
        theta_in_dim = wyc_scalar_dim + wyc_vec_dim + g_dim + theta_ctx_dim + (1 if self.theta_use_q else 0)

        self.theta_head_keep = ThetaHead(theta_in_dim, theta_hidden)
        if self.use_refinement:
            self.theta_head_keep_refine = ThetaHead(theta_in_dim, theta_hidden)

        # optional aux distance head
        if self.use_aux_dist:
            self.aux_node = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                ScaledSiLU(),
            )
            self.aux_edge = nn.Sequential(
                nn.Linear(self.edge_feat_channels, hidden_channels),
                ScaledSiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                ScaledSiLU(),
            )
            self.aux_out = nn.Sequential(
                nn.Linear(hidden_channels * 3, hidden_channels),
                ScaledSiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                ScaledSiLU(),
                nn.Linear(hidden_channels, 2),
            )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    # --------------------------------------------------------
    # geometry utils
    # --------------------------------------------------------
    def _build_edges(self, pos, cell, cell_offsets, edge_index, neighbors):
        j, i = edge_index
        cell_offsets_unsqueeze = cell_offsets.unsqueeze(1).to(dtype=cell.dtype)
        abc_unsqueeze = cell.repeat_interleave(neighbors, dim=0)
        vecs = (pos[j] + (cell_offsets_unsqueeze @ abc_unsqueeze).squeeze(1)) - pos[i]
        edge_dist = vector_norm(vecs, dim=-1)
        edge_vector = -vecs / edge_dist.unsqueeze(-1).clamp(min=1e-8)
        return edge_dist, edge_vector

    def _edge_features(self, edge_dist, cell_offsets):
        edge_rbf = self.radial_basis(edge_dist)
        cell_offsets_int = cell_offsets_to_num(cell_offsets)
        cof_emb = sinusoidal_positional_encoding(cell_offsets_int, d_model=self.d_model).to(edge_rbf.dtype)
        return torch.cat([edge_rbf, cof_emb], dim=-1)

    @staticmethod
    def _cell_dof_mask_from_lattice_type(lattice_type: torch.Tensor, max_cell_dof: int = 6) -> torch.Tensor:
        lt = lattice_type.long().view(-1)
        B = lt.size(0)
        device = lt.device
        mask = torch.zeros((B, 6), device=device, dtype=torch.float32)

        mask[lt == 0, :] = 1.0
        mask[lt == 1, 0:4] = 1.0
        mask[lt == 2, 0:3] = 1.0
        mask[lt == 3, 0:2] = 1.0
        mask[lt == 4, 0:2] = 1.0
        mask[lt == 5, 0:1] = 1.0

        if int(max_cell_dof) < 6:
            mask = mask[:, :int(max_cell_dof)]
        return mask

    def _apply_cell_dof_mask(self, delta_cell: torch.Tensor, lattice_type: torch.Tensor):
        if lattice_type is None:
            return delta_cell
        dof_mask = self._cell_dof_mask_from_lattice_type(
            lattice_type=lattice_type,
            max_cell_dof=self.max_cell_dof,
        ).to(dtype=delta_cell.dtype, device=delta_cell.device)
        return delta_cell * dof_mask

    def _pool_stats(self, x: torch.Tensor, batch: torch.Tensor):
        x_mean = scatter(x, batch, dim=0, reduce="mean")
        feats = [x_mean]

        if self.graph_pool in ("meanmax", "meanmaxstd"):
            x_max = scatter(x, batch, dim=0, reduce="max")
            feats.append(x_max)

        if self.graph_pool == "meanmaxstd":
            x2 = scatter(x * x, batch, dim=0, reduce="mean")
            var = torch.clamp(x2 - x_mean * x_mean, min=0.0)
            x_std = torch.sqrt(var + 1e-8)

            if self.std_min_count > 1:
                counts = scatter(x.new_ones(batch.size(0)), batch, dim=0, reduce="sum").unsqueeze(-1)
                x_std = torch.where(counts >= float(self.std_min_count), x_std, torch.zeros_like(x_std))
            feats.append(x_std)

        return torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]

    def _graph_pool(self, x: torch.Tensor, vec: torch.Tensor, batch: torch.Tensor, lattice_type: torch.Tensor = None):
        g_x = self._pool_stats(x, batch)
        vec_norm = torch.sqrt(torch.sum(vec ** 2, dim=1) + 1e-8)   # [N,H]
        g_v = self._pool_stats(vec_norm, batch)
        g = torch.cat([g_x, g_v], dim=-1)

        if self.use_lattice_embed:
            if lattice_type is not None:
                lt = lattice_type.long().view(-1).clamp(0, 5)
                lt_emb = self.lattice_emb(lt).to(dtype=g.dtype)
            else:
                lt_emb = g.new_zeros((g.size(0), self.lattice_emb.embedding_dim))
            g = torch.cat([g, lt_emb], dim=-1)

        return g

    def _wyc_pool(self, x: torch.Tensor, vec: torch.Tensor, idx_u: torch.Tensor, nwu: int):
        wyc_x_mean = scatter(x, idx_u, dim=0, dim_size=nwu, reduce="mean")
        wyc_x_max = scatter(x, idx_u, dim=0, dim_size=nwu, reduce="max")

        vec_norm = torch.sqrt(torch.sum(vec ** 2, dim=1) + 1e-8)
        wyc_v_mean = scatter(vec_norm, idx_u, dim=0, dim_size=nwu, reduce="mean")
        wyc_v_max = scatter(vec_norm, idx_u, dim=0, dim_size=nwu, reduce="max")

        return torch.cat([wyc_x_mean, wyc_x_max, wyc_v_mean, wyc_v_max], dim=-1)

    def _theta_free_to_unit(self, theta_raw, dof_mask):
        theta_unit = self.theta_free_scale * torch.tanh(theta_raw)
        theta_unit = _wrap01(theta_unit)
        return theta_unit * dof_mask.to(theta_unit.dtype)

    def _rep_frac_from_theta(self, theta_unit, A, b):
        A = A.to(dtype=theta_unit.dtype, device=theta_unit.device)
        b = b.to(dtype=theta_unit.dtype, device=theta_unit.device)
        rep = torch.bmm(A, theta_unit.unsqueeze(-1)).squeeze(-1) + b
        return _wrap01(rep)

    def _symm_expand(self, rep_frac, cell_pred, atom_wyc_global, atom_op_id, atom_batch, symm_R, symm_t):
        atom_wyc_global = atom_wyc_global.long()
        atom_op_id = atom_op_id.long()
        atom_batch = atom_batch.long()

        R = symm_R[atom_batch, atom_op_id].to(dtype=rep_frac.dtype, device=rep_frac.device)
        t = symm_t[atom_batch, atom_op_id].to(dtype=rep_frac.dtype, device=rep_frac.device)

        frac_rep = rep_frac[atom_wyc_global]
        frac = torch.bmm(R, frac_rep.unsqueeze(-1)).squeeze(-1) + t
        frac = _wrap01(frac)

        cell_atom = cell_pred[atom_batch]
        pos = torch.bmm(frac.unsqueeze(1), cell_atom).squeeze(1)
        return frac, pos

    def _run_mp_stack(self, x, vec, edge_index, edge_feat, edge_vector, msg_layers, upd_layers):
        for k in range(len(msg_layers)):
            dx, dvec = msg_layers[k](x, vec, edge_index, edge_feat, edge_vector)
            x = (x + dx) * self.inv_sqrt_2
            vec = vec + dvec

            dx, dvec = upd_layers[k](x, vec)
            x = x + dx
            vec = vec + dvec
        return x, vec

    # --------------------------------------------------------
    # keep-only geometry builder
    # --------------------------------------------------------
    def geometry_from_latent_keep(self, data, delta_cell, theta_keep, return_delta_cell_eff: bool = False):
        cell_u = data.cell_u
        atom_batch = getattr(data, "atom_batch", data.batch)

        delta_cell = torch.nan_to_num(delta_cell, nan=0.0, posinf=0.0, neginf=0.0)
        theta_keep = torch.nan_to_num(theta_keep, nan=0.0, posinf=0.0, neginf=0.0)

        if (
            hasattr(data, "cell_param_u")
            and hasattr(data, "lattice_type")
            and data.cell_param_u is not None
            and data.lattice_type is not None
        ):
            delta_cell_eff = self._apply_cell_dof_mask(delta_cell, data.lattice_type)
            cell_pred, cell_param_pred = build_cell_from_params(
                cell_u=cell_u,
                delta_params=delta_cell_eff,
                lattice_type=data.lattice_type,
                cell_param_u=data.cell_param_u,
                delta_clip=self.cell_delta_clip,
            )

            bad_cell_graph = (~torch.isfinite(cell_pred.view(cell_pred.size(0), -1)).all(dim=-1))
            bad_param_graph = (~torch.isfinite(cell_param_pred).all(dim=-1))
            bad_graph = bad_cell_graph | bad_param_graph

            if bad_graph.any():
                cell_pred = cell_pred.clone()
                cell_param_pred = cell_param_pred.clone()
                delta_cell_eff = delta_cell_eff.clone()

                cell_pred[bad_graph] = data.cell_u[bad_graph].to(cell_pred.dtype)
                cell_param_pred[bad_graph] = data.cell_param_u[bad_graph].to(cell_param_pred.dtype)
                delta_cell_eff[bad_graph] = 0.0
        else:
            delta_cell_eff = delta_cell
            cell_pred = cell_u + delta_cell.new_zeros((cell_u.size(0), 3, 3))
            cell_param_pred = delta_cell.new_zeros((cell_u.size(0), 6))

        rep_frac_keep = self._rep_frac_from_theta(theta_keep, data.wyc_A_u, data.wyc_b_u)
        frac_keep, pos_keep = self._symm_expand(
            rep_frac=rep_frac_keep,
            cell_pred=cell_pred,
            atom_wyc_global=data.atom_wyc_global_u,
            atom_op_id=data.atom_op_id_u,
            atom_batch=atom_batch,
            symm_R=data.symm_R_u,
            symm_t=data.symm_t_u,
        )

        if return_delta_cell_eff:
            return (
                cell_pred,
                cell_param_pred,
                rep_frac_keep,
                frac_keep,
                pos_keep,
                delta_cell_eff,
            )

        return (
            cell_pred,
            cell_param_pred,
            rep_frac_keep,
            frac_keep,
            pos_keep,
        )

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------
    def forward(self, data, y_change_for_aux: torch.Tensor = None):
        """
        KEEP-only forward with:
          - coarse prediction on unrelaxed geometry
          - optional 1-step refinement on predicted geometry
          - theta explicitly conditioned on cell latent + graph context
        """
        pos_u = data.pos_u
        cell_u = data.cell_u
        cell_offsets = data.cell_offsets
        edge_index = data.edge_index
        neighbors = data.neighbors

        batch = data.batch.long()
        atom_batch = getattr(data, "atom_batch", batch).long()
        z = _get_atomic_numbers(data).clamp_min(0)

        # edges on unrelaxed geometry
        edge_dist_u, edge_vector_u = self._build_edges(pos_u, cell_u, cell_offsets, edge_index, neighbors)
        edge_feat_u = self._edge_features(edge_dist_u, cell_offsets)

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device, dtype=x.dtype)

        q_pred = None
        ct_stats = {}
        x_pre_ct = None
        ct_inserted = False

        # backbone MP on unrelaxed geometry
        for k in range(self.num_layers):
            dx, dvec = self.message_layers[k](x, vec, edge_index, edge_feat_u, edge_vector_u)
            x = (x + dx) * self.inv_sqrt_2
            vec = vec + dvec

            dx, dvec = self.update_layers[k](x, vec)
            x = x + dx
            vec = vec + dvec

            if self.use_charge_transfer and (not ct_inserted) and ((k + 1) == self.ct_insert_after):
                x_pre_ct = x
                x, q_pred, ct_stats = self.charge_transfer(
                    x=x,
                    z=z,
                    edge_index=edge_index,
                    edge_feat=edge_feat_u,
                    batch=batch,
                )
                ct_inserted = True

        if self.use_charge_transfer and (not ct_inserted):
            x_pre_ct = x
            x, q_pred, ct_stats = self.charge_transfer(
                x=x,
                z=z,
                edge_index=edge_index,
                edge_feat=edge_feat_u,
                batch=batch,
            )

        lattice_type = data.lattice_type if hasattr(data, "lattice_type") else None
        g = self._graph_pool(x, vec, batch, lattice_type=lattice_type)

        # coarse cell latent
        delta_cell_0 = self.cell_head(g)
        cell_ctx_0 = self.cell_ctx_proj(delta_cell_0)

        # coarse theta latent
        idx_u = data.atom_wyc_global_u.long()
        nwu = int(data.wyc_A_u.size(0))
        wyc_feat_keep = self._wyc_pool(x, vec, idx_u, nwu)
        wyc_batch_u = data.wyc_batch_u.long() if hasattr(data, "wyc_batch_u") else scatter(atom_batch, idx_u, dim=0, dim_size=nwu, reduce="min")

        theta_inputs = [
            wyc_feat_keep,
            g[wyc_batch_u],
            cell_ctx_0[wyc_batch_u],
        ]
        if self.theta_use_q and (q_pred is not None):
            wyc_q_keep = scatter(q_pred, idx_u, dim=0, dim_size=nwu, reduce="mean").unsqueeze(-1)
            theta_inputs.append(wyc_q_keep)

        theta_keep_in = torch.cat(theta_inputs, dim=-1)
        theta_keep_raw_0 = self.theta_head_keep(theta_keep_in)
        theta_keep_0 = self._theta_free_to_unit(theta_keep_raw_0, data.wyc_dof_mask_u)

        (
            cell_pred_0,
            cell_param_pred_0,
            rep_frac_keep_0,
            frac_keep_0,
            pos_keep_0,
            delta_cell_eff_0,
        ) = self.geometry_from_latent_keep(
            data,
            delta_cell_0,
            theta_keep_0,
            return_delta_cell_eff=True,
        )

        # optional 1-step refinement
        if self.use_refinement and self.refinement_layers > 0:
            edge_dist_r, edge_vector_r = self._build_edges(
                pos_keep_0, cell_pred_0, cell_offsets, edge_index, neighbors
            )
            edge_feat_r = self._edge_features(edge_dist_r, cell_offsets)

            x_ref, vec_ref = self._run_mp_stack(
                x=x,
                vec=vec,
                edge_index=edge_index,
                edge_feat=edge_feat_r,
                edge_vector=edge_vector_r,
                msg_layers=self.refine_message_layers,
                upd_layers=self.refine_update_layers,
            )

            g_ref = self._graph_pool(x_ref, vec_ref, batch, lattice_type=lattice_type)
            delta_cell_res = self.cell_head_refine(g_ref)
            delta_cell = delta_cell_0 + 0.5 * delta_cell_res
            cell_ctx_ref = self.cell_ctx_proj_refine(delta_cell)

            wyc_feat_keep_ref = self._wyc_pool(x_ref, vec_ref, idx_u, nwu)

            theta_inputs_ref = [
                wyc_feat_keep_ref,
                g_ref[wyc_batch_u],
                cell_ctx_ref[wyc_batch_u],
            ]
            if self.theta_use_q and (q_pred is not None):
                wyc_q_keep = scatter(q_pred, idx_u, dim=0, dim_size=nwu, reduce="mean").unsqueeze(-1)
                theta_inputs_ref.append(wyc_q_keep)

            theta_keep_in_ref = torch.cat(theta_inputs_ref, dim=-1)
            theta_keep_raw_res = self.theta_head_keep_refine(theta_keep_in_ref)
            theta_keep_raw = theta_keep_raw_0 + 0.5 * theta_keep_raw_res
            theta_keep = self._theta_free_to_unit(theta_keep_raw, data.wyc_dof_mask_u)

            (
                cell_pred,
                cell_param_pred,
                rep_frac_keep,
                frac_keep,
                pos_keep,
                delta_cell_eff,
            ) = self.geometry_from_latent_keep(
                data,
                delta_cell,
                theta_keep,
                return_delta_cell_eff=True,
            )

            x_final = x_ref
            vec_final = vec_ref
        else:
            cell_pred = cell_pred_0
            cell_param_pred = cell_param_pred_0
            rep_frac_keep = rep_frac_keep_0
            frac_keep = frac_keep_0
            pos_keep = pos_keep_0
            delta_cell = delta_cell_0
            delta_cell_eff = delta_cell_eff_0
            theta_keep = theta_keep_0
            x_final = x
            vec_final = vec

        out = {
            "cell_pred": cell_pred,
            "cell_param_pred": cell_param_pred,

            "rep_frac_keep": rep_frac_keep,
            "frac_keep": frac_keep,
            "pos_keep": pos_keep,

            # compatibility aliases
            "rep_frac_u": rep_frac_keep,
            "frac_same": frac_keep,
            "pos_same": pos_keep,

            # coarse outputs for training/debug
            "cell_pred_0": cell_pred_0,
            "cell_param_pred_0": cell_param_pred_0,
            "rep_frac_keep_0": rep_frac_keep_0,
            "frac_keep_0": frac_keep_0,
            "pos_keep_0": pos_keep_0,

            # debug
            "delta_cell": delta_cell,
            "delta_cell_eff": delta_cell_eff,
            "theta_keep": theta_keep,
            "x_backbone": x_final,
            "vec_backbone": vec_final,
        }

        if q_pred is not None:
            out["q_pred"] = q_pred
            out["chi_atom"] = ct_stats["chi_atom"]
            out["q_sum_per_graph"] = ct_stats["q_sum_per_graph"]
            out["ct_gate_mean"] = ct_stats["gate_mean"]
            out["ct_gate_saturated_frac"] = ct_stats["gate_saturated_frac"]
            out["ct_q_abs_mean"] = ct_stats["q_abs_mean"]
            out["q_neutral_l1"] = ct_stats["q_neutral_l1"]
            out["q_neutral_max"] = ct_stats["q_neutral_max"]
            if x_pre_ct is not None:
                out["x_pre_ct"] = x_pre_ct

        # aux head on final KEEP geometry
        if self.use_aux_dist and (y_change_for_aux is not None):
            pos_pred_aux = pos_keep
            edge_dist_aux, _ = self._build_edges(pos_pred_aux, cell_pred, cell_offsets, edge_index, neighbors)
            edge_feat_aux = self._edge_features(edge_dist_aux, cell_offsets)

            edge_h = self.aux_edge(edge_feat_aux)
            j, i = edge_index
            xh = self.aux_node(x_final)
            dist_feat = torch.cat([xh[i], xh[j], edge_h], dim=-1)

            mu_rawvar = self.aux_out(dist_feat)
            mu, raw_var = torch.split(mu_rawvar, 1, dim=-1)

            out["aux_mu"] = F.softplus(mu).squeeze(-1)
            out["aux_raw_var"] = raw_var.squeeze(-1)

        return out
