#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import math
import shutil
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import spglib
from ase import Atoms
from ase.io import write
from torch.utils.data import DataLoader

from crystal_data import CrystalDataset, make_batch
from SymmRelax import (
    DeepRelaxSymmEnergy,
    gaussian_nll_1d,
    build_edge_distances_pbc,
)


# ============================================================
# Optional CGCNN energy scorer
# ============================================================

_HAS_CGCNN = False
try:
    from cgcnn.data import CIFData, collate_pool
    from cgcnn.model import CrystalGraphConvNet
    _HAS_CGCNN = True
except Exception:
    _HAS_CGCNN = False


# ============================================================
# Config
# ============================================================

def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)

    script_dir = Path(__file__).resolve().parent
    p2 = script_dir / path
    if p2.is_file():
        return str(p2)

    raise FileNotFoundError(f"File not found: {path}")


def load_json_config(path: str) -> dict:
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def torch_load_compat(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ============================================================
# Basic utils
# ============================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_first(x, default=None):
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) else default
    return x


def safe_log(x: float, eps: float = 1e-12) -> float:
    return float(math.log(max(float(x), eps)))


def frac_mod1_np(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)


def frac_diff_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    d = d - np.round(d)
    return d


def _get_atomic_numbers_tensor(data) -> torch.Tensor:
    if hasattr(data, "atomic_numbers") and data.atomic_numbers is not None:
        z = data.atomic_numbers.view(-1)
    elif hasattr(data, "z") and data.z is not None:
        z = data.z.view(-1)
    elif hasattr(data, "x") and data.x is not None:
        x = data.x
        z = x if x.dim() == 1 else x[:, 0]
    else:
        raise AttributeError("Cannot find atomic numbers in data.atomic_numbers / data.z / data.x")

    if torch.is_floating_point(z):
        z = torch.round(z)

    return z.long()


# ============================================================
# Frac shape normalization
# ============================================================

def ensure_frac_n3_np(frac_np: np.ndarray, n_atoms: int):
    if frac_np is None:
        return None

    frac_np = np.asarray(frac_np)

    while frac_np.ndim > 2 and frac_np.shape[0] == 1:
        frac_np = frac_np[0]

    if frac_np.ndim == 1:
        if frac_np.size == n_atoms * 3:
            frac_np = frac_np.reshape(n_atoms, 3)
        elif frac_np.size % 3 == 0:
            cand = frac_np.reshape(-1, 3)
            if cand.shape[0] == n_atoms:
                frac_np = cand
            else:
                return None
        else:
            return None

    if frac_np.ndim != 2:
        return None

    if frac_np.shape == (n_atoms, 3):
        return frac_np

    if frac_np.shape == (3, n_atoms):
        return frac_np.T

    if frac_np.shape[0] == n_atoms and frac_np.shape[1] >= 3:
        return frac_np[:, :3]

    if frac_np.size == n_atoms * 3:
        return frac_np.reshape(n_atoms, 3)

    return None


def ensure_frac_n3_torch(frac: torch.Tensor, n_atoms: int):
    if frac is None or (not torch.is_tensor(frac)):
        return None

    while frac.dim() > 2 and frac.size(0) == 1:
        frac = frac[0]

    if frac.dim() == 1:
        if frac.numel() == n_atoms * 3:
            return frac.view(n_atoms, 3)
        if frac.numel() % 3 == 0:
            cand = frac.view(-1, 3)
            if cand.size(0) == n_atoms:
                return cand
        return None

    if frac.dim() != 2:
        return None

    if frac.size(0) == n_atoms and frac.size(1) == 3:
        return frac

    if frac.size(0) == 3 and frac.size(1) == n_atoms:
        return frac.t().contiguous()

    if frac.size(0) == n_atoms and frac.size(1) >= 3:
        return frac[:, :3].contiguous()

    if frac.numel() == n_atoms * 3:
        return frac.view(n_atoms, 3)

    return None


# ============================================================
# Cell / frac conversions
# ============================================================

def cart_to_frac_solve(pos_cart: torch.Tensor, cell33: torch.Tensor) -> torch.Tensor:
    frac = torch.linalg.solve(cell33.T.double(), pos_cart.T.double()).T.float()
    frac = frac - torch.floor(frac)
    return frac


def frac_pbc_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    d = d - torch.round(d)
    return d


def cell_to_params_torch(cell33: torch.Tensor) -> torch.Tensor:
    a_vec = cell33[0]
    b_vec = cell33[1]
    c_vec = cell33[2]

    a = torch.linalg.norm(a_vec)
    b = torch.linalg.norm(b_vec)
    c = torch.linalg.norm(c_vec)

    def angle(u, v):
        uu = torch.linalg.norm(u).clamp_min(1e-12)
        vv = torch.linalg.norm(v).clamp_min(1e-12)
        cosang = (u @ v) / (uu * vv)
        cosang = torch.clamp(cosang, -1.0, 1.0)
        return torch.arccos(cosang)

    alpha = angle(b_vec, c_vec)
    beta = angle(a_vec, c_vec)
    gamma = angle(a_vec, b_vec)

    return torch.stack([a, b, c, alpha, beta, gamma], dim=0)


def params_to_cell_torch(a, b, c, alpha, beta, gamma) -> torch.Tensor:
    ax = a
    ay = torch.zeros_like(a)
    az = torch.zeros_like(a)

    cg = torch.cos(gamma)
    sg = torch.sin(gamma).clamp_min(1e-4)

    bx = b * cg
    by = b * sg
    bz = torch.zeros_like(b)

    ca = torch.cos(alpha)
    cb = torch.cos(beta)

    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = (c * c - cx * cx - cy * cy).clamp_min(1e-8)
    cz = torch.sqrt(cz_sq)

    row_a = torch.stack([ax, ay, az], dim=0)
    row_b = torch.stack([bx, by, bz], dim=0)
    row_c = torch.stack([cx, cy, cz], dim=0)

    return torch.stack([row_a, row_b, row_c], dim=0)


# ============================================================
# Crystal-system helpers
# ============================================================

def crystal_system_id(spacegroup_number: int) -> int:
    sg = int(spacegroup_number)

    if 1 <= sg <= 2:
        return 0
    if 3 <= sg <= 15:
        return 1
    if 16 <= sg <= 74:
        return 2
    if 75 <= sg <= 142:
        return 3
    if 143 <= sg <= 167:
        return 4
    if 168 <= sg <= 194:
        return 5
    if 195 <= sg <= 230:
        return 6

    return -1


def cs_to_lattice_type(cs_id: int) -> int:
    cs = int(cs_id)

    if cs == 0:
        return 0
    if cs == 1:
        return 1
    if cs == 2:
        return 2
    if cs == 3:
        return 3
    if cs in (4, 5):
        return 4
    if cs == 6:
        return 5

    return 0


def project_cell_to_lattice_type(
    cell33: torch.Tensor,
    lattice_type: int,
    preserve_volume: bool = True,
) -> torch.Tensor:
    p = cell_to_params_torch(cell33)

    a, b, c = p[0], p[1], p[2]
    alpha, beta, gamma = p[3], p[4], p[5]

    a = torch.relu(a) + 1e-8
    b = torch.relu(b) + 1e-8
    c = torch.relu(c) + 1e-8

    def fixed_angle(x):
        return torch.tensor(x, device=cell33.device, dtype=torch.float32)

    lt = int(lattice_type)

    if lt == 0:
        pass
    elif lt == 1:
        alpha = fixed_angle(math.pi / 2)
        gamma = fixed_angle(math.pi / 2)
    elif lt == 2:
        alpha = beta = gamma = fixed_angle(math.pi / 2)
    elif lt == 3:
        s = 0.5 * (a + b)
        a = s
        b = s
        alpha = beta = gamma = fixed_angle(math.pi / 2)
    elif lt == 4:
        s = 0.5 * (a + b)
        a = s
        b = s
        alpha = fixed_angle(math.pi / 2)
        beta = fixed_angle(math.pi / 2)
        gamma = fixed_angle(2 * math.pi / 3)
    elif lt == 5:
        s = (a + b + c) / 3.0
        a = s
        b = s
        c = s
        alpha = beta = gamma = fixed_angle(math.pi / 2)

    cell_proj = params_to_cell_torch(a, b, c, alpha, beta, gamma)

    if preserve_volume:
        try:
            v0 = torch.abs(torch.det(cell33.double())).clamp_min(1e-12)
            v1 = torch.abs(torch.det(cell_proj.double())).clamp_min(1e-12)
            scale = (v0 / v1).pow(1.0 / 3.0).float()
            scale = torch.clamp(scale, 0.2, 5.0)
            cell_proj = cell_proj * scale
        except Exception:
            pass

    return cell_proj


# ============================================================
# Spglib helpers
# ============================================================

def _spglib_get(ds, key, default=None):
    try:
        return ds[key]
    except Exception:
        return getattr(ds, key, default)


def spglib_sg_cs(
    numbers_np: np.ndarray,
    frac_np: np.ndarray,
    cell_np: np.ndarray,
    symprec: float,
    angtol: float,
):
    try:
        n_atoms = int(len(numbers_np))
        frac_np = ensure_frac_n3_np(frac_np, n_atoms)

        if frac_np is None:
            return None, None

        cell_np = np.asarray(cell_np)

        ds = spglib.get_symmetry_dataset(
            (cell_np, frac_np, numbers_np),
            symprec=float(symprec),
            angle_tolerance=float(angtol),
        )

        if ds is None:
            return None, None

        sg = int(_spglib_get(ds, "number"))
        cs = crystal_system_id(sg)

        return sg, cs

    except Exception:
        return None, None


# ============================================================
# Geometry scoring
# ============================================================

def build_edge_distances_pbc_wrapper(pos, cell33, data):
    return build_edge_distances_pbc(
        pos=pos,
        cell=cell33.unsqueeze(0),
        cell_offsets=data.cell_offsets,
        edge_index=data.edge_index,
        neighbors=data.neighbors,
    )


@torch.no_grad()
def score_structure_fast(
    data,
    pos: torch.Tensor,
    cell33: torch.Tensor,
    use_aux: bool,
    aux_mu=None,
    aux_raw_var=None,
    short_cut: float = 0.75,
):
    ed = build_edge_distances_pbc_wrapper(pos, cell33, data)

    if ed.numel() == 0:
        return 0.0, {"min_d": 999.0, "frac_short": 0.0}

    min_d = float(ed.min().item())
    frac_short = float((ed < short_cut).float().mean().item())

    score = 0.0

    if min_d < 0.55:
        score += 1e6

    score += 200.0 * max(0.0, short_cut - min_d)
    score += 50.0 * frac_short

    if use_aux and (aux_mu is not None) and (aux_raw_var is not None):
        nll = gaussian_nll_1d(aux_mu, aux_raw_var, ed, eps=1e-6)
        score += 5.0 * float(nll.item())

    return float(score), {"min_d": min_d, "frac_short": frac_short}


def geom_prob_from_min_d(min_d: float, d0: float, k: float) -> float:
    d0 = float(d0)
    k = float(max(k, 1e-6))
    x = (float(min_d) - d0) / k

    if x >= 0:
        z = math.exp(-x)
        p = 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        p = z / (1.0 + z)

    return float(max(1e-12, min(1.0 - 1e-12, p)))


def differentiable_keep_penalty(
    data,
    pos: torch.Tensor,
    cell33: torch.Tensor,
    short_cut: float = 0.75,
) -> torch.Tensor:
    ed = build_edge_distances_pbc(
        pos=pos,
        cell=cell33.unsqueeze(0),
        cell_offsets=data.cell_offsets,
        edge_index=data.edge_index,
        neighbors=data.neighbors,
    )

    if ed.numel() == 0:
        return pos.sum() * 0.0

    soft_short = F.relu(float(short_cut) - ed)
    loss = 200.0 * soft_short.max() + 50.0 * soft_short.mean()

    return loss


# ============================================================
# CIF saving
# ============================================================

def save_atoms(numbers_np, frac_np, cell_np, out_path):
    if frac_np is None or cell_np is None:
        return

    n_atoms = int(len(numbers_np))
    frac_fix = ensure_frac_n3_np(frac_np, n_atoms)

    if frac_fix is None:
        return

    frac_fix = frac_mod1_np(frac_fix.astype(np.float64))

    atoms = Atoms(
        numbers=numbers_np,
        scaled_positions=frac_fix,
        cell=np.asarray(cell_np),
        pbc=True,
    )

    write(out_path, atoms)


# ============================================================
# Optional CGCNN scorer
# ============================================================

class CGCNNEnergyScorer:
    def __init__(
        self,
        ckpt_path: str,
        tmp_root: str,
        device: str = "cpu",
        max_num_nbr: int = 12,
        radius: float = 12.0,
    ):
        if not _HAS_CGCNN:
            raise RuntimeError("cgcnn is not importable. Ensure cgcnn/ is on PYTHONPATH.")

        self.ckpt_path = ckpt_path
        self.tmp_root = tmp_root
        self.device = torch.device(device)
        self.max_num_nbr = int(max_num_nbr)
        self.radius = float(radius)

        ensure_dir(self.tmp_root)

        atom_init = os.path.join(self.tmp_root, "atom_init.json")
        if not os.path.exists(atom_init):
            raise FileNotFoundError(f"missing atom_init.json in {self.tmp_root}")

        ck = torch_load_compat(self.ckpt_path, map_location="cpu")

        self.sd = ck["state_dict"]

        self.mean = ck["normalizer"]["mean"]
        self.std = ck["normalizer"]["std"]

        if not torch.is_tensor(self.mean):
            self.mean = torch.tensor(float(self.mean), dtype=torch.float32)
        else:
            self.mean = self.mean.float()

        if not torch.is_tensor(self.std):
            self.std = torch.tensor(float(self.std), dtype=torch.float32)
        else:
            self.std = self.std.float()

        self.std = self.std.clamp_min(1e-8)

        a = ck.get("args", {}) or {}
        if not isinstance(a, dict):
            a = vars(a)

        ct_insert_after = a.get("ct_insert_after", None)
        if ct_insert_after is not None:
            ct_insert_after = int(ct_insert_after)

        if "ct_learnable_chi" in a:
            ct_learnable_chi = bool(a.get("ct_learnable_chi"))
        else:
            ct_learnable_chi = not bool(a.get("ct_freeze_chi", False))

        self.model_args = dict(
            atom_fea_len=int(a.get("atom_fea_len", 64)),
            n_conv=int(a.get("n_conv", 3)),
            h_fea_len=int(a.get("h_fea_len", 128)),
            n_h=int(a.get("n_h", 1)),

            use_charge_transfer=bool(a.get("use_charge_transfer", True)),
            ct_insert_after=ct_insert_after,
            num_elements=int(a.get("num_elements", 118)),
            ct_chi0=float(a.get("ct_chi0", 0.4)),
            ct_tau=float(a.get("ct_tau", 0.15)),
            ct_q_clip=float(a.get("ct_q_clip", 2.5)),
            ct_learnable_chi=bool(ct_learnable_chi),
            ct_z_index_mode=str(a.get("ct_z_index_mode", "atomic_number")),
        )

        self.model = None
        self._cache = {}

    def _lazy_init_model(self):
        ds = CIFData(
            self.tmp_root,
            max_num_nbr=self.max_num_nbr,
            radius=self.radius,
        )

        (atom_fea, nbr_fea, _, _, _), _, _ = ds[0]

        orig_atom_fea_len = atom_fea.shape[-1]
        nbr_fea_len = nbr_fea.shape[-1]

        m = CrystalGraphConvNet(
            orig_atom_fea_len,
            nbr_fea_len,
            atom_fea_len=self.model_args["atom_fea_len"],
            n_conv=self.model_args["n_conv"],
            h_fea_len=self.model_args["h_fea_len"],
            n_h=self.model_args["n_h"],
            classification=False,

            use_charge_transfer=self.model_args.get("use_charge_transfer", True),
            ct_insert_after=self.model_args.get("ct_insert_after", None),
            num_elements=self.model_args.get("num_elements", 118),
            ct_chi0=self.model_args.get("ct_chi0", 0.4),
            ct_tau=self.model_args.get("ct_tau", 0.15),
            ct_q_clip=self.model_args.get("ct_q_clip", 2.5),
            ct_learnable_chi=self.model_args.get("ct_learnable_chi", True),
            ct_z_index_mode=self.model_args.get("ct_z_index_mode", "atomic_number"),
        ).to(self.device)

        m.load_state_dict(self.sd, strict=True)
        m.eval()

        self.model = m

    @staticmethod
    def _key_from_cif_path(cif_path: str) -> str:
        try:
            import hashlib
            with open(cif_path, "rb") as f:
                b = f.read()
            return hashlib.md5(b).hexdigest()
        except Exception:
            return cif_path

    @torch.no_grad()
    def predict_e_per_atom(self, cif_path: str) -> float:
        k = self._key_from_cif_path(cif_path)
        if k in self._cache:
            return float(self._cache[k])

        cif_id = "tmp0"

        with open(os.path.join(self.tmp_root, "id_prop.csv"), "w") as f:
            f.write(f"{cif_id},0\n")

        dst = os.path.join(self.tmp_root, f"{cif_id}.cif")
        shutil.copyfile(cif_path, dst)

        ds = CIFData(
            self.tmp_root,
            max_num_nbr=self.max_num_nbr,
            radius=self.radius,
        )

        if self.model is None:
            self._lazy_init_model()

        batch_x, _, _ = collate_pool([ds[0]])

        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, z, nbr_mask = batch_x

        atom_fea = atom_fea.to(self.device)
        nbr_fea = nbr_fea.to(self.device)
        nbr_fea_idx = nbr_fea_idx.to(self.device)
        crystal_atom_idx = [t.to(self.device) for t in crystal_atom_idx]
        z = z.to(self.device)
        nbr_mask = nbr_mask.to(self.device)

        y_norm = self.model(
            atom_fea,
            nbr_fea,
            nbr_fea_idx,
            crystal_atom_idx,
            z=z,
            nbr_mask=nbr_mask,
        ).view(-1)[0].float().cpu()

        y = (y_norm * self.std + self.mean).item()

        self._cache[k] = float(y)
        return float(y)


# ============================================================
# Candidate
# ============================================================

@dataclass
class KeepCandidate:
    name: str
    cell: torch.Tensor
    frac: torch.Tensor
    meta: dict = field(default_factory=dict)

    min_d: float = 0.0
    frac_short: float = 0.0
    p_geom: float = 1e-12
    sg_spglib: Optional[int] = None
    cs_spglib: Optional[int] = None
    score_total: float = -1e18
    energy: float = float("inf")


# ============================================================
# Optional tiny refine
# ============================================================

def keep_refine_final_frac(
    frac_init: torch.Tensor,
    frac_u: torch.Tensor,
    final_cell: torch.Tensor,
    data,
    short_cut: float = 0.75,
    steps: int = 3,
    lr: float = 1e-2,
    w_anchor: float = 1.0,
    w_short: float = 1.0,
):
    if steps <= 0:
        return frac_init.detach()

    theta = frac_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=float(lr))

    best_frac = frac_init.detach().clone()
    best_loss = None

    for _ in range(int(steps)):
        opt.zero_grad()

        frac_new = theta - torch.floor(theta)
        pos_new = torch.matmul(frac_new, final_cell)

        loss_anchor = ((frac_pbc_diff_torch(frac_new, frac_u)) ** 2).mean()

        loss_short = differentiable_keep_penalty(
            data,
            pos_new,
            final_cell,
            short_cut=float(short_cut),
        )

        loss = float(w_anchor) * loss_anchor + float(w_short) * loss_short

        if not torch.isfinite(loss):
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 10.0)
        opt.step()

        cur = float(loss.detach().item())

        if (best_loss is None) or (cur < best_loss):
            best_loss = cur
            best_frac = frac_new.detach().clone()

    return best_frac


# ============================================================
# Model loading
# ============================================================

def build_model_from_old_args(geom_args: Dict[str, Any], device: torch.device) -> DeepRelaxSymmEnergy:
    model = DeepRelaxSymmEnergy(
        hidden_channels=int(geom_args.get("hidden_channels", 384)),
        num_layers=int(geom_args.get("num_layers", 4)),
        num_rbf=int(geom_args.get("num_rbf", 128)),
        cutoff=float(geom_args.get("cutoff", 25.0)),
        num_elements=int(geom_args.get("num_elements", 118)),
        d_model=int(geom_args.get("d_model", 128)),
        max_cell_dof=int(geom_args.get("max_cell_dof", 6)),
        theta_hidden=int(geom_args.get("theta_hidden", 256)),
        use_aux_dist=bool(geom_args.get("use_aux_dist", False)),

        use_charge_transfer=bool(geom_args.get("use_charge_transfer", True)),
        ct_insert_after=int(geom_args.get("ct_insert_after", 2)),
        ct_chi0=float(geom_args.get("ct_chi0", 0.4)),
        ct_tau=float(geom_args.get("ct_tau", 0.15)),
        ct_q_clip=float(geom_args.get("ct_q_clip", 2.5)),
        ct_learnable_chi=bool(
            geom_args.get(
                "ct_learnable_chi",
                not bool(geom_args.get("ct_freeze_chi", False)),
            )
        ),
        ct_z_index_mode=str(geom_args.get("ct_z_index_mode", "atomic_number")),

        graph_pool=str(geom_args.get("graph_pool", "meanmaxstd")),
        use_lattice_embed=bool(geom_args.get("use_lattice_embed", True)),
        lattice_emb_dim=int(geom_args.get("lattice_emb_dim", 32)),
        std_min_count=int(geom_args.get("std_min_count", 3)),

        theta_use_q=bool(geom_args.get("theta_use_q", False)),
        cell_delta_clip=float(geom_args.get("cell_delta_clip", 2.0)),

        use_refinement=bool(geom_args.get("use_refinement", True)),
        refinement_layers=int(geom_args.get("refinement_layers", 1)),
        theta_free_scale=float(geom_args.get("theta_free_scale", 0.25)),
        theta_ctx_dim=int(geom_args.get("theta_ctx_dim", 64)),
    ).to(device)

    return model


def load_geom_model(ckpt_path: str, device: torch.device):
    ckpt = torch_load_compat(ckpt_path, map_location="cpu")

    model_cfg = ckpt.get("model_config", None)

    if model_cfg is not None:
        model = DeepRelaxSymmEnergy(**model_cfg).to(device)
        use_aux_dist = bool(model_cfg.get("use_aux_dist", False))
    else:
        geom_args = ckpt.get("args", {}) or {}
        if not isinstance(geom_args, dict):
            geom_args = vars(geom_args)
        model = build_model_from_old_args(geom_args, device=device)
        use_aux_dist = bool(geom_args.get("use_aux_dist", False))

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    return model, ckpt, use_aux_dist


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--geom_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="infer_config.json")

    args = ap.parse_args()

    cfg = load_json_config(args.config)

    data_cfg = cfg.get("data", {})
    runtime_cfg = cfg.get("runtime", {})
    symm_cfg = cfg.get("symmetry", {})
    score_cfg = cfg.get("scoring", {})
    post_cfg = cfg.get("postprocess", {})
    energy_cfg = cfg.get("energy", {})
    stats_cfg = cfg.get("stats", {})

    split = str(data_cfg.get("split", "test"))
    num_workers = int(data_cfg.get("num_workers", 0))

    device_name = str(runtime_cfg.get("device", "cuda"))
    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")

    ensure_dir(args.out_dir)

    init_dir = os.path.join(args.out_dir, "init")
    pred_dir = os.path.join(args.out_dir, "pred")
    dft_dir = os.path.join(args.out_dir, "dft")

    ensure_dir(init_dir)
    ensure_dir(pred_dir)
    ensure_dir(dft_dir)

    # -----------------------------
    # Dataset
    # -----------------------------
    ds = CrystalDataset(args.root, split=split)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=make_batch,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # -----------------------------
    # Model
    # -----------------------------
    geom_model, geom_ck, ckpt_use_aux = load_geom_model(args.geom_ckpt, device=device)

    use_aux_score = bool(score_cfg.get("use_aux_score", False))

    if use_aux_score and (not ckpt_use_aux):
        print("[WARN] use_aux_score=True, but checkpoint use_aux_dist=False. Aux score will be ignored.")
        use_aux_score = False

    # -----------------------------
    # Params
    # -----------------------------
    symprec = float(symm_cfg.get("symprec", 5e-2))
    angtol = float(symm_cfg.get("angtol", 5.0))

    short_cut = float(score_cfg.get("short_cut", 0.75))
    min_d_thr_keep = float(score_cfg.get("min_d_thr_keep", 0.60))
    geom_d0 = float(score_cfg.get("geom_d0", 1.6))
    geom_k = float(score_cfg.get("geom_k", 0.2))

    keep_final_bonus = float(score_cfg.get("keep_final_bonus", 0.02))
    keep_raw_penalty = float(score_cfg.get("keep_raw_penalty", 0.10))
    keep_cellu_bonus = float(score_cfg.get("keep_cellu_bonus", 0.00))
    keep_cs_drop_penalty = float(score_cfg.get("keep_cs_drop_penalty", 1.0))

    project_cell_post = bool(post_cfg.get("project_cell_post", False))
    keep_refine_steps = int(post_cfg.get("keep_refine_steps", 0))
    keep_refine_lr = float(post_cfg.get("keep_refine_lr", 1e-2))
    keep_refine_w_anchor = float(post_cfg.get("keep_refine_w_anchor", 1.0))
    keep_refine_w_short = float(post_cfg.get("keep_refine_w_short", 1.0))

    write_stats = bool(stats_cfg.get("enabled", True))
    stats_out = stats_cfg.get("stats_out", "")
    stats_csv = stats_cfg.get("stats_csv", "")

    if not stats_out:
        stats_out = os.path.join(args.out_dir, "summary.json")
    if not stats_csv:
        stats_csv = os.path.join(args.out_dir, "per_sample.csv")

    # -----------------------------
    # Optional energy rerank
    # -----------------------------
    energy_enabled = bool(energy_cfg.get("enabled", False))
    energy_ckpt = str(energy_cfg.get("energy_ckpt", ""))
    cgcnn_tmp_root = str(energy_cfg.get("cgcnn_tmp_root", "/home/DeepRelax-main/_cgcnn_tmp"))
    energy_device = str(energy_cfg.get("energy_device", "cpu"))
    energy_max_num_nbr = int(energy_cfg.get("energy_max_num_nbr", 12))
    energy_radius = float(energy_cfg.get("energy_radius", 12.0))
    beta_E = float(energy_cfg.get("beta_E", 0.8))
    energy_topM = int(energy_cfg.get("energy_topM", 2))
    energy_only_tiebreak = bool(energy_cfg.get("energy_only_tiebreak", True))
    energy_tiebreak_margin = float(energy_cfg.get("energy_tiebreak_margin", 0.25))

    energy_scorer = None
    tmp_energy_dir = os.path.join(args.out_dir, "_tmp_energy")

    if energy_enabled:
        if not energy_ckpt:
            print("[WARN] energy.enabled=True but energy_ckpt is empty. Energy rerank disabled.")
            energy_enabled = False
        else:
            ensure_dir(tmp_energy_dir)
            energy_scorer = CGCNNEnergyScorer(
                ckpt_path=energy_ckpt,
                tmp_root=cgcnn_tmp_root,
                device=energy_device,
                max_num_nbr=energy_max_num_nbr,
                radius=energy_radius,
            )

    print(f"[INFO] root={args.root}")
    print(f"[INFO] ckpt={args.geom_ckpt}")
    print(f"[INFO] out_dir={args.out_dir}")
    print(f"[INFO] split={split} N={len(ds)} device={device}")
    print("[INFO] candidate pool = final / raw / cellu")
    print(f"[INFO] use_aux_score={use_aux_score}")
    print(f"[INFO] project_cell_post={project_cell_post}")
    print(f"[INFO] keep_refine_steps={keep_refine_steps}")
    print(f"[INFO] energy rerank={'ON' if energy_scorer is not None else 'OFF'}")
    print(f"[INFO] energy_only_tiebreak={'ON' if energy_only_tiebreak else 'OFF'}")

    rows = []

    n_keep = 0
    n_pick_final = 0
    n_pick_raw = 0
    n_pick_cellu = 0
    n_refine_used = 0

    # ========================================================
    # Inference loop
    # ========================================================
    for it, data in enumerate(loader, start=1):
        data = data.to(device)

        sid = safe_first(getattr(data, "cif_id", None), None)
        if not isinstance(sid, str):
            sid = f"sample_{it:07d}"

        z = _get_atomic_numbers_tensor(data)
        z_np = z.detach().cpu().numpy().astype(np.int64)
        n_atoms = int(len(z_np))

        # -----------------------------
        # input cell / frac
        # -----------------------------
        cell_u = None

        if hasattr(data, "cell_u") and data.cell_u is not None:
            cu = data.cell_u
            cell_u = cu.view(3, 3) if cu.numel() == 9 else cu[0]
        elif hasattr(data, "cell") and data.cell is not None:
            cc = data.cell
            cell_u = cc.view(3, 3) if cc.numel() == 9 else cc[0]

        frac_u = None

        if hasattr(data, "pos_frac_u") and data.pos_frac_u is not None:
            frac_u = data.pos_frac_u
        elif hasattr(data, "pos_u_frac") and data.pos_u_frac is not None:
            frac_u = data.pos_u_frac

        if frac_u is None and hasattr(data, "pos_u") and data.pos_u is not None and cell_u is not None:
            try:
                frac_u = cart_to_frac_solve(data.pos_u, cell_u)
            except Exception:
                frac_u = None

        frac_u = ensure_frac_n3_torch(frac_u, n_atoms)

        # -----------------------------
        # save init / dft
        # -----------------------------
        u_path = os.path.join(init_dir, f"{sid}.cif")

        if frac_u is not None and cell_u is not None:
            save_atoms(
                z_np,
                frac_u.detach().cpu().numpy(),
                cell_u.detach().cpu().numpy(),
                u_path,
            )

        cell_r_np = None
        frac_r_np = None

        if hasattr(data, "cell_r") and data.cell_r is not None and data.cell_r.numel() == 9:
            cell_r_np = data.cell_r.view(3, 3).detach().cpu().numpy()

        if hasattr(data, "pos_frac_r") and data.pos_frac_r is not None:
            frac_r_np = data.pos_frac_r.detach().cpu().numpy()

        r_path = os.path.join(dft_dir, f"{sid}.cif")

        if cell_r_np is not None and frac_r_np is not None:
            save_atoms(z_np, frac_r_np, cell_r_np, r_path)

        # -----------------------------
        # lattice type
        # -----------------------------
        if hasattr(data, "lattice_type") and data.lattice_type is not None:
            lt = int(data.lattice_type.view(-1)[0].item())
        else:
            sgu = None

            if hasattr(data, "sg_u") and data.sg_u is not None:
                sgu = int(data.sg_u.view(-1)[0].item())

            if sgu is not None and 1 <= sgu <= 230:
                lt = cs_to_lattice_type(crystal_system_id(sgu))
            else:
                lt = 0

        sg_u = None
        cs_u = None

        if frac_u is not None and cell_u is not None:
            sg_u, cs_u = spglib_sg_cs(
                z_np,
                frac_u.detach().cpu().numpy(),
                cell_u.detach().cpu().numpy(),
                symprec,
                angtol,
            )

        # -----------------------------
        # model forward
        # -----------------------------
        with torch.no_grad():
            out = geom_model(data, y_change_for_aux=None)

        if "cell_pred" not in out or out["cell_pred"] is None:
            raise KeyError(f"{sid}: missing output cell_pred")

        cell_final = out["cell_pred"][0].detach()

        if project_cell_post:
            cell_final = project_cell_to_lattice_type(
                cell_final,
                lt,
                preserve_volume=True,
            )

        frac_final = None

        if ("frac_keep" in out) and (out["frac_keep"] is not None):
            frac_final = out["frac_keep"].detach()
        elif ("pos_keep" in out) and (out["pos_keep"] is not None):
            frac_final = cart_to_frac_solve(out["pos_keep"].detach(), cell_final)
        elif ("frac_same" in out) and (out["frac_same"] is not None):
            frac_final = out["frac_same"].detach()
        elif ("pos_same" in out) and (out["pos_same"] is not None):
            frac_final = cart_to_frac_solve(out["pos_same"].detach(), cell_final)

        frac_final = ensure_frac_n3_torch(frac_final, n_atoms)

        if frac_final is None:
            raise RuntimeError(f"{sid}: cannot recover final fractional coordinates")

        if frac_u is not None and keep_refine_steps > 0:
            try:
                frac_final = keep_refine_final_frac(
                    frac_init=frac_final,
                    frac_u=frac_u.detach(),
                    final_cell=cell_final.detach(),
                    data=data,
                    short_cut=short_cut,
                    steps=keep_refine_steps,
                    lr=keep_refine_lr,
                    w_anchor=keep_refine_w_anchor,
                    w_short=keep_refine_w_short,
                )
                n_refine_used += 1
            except Exception:
                pass

        # -----------------------------
        # coarse info
        # -----------------------------
        cell_coarse = None
        frac_coarse = None

        if ("cell_pred_0" in out) and (out["cell_pred_0"] is not None):
            cell_coarse = out["cell_pred_0"][0].detach()

        if ("frac_keep_0" in out) and (out["frac_keep_0"] is not None):
            frac_coarse = out["frac_keep_0"].detach()
        elif ("pos_keep_0" in out) and (out["pos_keep_0"] is not None) and (cell_coarse is not None):
            frac_coarse = cart_to_frac_solve(out["pos_keep_0"].detach(), cell_coarse)

        frac_coarse = ensure_frac_n3_torch(frac_coarse, n_atoms)

        # -----------------------------
        # candidate pool
        # -----------------------------
        keep_cands: List[KeepCandidate] = []

        keep_cands.append(
            KeepCandidate(
                name="final",
                cell=cell_final.detach().clone(),
                frac=frac_final.detach().clone(),
                meta={"source": "frac_keep + cell_pred"},
            )
        )

        if frac_u is not None:
            keep_cands.append(
                KeepCandidate(
                    name="raw",
                    cell=cell_final.detach().clone(),
                    frac=frac_u.detach().clone(),
                    meta={"source": "frac_u + cell_pred"},
                )
            )

        if (frac_u is not None) and (cell_u is not None):
            keep_cands.append(
                KeepCandidate(
                    name="cellu",
                    cell=cell_u.detach().clone(),
                    frac=frac_u.detach().clone(),
                    meta={"source": "frac_u + cell_u"},
                )
            )

        # -----------------------------
        # score candidates
        # -----------------------------
        for c in keep_cands:
            pos_c = torch.matmul(c.frac.float(), c.cell.float())

            _, detail_fast = score_structure_fast(
                data,
                pos_c,
                c.cell,
                use_aux=use_aux_score,
                aux_mu=out.get("aux_mu", None) if c.name != "cellu" else None,
                aux_raw_var=out.get("aux_raw_var", None) if c.name != "cellu" else None,
                short_cut=short_cut,
            )

            c.min_d = float(detail_fast.get("min_d", 999.0))
            c.frac_short = float(detail_fast.get("frac_short", 0.0))
            c.p_geom = geom_prob_from_min_d(
                c.min_d,
                d0=geom_d0,
                k=geom_k,
            )

            try:
                sgk, csk = spglib_sg_cs(
                    z_np,
                    c.frac.detach().cpu().numpy(),
                    c.cell.detach().cpu().numpy(),
                    symprec,
                    angtol,
                )
                c.sg_spglib = sgk
                c.cs_spglib = csk
            except Exception:
                c.sg_spglib = None
                c.cs_spglib = None

            score = safe_log(c.p_geom)

            if c.name == "final":
                score += keep_final_bonus
            elif c.name == "raw":
                score -= keep_raw_penalty
            elif c.name == "cellu":
                score += keep_cellu_bonus

            if (cs_u is not None) and (c.cs_spglib is not None) and (int(c.cs_spglib) != int(cs_u)):
                score -= keep_cs_drop_penalty

            if c.min_d < min_d_thr_keep:
                score -= 100.0

            c.score_total = float(score)

        keep_cands.sort(key=lambda x: x.score_total, reverse=True)

        # -----------------------------
        # optional energy rerank / tiebreak
        # -----------------------------
        if energy_scorer is not None and len(keep_cands) > 0:
            base_best = keep_cands[0].score_total
            subset = keep_cands[: max(1, int(energy_topM))]

            if energy_only_tiebreak:
                subset = [
                    c for c in subset
                    if (base_best - c.score_total) <= float(energy_tiebreak_margin)
                ]
                if len(subset) == 0:
                    subset = [keep_cands[0]]

            finite_subset = []

            for i_c, c in enumerate(subset):
                cif_path_tmp = os.path.join(tmp_energy_dir, f"{sid}_cand{i_c}.cif")

                save_atoms(
                    z_np,
                    c.frac.detach().cpu().numpy(),
                    c.cell.detach().cpu().numpy(),
                    cif_path_tmp,
                )

                try:
                    c.energy = float(energy_scorer.predict_e_per_atom(cif_path_tmp))
                    finite_subset.append(c.energy)
                except Exception as e:
                    print(f"[WARN] energy scorer failed on {sid} cand={c.name}: {repr(e)}")
                    c.energy = float("inf")

            Emin = min(finite_subset) if len(finite_subset) > 0 else float("inf")

            if np.isfinite(Emin):
                for c in subset:
                    if np.isfinite(c.energy):
                        dE = float(c.energy - Emin)
                        c.score_total = float(c.score_total - float(beta_E) * dE)

                keep_cands.sort(key=lambda x: x.score_total, reverse=True)

        best = keep_cands[0]

        if best.name == "final":
            n_pick_final += 1
        elif best.name == "raw":
            n_pick_raw += 1
        elif best.name == "cellu":
            n_pick_cellu += 1

        n_keep += 1

        # -----------------------------
        # save pred
        # -----------------------------
        p_path = os.path.join(pred_dir, f"{sid}.cif")

        save_atoms(
            z_np,
            best.frac.detach().cpu().numpy(),
            best.cell.detach().cpu().numpy(),
            p_path,
        )

        coarse_info = ""

        if (cell_coarse is not None) and (frac_coarse is not None):
            try:
                sgc, csc = spglib_sg_cs(
                    z_np,
                    frac_coarse.detach().cpu().numpy(),
                    cell_coarse.detach().cpu().numpy(),
                    symprec,
                    angtol,
                )
                coarse_info = f" coarse_sg={sgc} coarse_cs={csc}"
            except Exception:
                coarse_info = ""

        print(
            f"[{it}/{len(ds)}] {sid} "
            f"picked={best.name} "
            f"score={best.score_total:.3f} "
            f"E={best.energy if np.isfinite(best.energy) else float('nan'):.4f} "
            f"min_d={best.min_d:.3f} "
            f"p_geom={best.p_geom:.3f} "
            f"sg_u={sg_u} sg_best={best.sg_spglib} "
            f"cs_u={cs_u} cs_best={best.cs_spglib}"
            f"{coarse_info} -> {p_path}"
        )

        if write_stats:
            row = {
                "cif_id": sid,
                "picked_keep_kind": best.name,
                "score_keep": best.score_total,
                "energy_keep": best.energy if np.isfinite(best.energy) else "",
                "min_d_keep": best.min_d,
                "p_geom_keep": best.p_geom,
                "frac_short_keep": best.frac_short,
                "sg_u": sg_u if sg_u is not None else "",
                "sg_keep": best.sg_spglib if best.sg_spglib is not None else "",
                "cs_u": cs_u if cs_u is not None else "",
                "cs_keep": best.cs_spglib if best.cs_spglib is not None else "",
                "init_cif": u_path,
                "pred_cif": p_path,
                "dft_cif": r_path if (cell_r_np is not None and frac_r_np is not None) else "",
            }

            if (cell_coarse is not None) and (frac_coarse is not None):
                try:
                    pos_coarse = torch.matmul(frac_coarse.float(), cell_coarse.float())

                    _, detail_coarse = score_structure_fast(
                        data,
                        pos_coarse,
                        cell_coarse,
                        use_aux=False,
                        short_cut=short_cut,
                    )

                    sgc, csc = spglib_sg_cs(
                        z_np,
                        frac_coarse.detach().cpu().numpy(),
                        cell_coarse.detach().cpu().numpy(),
                        symprec,
                        angtol,
                    )

                    row["min_d_coarse"] = detail_coarse.get("min_d", "")
                    row["frac_short_coarse"] = detail_coarse.get("frac_short", "")
                    row["sg_coarse"] = sgc if sgc is not None else ""
                    row["cs_coarse"] = csc if csc is not None else ""

                except Exception:
                    row["min_d_coarse"] = ""
                    row["frac_short_coarse"] = ""
                    row["sg_coarse"] = ""
                    row["cs_coarse"] = ""
            else:
                row["min_d_coarse"] = ""
                row["frac_short_coarse"] = ""
                row["sg_coarse"] = ""
                row["cs_coarse"] = ""

            rows.append(row)

    # ========================================================
    # Summary
    # ========================================================
    if write_stats:
        summary = {
            "split": split,
            "n_samples": len(ds),
            "counts": {
                "final_keep": n_keep,
                "pick_final": n_pick_final,
                "pick_raw": n_pick_raw,
                "pick_cellu": n_pick_cellu,
                "keep_refine_used": n_refine_used,
            },
            "params": {
                "symprec": symprec,
                "angtol": angtol,
                "short_cut": short_cut,
                "min_d_thr_keep": min_d_thr_keep,
                "geom_d0": geom_d0,
                "geom_k": geom_k,
                "project_cell_post": project_cell_post,
                "keep_final_bonus": keep_final_bonus,
                "keep_raw_penalty": keep_raw_penalty,
                "keep_cellu_bonus": keep_cellu_bonus,
                "keep_cs_drop_penalty": keep_cs_drop_penalty,
                "keep_refine_steps": keep_refine_steps,
                "keep_refine_lr": keep_refine_lr,
                "keep_refine_w_anchor": keep_refine_w_anchor,
                "keep_refine_w_short": keep_refine_w_short,
                "use_aux_score": use_aux_score,

                "energy_enabled": bool(energy_scorer is not None),
                "energy_ckpt": energy_ckpt,
                "cgcnn_tmp_root": cgcnn_tmp_root,
                "energy_device": energy_device,
                "energy_max_num_nbr": energy_max_num_nbr,
                "energy_radius": energy_radius,
                "beta_E": beta_E,
                "energy_topM": energy_topM,
                "energy_only_tiebreak": energy_only_tiebreak,
                "energy_tiebreak_margin": energy_tiebreak_margin,
            },
        }

        print("\n========== SUMMARY ==========")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        if stats_out:
            with open(stats_out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"wrote stats json -> {stats_out}")

        if stats_csv:
            keys = list(rows[0].keys()) if rows else []

            with open(stats_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()

                for r in rows:
                    w.writerow(r)

            print(f"wrote per-sample csv -> {stats_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
