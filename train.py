#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_scatter import scatter
from torch.amp import autocast

from crystal_data import CrystalDataset, make_batch
from SymmRelax import (
    DeepRelaxSymmEnergy,
    build_edge_distances_pbc,
)


# ============================================================
# Config
# ============================================================

def resolve_config_path(path: str) -> str:
    p = Path(path)

    if p.is_file():
        return str(p)

    script_dir = Path(__file__).resolve().parent
    p2 = script_dir / path

    if p2.is_file():
        return str(p2)

    raise FileNotFoundError(f"Config file not found: {path}")


def load_json_config(path: str):
    path = resolve_config_path(path)

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    model_cfg["use_aux_dist"] = bool(train_cfg.get("use_aux_dist", model_cfg.get("use_aux_dist", False)))

    return train_cfg, model_cfg


class TrainState:
    pass


def build_epoch_cfg(ep: int, epochs1: int, train_cfg: dict):
    cfg = TrainState()

    cfg.aux_var_eps = float(train_cfg["aux_var_eps"])
    cfg.q_sign_on_changed_only = bool(train_cfg["q_sign_on_changed_only"])

    if ep <= epochs1:
        w_pos_chg = float(train_cfg["stage1_w_pos"])
    else:
        warm_n = max(1, int(train_cfg["pos_warmup_epochs"]))
        t = min(1.0, float(ep - epochs1) / float(warm_n))
        w_pos_chg = float(train_cfg["stage1_w_pos"]) + t * (
            float(train_cfg["w_pos"]) - float(train_cfg["stage1_w_pos"])
        )

    if ep < int(train_cfg["ct_reg_start_epoch"]):
        ct_scale = 0.0
    else:
        warm = max(1, int(train_cfg["ct_reg_warmup_epochs"]))
        ct_scale = min(
            1.0,
            float(ep - int(train_cfg["ct_reg_start_epoch"]) + 1) / float(warm),
        )

    cfg.w_rep_keep = float(train_cfg["w_rep_keep"])
    cfg.w_frac_keep = float(train_cfg["w_frac_keep"])
    cfg.w_pos_keep = float(train_cfg["w_pos_keep"])
    cfg.w_cell_keep = float(train_cfg["w_cell_keep"])
    cfg.w_dist_keep = float(train_cfg["w_dist_keep"])

    cfg.w_rep_chg = float(train_cfg["w_rep"])
    cfg.w_frac_chg = float(train_cfg["w_frac"])
    cfg.w_pos_chg = float(w_pos_chg)
    cfg.w_cell_chg = float(train_cfg["w_cell"])
    cfg.w_dist_chg = float(train_cfg["w_dist"])

    cfg.w_dist_any = max(cfg.w_dist_keep, cfg.w_dist_chg)

    cfg.w_coarse = float(train_cfg["w_coarse"])

    cfg.w_neutral = float(train_cfg["w_neutral"]) * ct_scale
    cfg.w_q_sign = float(train_cfg["w_q_sign"]) * ct_scale
    cfg.w_q_mm = float(train_cfg["w_q_mm"]) * ct_scale

    cfg.ct_scale = ct_scale

    return cfg


# ============================================================
# DDP helpers
# ============================================================

def ddp_is_on() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def ddp_barrier():
    if ddp_is_on() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def ddp_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if ddp_is_on() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


def ddp_all_reduce_max(t: torch.Tensor) -> torch.Tensor:
    if ddp_is_on() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return t


def ddp_global_bad_flag(local_bad: bool, device: torch.device) -> bool:
    flag = torch.tensor([1 if local_bad else 0], device=device, dtype=torch.int32)
    ddp_all_reduce_max(flag)
    return bool(flag.item())


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def model_has_nonfinite_grads(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


def _get_pos_r(data):
    if hasattr(data, "pos_r") and data.pos_r is not None:
        return data.pos_r

    if (
        hasattr(data, "pos_frac_r")
        and hasattr(data, "cell_r")
        and data.pos_frac_r is not None
        and data.cell_r is not None
    ):
        atom_batch = getattr(data, "atom_batch", data.batch).long()
        cell_atom = data.cell_r[atom_batch]
        return torch.bmm(data.pos_frac_r.unsqueeze(1), cell_atom).squeeze(1)

    raise AttributeError("Need data.pos_r OR pos_frac_r + cell_r")


def _safe_cell_r(data):
    if hasattr(data, "cell_r") and data.cell_r is not None:
        return data.cell_r
    return data.cell_u


def _wyc_batch_from_atoms(atom_wyc_global: torch.Tensor, atom_batch: torch.Tensor) -> torch.Tensor:
    if atom_wyc_global is None or atom_wyc_global.numel() == 0:
        return atom_batch.new_zeros((0,), dtype=torch.long)

    n_w = int(atom_wyc_global.max().item()) + 1

    if n_w <= 0:
        return atom_batch.new_zeros((0,), dtype=torch.long)

    return scatter(
        atom_batch.long(),
        atom_wyc_global.long(),
        dim=0,
        dim_size=n_w,
        reduce="min",
    )


def _scatter_mean_safe(
    val: torch.Tensor,
    idx: torch.Tensor,
    dim_size: int,
    device: torch.device,
) -> torch.Tensor:
    if val.numel() == 0:
        return torch.zeros((dim_size,), device=device)

    return scatter(val, idx, dim=0, dim_size=dim_size, reduce="mean")


def _as_graph_level(
    y,
    atom_batch: torch.Tensor,
    B: int,
    device: torch.device,
    reduce: str = "max",
):
    if y is None:
        return None

    y = torch.as_tensor(y, device=device).float().view(-1)

    if y.numel() == B:
        return y

    if y.numel() == atom_batch.numel():
        return scatter(y, atom_batch.long(), dim=0, dim_size=B, reduce=reduce)

    if y.numel() > B:
        return y[:B]

    out = torch.zeros((B,), device=device, dtype=y.dtype)
    out[: y.numel()] = y
    return out


def _ensure_graph_y_change(data, B: int, device: torch.device) -> torch.Tensor:
    y = getattr(data, "y_change", None)

    if y is None:
        return torch.zeros((B,), device=device)

    y = y.float().view(-1).to(device)

    if y.numel() == B:
        return y

    atom_batch = getattr(data, "atom_batch", data.batch).long().to(device)

    if y.numel() == atom_batch.numel():
        return scatter(y, atom_batch, dim=0, dim_size=B, reduce="max")

    if y.numel() > B:
        return y[:B]

    return torch.zeros((B,), device=device)


def cart_to_frac_solve(pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    frac = torch.linalg.solve(cell.T.double(), pos.T.double()).T.float()
    return frac - torch.floor(frac)


def wrapped_frac_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return d - torch.round(d)


def graph_frac_loss_per_graph(
    frac_pred: torch.Tensor,
    frac_tgt: torch.Tensor,
    batch_idx: torch.Tensor,
    B: int,
) -> torch.Tensor:
    dfrac = wrapped_frac_diff(frac_pred, frac_tgt)
    per_atom = dfrac.abs().mean(dim=-1)
    return scatter(per_atom, batch_idx, dim=0, dim_size=B, reduce="mean")


def graph_pbc_aligned_pos_loss_per_graph(
    pos_pred_cart: torch.Tensor,
    pos_tgt_cart: torch.Tensor,
    cell_ref: torch.Tensor,
    batch_idx: torch.Tensor,
    B: int,
) -> torch.Tensor:
    out = torch.zeros((B,), device=pos_pred_cart.device)

    for g in range(B):
        mask = batch_idx == g

        if mask.sum() < 1:
            continue

        try:
            frac_pred = cart_to_frac_solve(pos_pred_cart[mask], cell_ref[g])
            frac_tgt = cart_to_frac_solve(pos_tgt_cart[mask], cell_ref[g])
            dfrac = wrapped_frac_diff(frac_pred, frac_tgt)
            dcart = torch.matmul(dfrac, cell_ref[g])
            out[g] = torch.linalg.norm(dcart, dim=-1).mean()
        except torch._C._LinAlgError:
            out[g] = 0.0

    return out


def gaussian_nll_1d_per_edge(
    mu: torch.Tensor,
    raw_var: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    var = F.softplus(raw_var) + eps
    return 0.5 * (((target - mu) ** 2) / var + torch.log(var))


# ============================================================
# Loss
# ============================================================

def compute_graph_losses(
    out: dict,
    data,
    y_change: torch.Tensor,
    aux_var_eps: float,
    use_dist: bool,
    q_sign_on_changed_only: bool = False,
):
    device = y_change.device

    batch_nodes = data.batch.long()
    atom_batch = getattr(data, "atom_batch", batch_nodes).long()

    B = int(atom_batch.max().item()) + 1 if atom_batch.numel() else int(y_change.numel())

    pos_r = _get_pos_r(data)
    cell_r = _safe_cell_r(data)

    if not hasattr(data, "pos_frac_r") or data.pos_frac_r is None:
        raise RuntimeError("Expected data.pos_frac_r.")

    pos_frac_r = data.pos_frac_r

    wyc_batch_u = _wyc_batch_from_atoms(
        getattr(data, "atom_wyc_global_u", None),
        atom_batch,
    )

    rep_tgt_u = data.wyc_rep_target_u

    rep_keep = out["rep_frac_keep"]
    frac_keep = out["frac_keep"]
    pos_keep = out["pos_keep"]

    rep_keep_0 = out["rep_frac_keep_0"]
    frac_keep_0 = out["frac_keep_0"]
    pos_keep_0 = out["pos_keep_0"]

    graph_rep = torch.zeros((B,), device=device)
    graph_rep0 = torch.zeros((B,), device=device)

    if wyc_batch_u.numel():
        err_rep = wrapped_frac_diff(rep_keep, rep_tgt_u).abs().mean(dim=-1)
        err_rep0 = wrapped_frac_diff(rep_keep_0, rep_tgt_u).abs().mean(dim=-1)

        graph_rep = _scatter_mean_safe(err_rep, wyc_batch_u, B, device)
        graph_rep0 = _scatter_mean_safe(err_rep0, wyc_batch_u, B, device)

    graph_frac = graph_frac_loss_per_graph(frac_keep, pos_frac_r, atom_batch, B)
    graph_frac0 = graph_frac_loss_per_graph(frac_keep_0, pos_frac_r, atom_batch, B)

    graph_pos = graph_pbc_aligned_pos_loss_per_graph(
        pos_pred_cart=pos_keep,
        pos_tgt_cart=pos_r,
        cell_ref=cell_r,
        batch_idx=atom_batch,
        B=B,
    )

    graph_pos0 = graph_pbc_aligned_pos_loss_per_graph(
        pos_pred_cart=pos_keep_0,
        pos_tgt_cart=pos_r,
        cell_ref=cell_r,
        batch_idx=atom_batch,
        B=B,
    )

    if hasattr(data, "cell_param_r") and data.cell_param_r is not None:
        graph_cell = (out["cell_param_pred"] - data.cell_param_r).abs().mean(dim=-1)
        graph_cell0 = (out["cell_param_pred_0"] - data.cell_param_r).abs().mean(dim=-1)
    else:
        graph_cell = (out["cell_pred"] - cell_r).abs().view(B, -1).mean(dim=-1)
        graph_cell0 = (out["cell_pred_0"] - cell_r).abs().view(B, -1).mean(dim=-1)

    graph_dist = torch.zeros((B,), device=device)

    if use_dist and ("aux_mu" in out) and (out["aux_mu"] is not None):
        dist_t = build_edge_distances_pbc(
            pos=pos_r,
            cell=cell_r,
            cell_offsets=data.cell_offsets,
            edge_index=data.edge_index,
            neighbors=data.neighbors,
        )

        per_edge = gaussian_nll_1d_per_edge(
            out["aux_mu"],
            out["aux_raw_var"],
            dist_t,
            eps=aux_var_eps,
        )

        _, i = data.edge_index.long()
        edge_graph = atom_batch[i]

        graph_dist = _scatter_mean_safe(per_edge, edge_graph, B, device)

    graph_neutral = torch.zeros((B,), device=device)
    graph_q_sign = torch.zeros((B,), device=device)
    graph_q_mm = torch.zeros((B,), device=device)

    if ("q_pred" in out) and (out["q_pred"] is not None):
        q = out["q_pred"].view(-1)

        if (
            "q_sum_per_graph" in out
            and torch.is_tensor(out["q_sum_per_graph"])
            and out["q_sum_per_graph"].numel() == B
        ):
            qsum = out["q_sum_per_graph"].to(device=device, dtype=q.dtype)
        else:
            qsum = scatter(q, atom_batch, dim=0, dim_size=B, reduce="sum")

        graph_neutral = qsum.pow(2)

        if ("chi_atom" in out) and (out["chi_atom"] is not None):
            chi = out["chi_atom"].view(-1).to(device=device, dtype=q.dtype)

            j, i = data.edge_index.long()
            score = (q[i] - q[j]) * (chi[j] - chi[i])
            pen = F.relu(-score)

            edge_graph = atom_batch[i]
            graph_q_sign = _scatter_mean_safe(pen, edge_graph, B, device)

            if q_sign_on_changed_only:
                graph_q_sign = graph_q_sign * (y_change > 0.5).float()

        if hasattr(data, "is_metal_metal_graph") and data.is_metal_metal_graph is not None:
            mm_graph = _as_graph_level(
                data.is_metal_metal_graph,
                atom_batch,
                B,
                device,
                reduce="max",
            )

            q2_graph = scatter(q * q, atom_batch, dim=0, dim_size=B, reduce="mean")
            graph_q_mm = q2_graph * (mm_graph > 0.5).float()

    return {
        "rep": graph_rep,
        "frac": graph_frac,
        "pos": graph_pos,
        "cell": graph_cell,

        "rep0": graph_rep0,
        "frac0": graph_frac0,
        "pos0": graph_pos0,
        "cell0": graph_cell0,

        "dist": graph_dist,
        "neutral": graph_neutral,
        "q_sign": graph_q_sign,
        "q_mm": graph_q_mm,
    }


def build_total_graph_loss(losses: dict, y_change: torch.Tensor, cfg, device):
    B = int(y_change.numel())

    keep = y_change <= 0.5
    chg = y_change > 0.5

    def _two_bucket_weights(w_keep, w_chg):
        w = torch.zeros((B,), device=device)
        w[keep] = float(w_keep)
        w[chg] = float(w_chg)
        return w

    w_rep = _two_bucket_weights(cfg.w_rep_keep, cfg.w_rep_chg)
    w_frac = _two_bucket_weights(cfg.w_frac_keep, cfg.w_frac_chg)
    w_pos = _two_bucket_weights(cfg.w_pos_keep, cfg.w_pos_chg)
    w_cell = _two_bucket_weights(cfg.w_cell_keep, cfg.w_cell_chg)
    w_dist = _two_bucket_weights(cfg.w_dist_keep, cfg.w_dist_chg)

    g_total = (
        w_rep * losses["rep"]
        + w_frac * losses["frac"]
        + w_pos * losses["pos"]
        + w_cell * losses["cell"]
        + cfg.w_coarse
        * (
            w_rep * losses["rep0"]
            + w_frac * losses["frac0"]
            + w_pos * losses["pos0"]
            + w_cell * losses["cell0"]
        )
        + w_dist * losses["dist"]
        + cfg.w_neutral * losses["neutral"]
        + cfg.w_q_sign * losses["q_sign"]
        + cfg.w_q_mm * losses["q_mm"]
    )

    return g_total


# ============================================================
# Meters
# ============================================================

def make_meter(device):
    keys = [
        "loss",
        "rep",
        "frac",
        "pos",
        "cell",
        "rep0",
        "frac0",
        "pos0",
        "cell0",
        "dist",
        "neutral",
        "q_sign",
        "q_mm",
        "q_abs",
        "q_neutral_l1",
        "gate_sat",
        "steps",
        "skipped",
    ]

    return {k: torch.tensor(0.0, device=device) for k in keys}


def update_meter(meter, loss, losses, out):
    meter["loss"] += loss.detach()

    for k in [
        "rep",
        "frac",
        "pos",
        "cell",
        "rep0",
        "frac0",
        "pos0",
        "cell0",
        "dist",
        "neutral",
        "q_sign",
        "q_mm",
    ]:
        meter[k] += losses[k].mean().detach()

    if "ct_q_abs_mean" in out:
        meter["q_abs"] += out["ct_q_abs_mean"].detach()

    if "q_neutral_l1" in out:
        meter["q_neutral_l1"] += out["q_neutral_l1"].detach()

    if "ct_gate_saturated_frac" in out:
        meter["gate_sat"] += out["ct_gate_saturated_frac"].detach()

    meter["steps"] += 1.0


def reduce_meter(meter):
    for v in meter.values():
        ddp_all_reduce_sum(v)

    denom = torch.clamp(meter["steps"], min=1.0)

    return {
        "loss": float((meter["loss"] / denom).item()),
        "rep": float((meter["rep"] / denom).item()),
        "frac": float((meter["frac"] / denom).item()),
        "pos": float((meter["pos"] / denom).item()),
        "cell": float((meter["cell"] / denom).item()),
        "rep0": float((meter["rep0"] / denom).item()),
        "frac0": float((meter["frac0"] / denom).item()),
        "pos0": float((meter["pos0"] / denom).item()),
        "cell0": float((meter["cell0"] / denom).item()),
        "dist": float((meter["dist"] / denom).item()),
        "neutral": float((meter["neutral"] / denom).item()),
        "q_sign": float((meter["q_sign"] / denom).item()),
        "q_mm": float((meter["q_mm"] / denom).item()),
        "q_abs": float((meter["q_abs"] / denom).item()),
        "q_neutral_l1": float((meter["q_neutral_l1"] / denom).item()),
        "gate_sat": float((meter["gate_sat"] / denom).item()),
        "steps": float(meter["steps"].item()),
        "skipped": float(meter["skipped"].item()),
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()

    meter = make_meter(device)

    for data in loader:
        data = data.to(device, non_blocking=True)

        B = int(data.batch.max().item()) + 1
        y_change = _ensure_graph_y_change(data, B, device)

        out = model(
            data,
            y_change_for_aux=y_change if cfg.w_dist_any > 0 else None,
        )

        local_bad = False
        for v in out.values():
            if torch.is_tensor(v) and not torch.isfinite(v).all():
                local_bad = True
                break

        if ddp_global_bad_flag(local_bad, device):
            meter["skipped"] += 1.0
            continue

        losses = compute_graph_losses(
            out=out,
            data=data,
            y_change=y_change,
            aux_var_eps=cfg.aux_var_eps,
            use_dist=(cfg.w_dist_any > 0),
            q_sign_on_changed_only=cfg.q_sign_on_changed_only,
        )

        loss = build_total_graph_loss(losses, y_change, cfg, device).mean()

        local_bad_loss = not is_finite_tensor(loss)

        if ddp_global_bad_flag(local_bad_loss, device):
            meter["skipped"] += 1.0
            continue

        update_meter(meter, loss, losses, out)

    log = reduce_meter(meter)

    model.train()
    return log


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, default="ckpt_symmrelax.pt")
    ap.add_argument("--config", type=str, default="train_config.json")

    ap.add_argument("--steps_per_epoch", type=int, default=800)
    ap.add_argument("--epochs1", type=int, default=50)
    ap.add_argument("--epochs2", type=int, default=70)

    args = ap.parse_args()

    train_cfg, model_cfg = load_json_config(args.config)

    train_cfg["steps_per_epoch"] = int(args.steps_per_epoch)
    train_cfg["epochs1"] = int(args.epochs1)
    train_cfg["epochs2"] = int(args.epochs2)

    local_rank = 0

    if ddp_is_on():
        local_rank = ddp_local_rank()

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"

        torch.distributed.init_process_group(backend=backend)

    set_seed(int(train_cfg["seed"]) + ddp_rank())

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    if ddp_rank() == 0:
        print(f"root: {args.root}")
        print(f"out : {args.out}")
        print(f"config: {resolve_config_path(args.config)}")
        print(
            f"epochs: {train_cfg['epochs1']} + {train_cfg['epochs2']} | "
            f"steps/epoch: {train_cfg['steps_per_epoch']} | "
            f"batch: {train_cfg['batch_size']}"
        )

    train_set = CrystalDataset(args.root, split="train")
    val_set = CrystalDataset(args.root, split="val")

    if ddp_is_on():
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=ddp_world_size(),
            rank=ddp_rank(),
            shuffle=True,
            drop_last=False,
        )

        val_sampler = DistributedSampler(
            val_set,
            num_replicas=ddp_world_size(),
            rank=ddp_rank(),
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=make_batch,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(int(train_cfg["num_workers"]) > 0),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        collate_fn=make_batch,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(int(train_cfg["num_workers"]) > 0),
    )

    base_model = DeepRelaxSymmEnergy(**model_cfg).to(device)

    if ddp_is_on():
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )
    else:
        model = base_model

    opt = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=0.0,
        amsgrad=True,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.8,
        patience=5,
        min_lr=1e-8,
    )

    total_epochs = int(train_cfg["epochs1"]) + int(train_cfg["epochs2"])
    epochs1 = int(train_cfg["epochs1"])

    best_val = float("inf")

    last_out = (
        args.out.replace(".pt", "_last.pt")
        if args.out.endswith(".pt")
        else args.out + "_last.pt"
    )

    amp_enabled = (
        bool(train_cfg["amp"])
        and torch.cuda.is_available()
    )

    amp_dtype = torch.bfloat16 if train_cfg["amp_dtype"] == "bf16" else torch.float16

    stage2_stable_start_ep = (
        epochs1 + int(train_cfg["pos_warmup_epochs"]) + 1
    )

    reset_done = False

    for ep in range(1, total_epochs + 1):
        if ddp_is_on() and train_sampler is not None:
            train_sampler.set_epoch(ep)

        if (not reset_done) and ep == epochs1 + 1:
            best_val = float("inf")
            reset_done = True
            ddp_barrier()

        cfg = build_epoch_cfg(ep, epochs1, train_cfg)

        model.train()
        meter = make_meter(device)

        for step, data in enumerate(train_loader, start=1):
            data = data.to(device, non_blocking=True)

            B = int(data.batch.max().item()) + 1
            y_change = _ensure_graph_y_change(data, B, device)

            opt.zero_grad(set_to_none=True)

            with autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                out = model(
                    data,
                    y_change_for_aux=y_change if cfg.w_dist_any > 0 else None,
                )

                local_bad_forward = False
                for v in out.values():
                    if torch.is_tensor(v) and not torch.isfinite(v).all():
                        local_bad_forward = True
                        break

                if ddp_global_bad_flag(local_bad_forward, device):
                    meter["skipped"] += 1.0
                    continue

                losses = compute_graph_losses(
                    out=out,
                    data=data,
                    y_change=y_change,
                    aux_var_eps=cfg.aux_var_eps,
                    use_dist=(cfg.w_dist_any > 0),
                    q_sign_on_changed_only=cfg.q_sign_on_changed_only,
                )

                local_bad_loss_comp = False
                for v in losses.values():
                    if torch.is_tensor(v) and not torch.isfinite(v).all():
                        local_bad_loss_comp = True
                        break

                g_total = build_total_graph_loss(losses, y_change, cfg, device)
                loss = g_total.mean()

                local_bad_loss = (
                    local_bad_loss_comp
                    or not is_finite_tensor(loss)
                    or (
                        float(train_cfg["loss_cap"]) > 0
                        and float(loss.detach().item()) > float(train_cfg["loss_cap"])
                    )
                )

                if ddp_global_bad_flag(local_bad_loss, device):
                    meter["skipped"] += 1.0
                    continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(train_cfg["max_norm"]),
            )

            local_bad_grad = model_has_nonfinite_grads(model)

            if ddp_global_bad_flag(local_bad_grad, device):
                opt.zero_grad(set_to_none=True)
                meter["skipped"] += 1.0
                continue

            opt.step()

            update_meter(meter, loss, losses, out)

            if step >= int(train_cfg["steps_per_epoch"]):
                break

        train_log = reduce_meter(meter)
        val_log = evaluate(model, val_loader, device, cfg)

        val_loss = val_log["loss"]

        in_stage2_warmup = (
            ep > epochs1
            and ep < stage2_stable_start_ep
        )

        if not in_stage2_warmup and np.isfinite(val_loss):
            scheduler.step(val_loss)

        allow_best = (
            ep <= epochs1
            or ep >= stage2_stable_start_ep
        )

        if ddp_rank() == 0:
            lr = opt.param_groups[0]["lr"]

            print(
                f"ep {ep:03d}/{total_epochs} | "
                f"train {train_log['loss']:.5f} | "
                f"val {val_log['loss']:.5f} | "
                f"pos {val_log['pos']:.4f} | "
                f"cell {val_log['cell']:.4f} | "
                f"frac {val_log['frac']:.4f} | "
                f"rep {val_log['rep']:.4f} | "
                f"q {val_log['q_abs']:.3f} | "
                f"skip {train_log['skipped']:.0f}/{val_log['skipped']:.0f} | "
                f"lr {lr:.2e}"
            )

        if ddp_rank() == 0 and allow_best and np.isfinite(val_loss):
            if (best_val - val_loss) >= float(train_cfg["early_stop_min_delta"]):
                best_val = val_loss

                sd = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )

                torch.save(
                    {
                        "model": sd,
                        "best_val": best_val,
                        "epoch": ep,
                        "train_config": train_cfg,
                        "model_config": model_cfg,
                    },
                    args.out,
                )

                print(f"saved best: {args.out} | best_val={best_val:.6f}")

        ddp_barrier()

    if ddp_rank() == 0:
        sd = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        torch.save(
            {
                "model": sd,
                "best_val": best_val,
                "epoch": total_epochs,
                "train_config": train_cfg,
                "model_config": model_cfg,
            },
            last_out,
        )

        print(f"saved last: {last_out}")
        print("Training completed.")

    if ddp_is_on():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
