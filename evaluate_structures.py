#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import warnings
import multiprocessing
import json
from collections import Counter

# =========================
# Silence warnings early
# =========================
os.environ["PYTHONWARNINGS"] = "ignore"

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module=r"pymatgen\.io\.cif")
warnings.filterwarnings(
    "ignore",
    message=r".*fractional coordinates rounded to ideal values.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Issues encountered while parsing CIF.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*dict interface is deprecated.*",
)

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN


# =========================
# Globals / defaults
# =========================
SYMPREC_SCAN_DEFAULT = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

CPU = multiprocessing.cpu_count()
DEFAULT_JOBS = min(max(1, CPU // 2), 32)

_BOND = None


def get_bond_strategy():
    global _BOND
    if _BOND is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _BOND = CrystalNN()
    return _BOND


matcher = StructureMatcher(
    ltol=0.2,
    stol=0.3,
    angle_tol=5,
    primitive_cell=False,
    scale=True,
    attempt_supercell=False,
)


# =========================
# JSON helpers
# =========================
def json_clean(x):
    if isinstance(x, dict):
        return {str(k): json_clean(v) for k, v in x.items()}

    if isinstance(x, list):
        return [json_clean(v) for v in x]

    if isinstance(x, tuple):
        return [json_clean(v) for v in x]

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        x = float(x)
        if not np.isfinite(x):
            return None
        return x

    if isinstance(x, float):
        if not np.isfinite(x):
            return None
        return x

    if isinstance(x, (np.bool_,)):
        return bool(x)

    return x


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))


def is_nan_like(x):
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


# =========================
# Symmetry helpers
# =========================
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


CS_NAME = {
    0: "triclinic",
    1: "monoclinic",
    2: "orthorhombic",
    3: "tetragonal",
    4: "trigonal",
    5: "hexagonal",
    6: "cubic",
}


def cs_name(cs_id):
    try:
        return CS_NAME.get(int(cs_id), "unknown")
    except Exception:
        return "unknown"


def load_structure(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore", module=r"pymatgen\.io\.cif")
        warnings.filterwarnings(
            "ignore",
            message=r".*fractional coordinates rounded to ideal values.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Issues encountered while parsing CIF.*",
        )
        return Structure.from_file(path)


def standardize(struct):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SpacegroupAnalyzer(
                struct,
                symprec=1e-2,
                angle_tolerance=5,
            ).get_conventional_standard_structure()
    except Exception:
        return struct.copy()


def spacegroup(struct, symprec, angtol=5):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SpacegroupAnalyzer(
                struct,
                symprec=float(symprec),
                angle_tolerance=float(angtol),
            ).get_space_group_number()
    except Exception:
        return None


def symmetry_recovery_tol(pred, ref, symprec_scan):
    sg_ref = spacegroup(ref, 1e-4)

    if sg_ref is None:
        return np.nan

    for sp in symprec_scan:
        try:
            if spacegroup(pred, sp) == sg_ref:
                return float(sp)
        except Exception:
            pass

    return np.nan


# =========================
# Group/subgroup relation
# =========================
def build_sg_to_halls(max_hall: int = 530):
    import spglib

    sg2h = {sg: [] for sg in range(1, 231)}

    for hall in range(1, int(max_hall) + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tp = spglib.get_spacegroup_type(hall)

            sg = int(tp["number"]) if isinstance(tp, dict) else int(tp.number)

            if 1 <= sg <= 230:
                sg2h[sg].append(hall)

        except Exception:
            continue

    return sg2h


def is_group_related(pred_sg: int, ref_sg: int, sg2halls: dict) -> str:
    import spglib

    if pred_sg is None or ref_sg is None:
        return "unrelated"

    try:
        if np.isnan(pred_sg) or np.isnan(ref_sg):
            return "unrelated"
    except Exception:
        pass

    pred_sg = int(pred_sg)
    ref_sg = int(ref_sg)

    if pred_sg == ref_sg:
        return "equal"

    def reachable_super_to_sub(super_sg, sub_sg, max_depth=6):
        super_sg = int(super_sg)
        sub_sg = int(sub_sg)

        if super_sg == sub_sg:
            return True

        seen = {super_sg}
        frontier = [super_sg]
        depth = 0

        while frontier and depth < max_depth:
            nxt = []

            for sg in frontier:
                halls = sg2halls.get(int(sg), [])
                if not halls:
                    continue

                hall = halls[0]

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        subs = spglib.get_maximal_subgroups(hall)
                except Exception:
                    subs = None

                if subs is None:
                    continue

                for h in subs:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            tp = spglib.get_spacegroup_type(int(h))

                        sg2 = int(tp["number"]) if isinstance(tp, dict) else int(tp.number)

                    except Exception:
                        continue

                    if sg2 == sub_sg:
                        return True

                    if sg2 not in seen:
                        seen.add(sg2)
                        nxt.append(sg2)

            frontier = nxt
            depth += 1

        return False

    if reachable_super_to_sub(ref_sg, pred_sg):
        return "pred_sub_ref"

    if reachable_super_to_sub(pred_sg, ref_sg):
        return "pred_sup_ref"

    return "unrelated"


# =========================
# Geometry metrics
# =========================
def lattice_mae(s1, s2):
    diff = np.abs(np.array(s1.lattice.abc) - np.array(s2.lattice.abc))

    return {
        "lat_mae_a": float(diff[0]),
        "lat_mae_b": float(diff[1]),
        "lat_mae_c": float(diff[2]),
        "lat_mae": float(np.mean(diff)),
    }


def volume_mae(s1, s2):
    return float(abs(s1.volume - s2.volume))


def coordinate_rmsd_pbc(aligned_pred, ref):
    df = aligned_pred.frac_coords - ref.frac_coords
    df -= np.round(df)

    cart = df @ ref.lattice.matrix
    return float(np.sqrt(np.mean(np.sum(cart ** 2, axis=1))))


def bond_length_mae(aligned_pred, ref):
    try:
        bond = get_bond_strategy()
        bonded = bond.get_bonded_structure(ref)

        diffs = []

        for i, j in bonded.graph.edges():
            diffs.append(abs(ref.get_distance(i, j) - aligned_pred.get_distance(i, j)))

        return float(np.mean(diffs)) if diffs else np.nan

    except Exception:
        return np.nan


def coordination_accuracy(aligned_pred, ref):
    bond = get_bond_strategy()

    correct = 0
    valid = 0

    for i in range(len(ref)):
        try:
            cn_ref = bond.get_cn(ref, i)
            cn_pred = bond.get_cn(aligned_pred, i)

            valid += 1

            if round(cn_ref) == round(cn_pred):
                correct += 1

        except Exception:
            pass

    return float(correct / valid) if valid > 0 else np.nan


# =========================
# Data table
# =========================
def build_eval_table(
    run_dir,
    pred_name,
    dft_name,
    meta_csv,
    split_npy,
    per_sample_csv="",
):
    pred_dir = os.path.join(run_dir, pred_name)
    dft_dir = os.path.join(run_dir, dft_name)

    meta = pd.read_csv(meta_csv)
    split = set(np.load(split_npy).astype(int).tolist())

    if "global_idx" in meta.columns:
        meta = meta[meta["global_idx"].isin(split)].copy()
    else:
        raise ValueError("meta.csv must contain column 'global_idx'")

    if "atoms_id" in meta.columns:
        meta["cif_id"] = meta["atoms_id"].astype(str)
    elif "cif_id" in meta.columns:
        meta["cif_id"] = meta["cif_id"].astype(str)
    else:
        raise ValueError("meta.csv must contain column 'atoms_id' or 'cif_id'")

    for col in ["y_type", "sg_u", "sg_r", "cs_u", "cs_r"]:
        if col not in meta.columns:
            raise ValueError(f"meta.csv missing required column: {col}")

    meta["gt_mode"] = np.where(meta["y_type"].astype(int) > 0, "CHANGE", "KEEP")

    pred_set = {f[:-4] for f in os.listdir(pred_dir) if f.endswith(".cif")}
    dft_set = {f[:-4] for f in os.listdir(dft_dir) if f.endswith(".cif")}
    avail = pred_set & dft_set

    meta = meta[meta["cif_id"].isin(avail)].copy()

    if per_sample_csv and os.path.exists(per_sample_csv):
        ps = pd.read_csv(per_sample_csv)

        if "cif_id" in ps.columns:
            keep_cols = [
                c
                for c in [
                    "cif_id",
                    "final_mode",
                    "p_change",
                    "sg_keep",
                    "cs_keep",
                    "best_sg_choice",
                    "picked_keep_kind",
                    "score_keep",
                    "min_d_keep",
                    "p_geom_keep",
                ]
                if c in ps.columns
            ]

            ps = ps[keep_cols].copy()
            meta = meta.merge(ps, on="cif_id", how="left")

        else:
            meta["final_mode"] = ""
            meta["p_change"] = np.nan
    else:
        meta["final_mode"] = ""
        meta["p_change"] = np.nan
        meta["sg_keep"] = np.nan
        meta["cs_keep"] = np.nan
        meta["best_sg_choice"] = np.nan

    meta["pred_path"] = meta["cif_id"].apply(lambda x: os.path.join(pred_dir, f"{x}.cif"))
    meta["dft_path"] = meta["cif_id"].apply(lambda x: os.path.join(dft_dir, f"{x}.cif"))

    return meta, pred_dir, dft_dir


# =========================
# Single sample eval
# =========================
def evaluate_one(row, symprec_base, symprec_scan, angtol, sg2halls):
    warnings.simplefilter("ignore")

    cif_id = row["cif_id"]
    pred_path = row["pred_path"]
    ref_path = row["dft_path"]

    out = {
        "cif_id": cif_id,
        "gt_mode": row.get("gt_mode", ""),
        "gt_y_type": int(row.get("y_type", -1)),
        "sg_u": int(row.get("sg_u", -1)),
        "sg_r": int(row.get("sg_r", -1)),
        "cs_u": int(row.get("cs_u", -1)),
        "cs_r": int(row.get("cs_r", -1)),
        "final_mode": row.get("final_mode", ""),
        "p_change": row.get("p_change", np.nan),
    }

    try:
        pred = standardize(load_structure(pred_path))
        ref = standardize(load_structure(ref_path))

    except Exception:
        out.update(
            {
                "struct_match": 0,
                "sg_pred": np.nan,
                "sg_ref": np.nan,
                "cs_pred": np.nan,
                "cs_ref": np.nan,
                "strict_sg_match": 0,
                "strict_cs_match": 0,
                "grouprel_sg_match": 0,
                "grouprel_relation": "unrelated",
                "sym_recovery_tol": np.nan,
                "lat_mae_a": np.nan,
                "lat_mae_b": np.nan,
                "lat_mae_c": np.nan,
                "lat_mae": np.nan,
                "vol_mae": np.nan,
                "coord_rmsd": np.nan,
                "bond_mae": np.nan,
                "cn_acc": np.nan,
                "pred_mode_proxy": "",
                "pred_mode_proxy_relation_to_u": "unrelated",
            }
        )
        return out

    struct_match = 1 if matcher.fit(pred, ref) else 0

    sg_pred = spacegroup(pred, symprec_base, angtol=angtol)
    sg_ref = spacegroup(ref, symprec_base, angtol=angtol)

    cs_pred = crystal_system_id(sg_pred) if sg_pred is not None else None
    cs_ref = crystal_system_id(sg_ref) if sg_ref is not None else None

    strict_sg_match = (
        1
        if (
            sg_pred is not None
            and sg_ref is not None
            and int(sg_pred) == int(sg_ref)
        )
        else 0
    )

    strict_cs_match = (
        1
        if (
            cs_pred is not None
            and cs_ref is not None
            and int(cs_pred) == int(cs_ref)
        )
        else 0
    )

    relation = is_group_related(sg_pred, sg_ref, sg2halls)
    grouprel_sg_match = 1 if relation in ("equal", "pred_sub_ref", "pred_sup_ref") else 0

    sym_tol = symmetry_recovery_tol(pred, ref, symprec_scan)

    lat_err = lattice_mae(pred, ref)
    vol_err = volume_mae(pred, ref)

    try:
        aligned_pred = matcher.get_s2_like_s1(ref, pred)
        rmsd = coordinate_rmsd_pbc(aligned_pred, ref)
        bond_err = bond_length_mae(aligned_pred, ref)
        cn_acc = coordination_accuracy(aligned_pred, ref)
    except Exception:
        rmsd = np.nan
        bond_err = np.nan
        cn_acc = np.nan

    out.update(
        {
            "struct_match": int(struct_match),
            "sg_pred": int(sg_pred) if sg_pred is not None else np.nan,
            "sg_ref": int(sg_ref) if sg_ref is not None else np.nan,
            "cs_pred": int(cs_pred) if cs_pred is not None else np.nan,
            "cs_ref": int(cs_ref) if cs_ref is not None else np.nan,
            "strict_sg_match": int(strict_sg_match),
            "strict_cs_match": int(strict_cs_match),
            "grouprel_sg_match": int(grouprel_sg_match),
            "grouprel_relation": relation,
            "sym_recovery_tol": sym_tol,
            "lat_mae_a": lat_err["lat_mae_a"],
            "lat_mae_b": lat_err["lat_mae_b"],
            "lat_mae_c": lat_err["lat_mae_c"],
            "lat_mae": lat_err["lat_mae"],
            "vol_mae": vol_err,
            "coord_rmsd": rmsd,
            "bond_mae": bond_err,
            "cn_acc": cn_acc,
        }
    )

    if not is_nan_like(out["sg_pred"]):
        rel_u = is_group_related(out["sg_pred"], out["sg_u"], sg2halls)
        out["pred_mode_proxy"] = (
            "KEEP" if rel_u in ("equal", "pred_sub_ref", "pred_sup_ref") else "CHANGE"
        )
        out["pred_mode_proxy_relation_to_u"] = rel_u
    else:
        out["pred_mode_proxy"] = ""
        out["pred_mode_proxy_relation_to_u"] = "unrelated"

    return out


# =========================
# Summary
# =========================
def build_summary(df: pd.DataFrame):
    total = int(len(df))

    if total == 0:
        return {
            "total_samples": 0,
            "metrics": {},
            "p1_triclinic_ratio": {},
            "group_relation_distribution": {},
            "top_cs_transitions": [],
            "keep_change_quadrants": {},
            "grouprel_sg_match_by_quadrant": {},
        }

    metrics = {
        "structure_match_rate_percent": float(df["struct_match"].mean() * 100.0),
        "strict_sg_match_rate_percent": float(df["strict_sg_match"].mean() * 100.0),
        "grouprel_sg_match_rate_percent": float(df["grouprel_sg_match"].mean() * 100.0),
        "strict_cs_match_rate_percent": float(df["strict_cs_match"].mean() * 100.0),
        "mean_symmetry_recovery_tol": safe_mean(df["sym_recovery_tol"]),
        "lattice_mae_a_angstrom": safe_mean(df["lat_mae_a"]),
        "lattice_mae_b_angstrom": safe_mean(df["lat_mae_b"]),
        "lattice_mae_c_angstrom": safe_mean(df["lat_mae_c"]),
        "lattice_mae_abc_mean_angstrom": safe_mean(df["lat_mae"]),
        "volume_mae_angstrom3": safe_mean(df["vol_mae"]),
        "coordinate_rmsd_angstrom": safe_mean(df["coord_rmsd"]),
        "bond_length_mae_angstrom": safe_mean(df["bond_mae"]),
        "coordination_accuracy_percent": float(safe_mean(df["cn_acc"]) * 100.0),
    }

    dft_p1 = float((df["sg_ref"] == 1).mean() * 100.0)
    pred_p1 = float((df["sg_pred"] == 1).mean() * 100.0)
    dft_tric = float((df["cs_ref"] == 0).mean() * 100.0)
    pred_tric = float((df["cs_pred"] == 0).mean() * 100.0)

    p1_triclinic_ratio = {
        "dft_p1_rate_percent": dft_p1,
        "pred_p1_rate_percent": pred_p1,
        "dft_triclinic_rate_percent": dft_tric,
        "pred_triclinic_rate_percent": pred_tric,
        "dft_p1_or_triclinic_rate_percent": max(dft_p1, dft_tric),
        "pred_p1_or_triclinic_rate_percent": max(pred_p1, pred_tric),
    }

    rel_cnt = df["grouprel_relation"].value_counts().to_dict()
    group_relation_distribution = {}

    for k in ["equal", "pred_sub_ref", "pred_sup_ref", "unrelated"]:
        n = int(rel_cnt.get(k, 0))
        group_relation_distribution[k] = {
            "count": n,
            "percent": float(n / max(total, 1) * 100.0),
        }

    trans = Counter()

    for a, b in zip(df["cs_ref"].tolist(), df["cs_pred"].tolist()):
        if is_nan_like(a) or is_nan_like(b):
            continue
        trans[(cs_name(int(a)), cs_name(int(b)))] += 1

    top_cs_transitions = []

    for (a, b), n in trans.most_common(20):
        top_cs_transitions.append(
            {
                "dft_cs": a,
                "pred_cs": b,
                "count": int(n),
                "percent": float(n / max(total, 1) * 100.0),
            }
        )

    use_final_mode = df["final_mode"].astype(str).str.len().gt(0).any()
    pred_mode_col = "final_mode" if use_final_mode else "pred_mode_proxy"

    tn = int(((df["gt_mode"] == "KEEP") & (df[pred_mode_col] == "KEEP")).sum())
    tp = int(((df["gt_mode"] == "CHANGE") & (df[pred_mode_col] == "CHANGE")).sum())
    fn = int(((df["gt_mode"] == "CHANGE") & (df[pred_mode_col] == "KEEP")).sum())
    fp = int(((df["gt_mode"] == "KEEP") & (df[pred_mode_col] == "CHANGE")).sum())

    keep_change_quadrants = {
        "pred_mode_source": pred_mode_col,
        "TN_keep": {
            "count": tn,
            "percent": float(tn / max(total, 1) * 100.0),
        },
        "TP_change": {
            "count": tp,
            "percent": float(tp / max(total, 1) * 100.0),
        },
        "FN_keep": {
            "count": fn,
            "percent": float(fn / max(total, 1) * 100.0),
        },
        "FP_change": {
            "count": fp,
            "percent": float(fp / max(total, 1) * 100.0),
        },
    }

    grouprel_by_quad = {}

    for name, mask in [
        ("TN_keep", (df["gt_mode"] == "KEEP") & (df[pred_mode_col] == "KEEP")),
        ("FP_change", (df["gt_mode"] == "KEEP") & (df[pred_mode_col] == "CHANGE")),
        ("FN_keep", (df["gt_mode"] == "CHANGE") & (df[pred_mode_col] == "KEEP")),
        ("TP_change", (df["gt_mode"] == "CHANGE") & (df[pred_mode_col] == "CHANGE")),
    ]:
        sub = df[mask]

        if len(sub) == 0:
            grouprel_by_quad[name] = {
                "count": 0,
                "grouprel_sg_match_percent": None,
            }
        else:
            grouprel_by_quad[name] = {
                "count": int(len(sub)),
                "grouprel_sg_match_percent": float(sub["grouprel_sg_match"].mean() * 100.0),
            }

    return {
        "total_samples": total,
        "metrics": metrics,
        "p1_triclinic_ratio": p1_triclinic_ratio,
        "group_relation_distribution": group_relation_distribution,
        "top_cs_transitions": top_cs_transitions,
        "keep_change_quadrants": keep_change_quadrants,
        "grouprel_sg_match_by_quadrant": grouprel_by_quad,
    }


# =========================
# Main
# =========================
def main():
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Evaluate predicted CIFs vs DFT CIFs."
    )

    ap.add_argument("--pred_name", type=str, default="pred")
    ap.add_argument("--dft_name", type=str, default="dft")

    ap.add_argument("--meta_csv", type=str, default="")
    ap.add_argument("--split_npy", type=str, default="")
    ap.add_argument("--per_sample_csv", type=str, default="")

    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--summary_json", type=str, default="")

    ap.add_argument("--symprec_base", type=float, default=1e-2)
    ap.add_argument("--angtol", type=float, default=5.0)
    ap.add_argument("--symprec_scan", type=str, default="")

    ap.add_argument("--n_jobs", type=int, default=DEFAULT_JOBS)
    ap.add_argument("--no_progress", action="store_true")

    args = ap.parse_args()

    warnings.simplefilter("ignore")

    script_dir = Path(__file__).resolve().parent

    pred_dir = script_dir / args.pred_name
    dft_dir = script_dir / args.dft_name

    meta_csv = Path(args.meta_csv).expanduser() if args.meta_csv.strip() else (script_dir / "meta.csv")
    split_npy = Path(args.split_npy).expanduser() if args.split_npy.strip() else (script_dir / "split_test.npy")

    if args.per_sample_csv.strip():
        per_sample_path = Path(args.per_sample_csv).expanduser()

        if not per_sample_path.is_absolute():
            per_sample_path = (script_dir / per_sample_path).resolve()

        per_sample_path = str(per_sample_path)

    else:
        cand = script_dir / "per_sample.csv"
        per_sample_path = str(cand) if cand.exists() else ""

    if args.out_csv.strip():
        out_csv = Path(args.out_csv).expanduser()

        if not out_csv.is_absolute():
            out_csv = (script_dir / out_csv).resolve()
    else:
        out_csv = script_dir / f"eval_{args.pred_name}_vs_{args.dft_name}.csv"

    if args.summary_json.strip():
        summary_json = Path(args.summary_json).expanduser()

        if not summary_json.is_absolute():
            summary_json = (script_dir / summary_json).resolve()
    else:
        summary_json = script_dir / f"summary_{args.pred_name}_vs_{args.dft_name}.json"

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")

    if not dft_dir.is_dir():
        raise FileNotFoundError(f"dft_dir not found: {dft_dir}")

    if not meta_csv.exists():
        raise FileNotFoundError(f"meta_csv not found: {meta_csv}")

    if not split_npy.exists():
        raise FileNotFoundError(f"split_npy not found: {split_npy}")

    symprec_scan = SYMPREC_SCAN_DEFAULT

    if args.symprec_scan.strip():
        symprec_scan = [float(x) for x in args.symprec_scan.split(",") if x.strip()]

    table, _, _ = build_eval_table(
        run_dir=str(script_dir),
        pred_name=args.pred_name,
        dft_name=args.dft_name,
        meta_csv=str(meta_csv),
        split_npy=str(split_npy),
        per_sample_csv=per_sample_path,
    )

    sg2halls = build_sg_to_halls(max_hall=530)

    rows = table.to_dict(orient="records")
    n_jobs = int(args.n_jobs)

    iterator = tqdm(
        rows,
        desc=f"Evaluating (jobs={n_jobs})",
        disable=bool(args.no_progress),
    )

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(evaluate_one)(
            row,
            symprec_base=float(args.symprec_base),
            symprec_scan=symprec_scan,
            angtol=float(args.angtol),
            sg2halls=sg2halls,
        )
        for row in iterator
    )

    df = pd.DataFrame(results)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_csv), index=False)

    summary = build_summary(df)

    summary_payload = {
        "input": {
            "pred_dir": str(pred_dir),
            "dft_dir": str(dft_dir),
            "meta_csv": str(meta_csv),
            "split_npy": str(split_npy),
            "per_sample_csv": str(per_sample_path),
            "out_csv": str(out_csv),
        },
        "settings": {
            "symprec_base": float(args.symprec_base),
            "angtol": float(args.angtol),
            "symprec_scan": [float(x) for x in symprec_scan],
            "n_jobs": int(n_jobs),
        },
        "summary": summary,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_clean(summary_payload), f, indent=2, ensure_ascii=False)

    if not args.no_progress:
        tqdm.write(f"saved eval csv: {out_csv}")
        tqdm.write(f"saved summary json: {summary_json}")


if __name__ == "__main__":
    main()
