# preprocess_sqlite.py

import multiprocessing as mp
import os
import sqlite3
import numpy as np
from tqdm import tqdm
from ase.io import read
from ase import Atoms
import pandas as pd
import pickle
from pathlib import Path
import argparse
import warnings
import re
import torch
import spglib
import math
import shutil
from typing import List, Tuple

from symm_graph import AtomsToGraphs

warnings.filterwarnings("ignore")


# -----------------------------
# CIF pair collector
# -----------------------------
def collect_initial_relax_pairs(
    data_root: str,
    initial_dir: str = "INITIAL",
    relax_dir: str = "RELAX",
    suffix: str = ".cif",
):
    initial_root = Path(data_root) / initial_dir
    relax_root = Path(data_root) / relax_dir

    if not initial_root.exists():
        raise FileNotFoundError(f"INITIAL dir not found: {initial_root}")
    if not relax_root.exists():
        raise FileNotFoundError(f"RELAX dir not found: {relax_root}")

    initial_map = {p.stem: p for p in initial_root.glob(f"*{suffix}")}
    relax_map = {p.stem: p for p in relax_root.glob(f"*{suffix}")}

    common_ids = sorted(set(initial_map.keys()) & set(relax_map.keys()))

    pairs = []
    for atoms_id in common_ids:
        pairs.append(
            {
                "atoms_id": atoms_id,
                "initial_path": str(initial_map[atoms_id]),
                "relax_path": str(relax_map[atoms_id]),
            }
        )

    print(f"✓ Found paired CIFs: {len(pairs)}")
    print(f"  INITIAL only: {len(set(initial_map.keys()) - set(relax_map.keys()))}")
    print(f"  RELAX only  : {len(set(relax_map.keys()) - set(initial_map.keys()))}")

    return pairs


# -----------------------------
# SQLite helpers
# -----------------------------
def open_write_db(db_path: str):
    conn = sqlite3.connect(db_path, timeout=120)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY,
            data BLOB NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def write_sample(cursor, idx: int, data):
    blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    cursor.execute(
        "INSERT INTO samples(id, data) VALUES (?, ?)",
        (int(idx), sqlite3.Binary(blob)),
    )


def merge_worker_sqlite_to_final(worker_meta, out_dir: str, db_name: str):
    out_dir = Path(out_dir)
    final_db_path = out_dir / db_name

    if final_db_path.exists():
        final_db_path.unlink()

    conn_out = open_write_db(str(final_db_path))
    cur_out = conn_out.cursor()

    offsets = []
    running = 0
    for wm in worker_meta:
        offsets.append(running)
        running += int(wm["length"])

    total = int(running)
    print(f"\n✓ Total usable samples written: {total}")

    meta_out_rows = []
    inserted = 0

    cur_out.execute("BEGIN")

    for wm, off in zip(worker_meta, offsets):
        worker_db_path = wm["worker_db_path"]

        conn_in = sqlite3.connect(f"file:{worker_db_path}?mode=ro", uri=True, timeout=120)
        cur_in = conn_in.cursor()

        for (
            local_idx,
            atoms_id,
            y_type,
            sg_u,
            sg_r,
            cs_u,
            cs_r,
            sym_level_u,
            sym_level_r,
        ) in wm["rows"]:
            gidx = int(off) + int(local_idx)

            row = cur_in.execute(
                "SELECT data FROM samples WHERE id=?",
                (int(local_idx),),
            ).fetchone()

            if row is None:
                raise RuntimeError(
                    f"Missing local sample {local_idx} in worker db: {worker_db_path}"
                )

            cur_out.execute(
                "INSERT INTO samples(id, data) VALUES (?, ?)",
                (gidx, sqlite3.Binary(row[0])),
            )

            meta_out_rows.append(
                [
                    gidx,
                    atoms_id,
                    y_type,
                    sg_u,
                    sg_r,
                    cs_u,
                    cs_r,
                    sym_level_u,
                    sym_level_r,
                ]
            )

            inserted += 1
            if inserted % 5000 == 0:
                conn_out.commit()
                cur_out.execute("BEGIN")

        conn_in.close()

    conn_out.commit()

    cur_out.execute(
        "INSERT OR REPLACE INTO info(key, value) VALUES (?, ?)",
        ("num_samples", str(total)),
    )
    conn_out.commit()

    conn_out.close()

    if inserted != total:
        raise RuntimeError(f"SQLite merge mismatch: inserted={inserted}, total={total}")

    print(f"✓ Saved {db_name}")
    return total, meta_out_rows


# -----------------------------
# CIF robust parser
# -----------------------------
def parse_cif_smart(filename: str) -> Atoms:
    temp_atoms = read(filename)
    cell = temp_atoms.get_cell()
    pbc = temp_atoms.get_pbc()

    symbols = []
    scaled_positions = []

    with open(filename) as f:
        content = f.read()

    lines = content.split("\n")
    in_atom_block = False
    header_map = {}

    for i, line in enumerate(lines):
        if line.strip().startswith("loop_"):
            next_lines = lines[i + 1 : i + 12]
            if any("_atom_site" in l for l in next_lines):
                in_atom_block = True
                j = i + 1
                col_idx = 0
                while j < len(lines) and lines[j].strip().startswith("_atom_site"):
                    header_name = lines[j].strip()
                    header_map[col_idx] = header_name
                    col_idx += 1
                    j += 1
                continue

        if in_atom_block:
            if line.strip().startswith("_atom_site"):
                continue
            if (
                line.strip().startswith("_")
                or line.strip().startswith("loop_")
                or not line.strip()
            ):
                if symbols:
                    break
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                symbol = None

                for col_idx, header_name in header_map.items():
                    if "type_symbol" in header_name and col_idx < len(parts):
                        symbol = parts[col_idx]
                        break

                if symbol is None:
                    for col_idx, header_name in header_map.items():
                        if "label" in header_name and col_idx < len(parts):
                            label = parts[col_idx]
                            symbol = re.sub(r"[^A-Za-z]", "", label)
                            break

                if not symbol:
                    continue

                x_idx = y_idx = z_idx = None

                for col_idx, header_name in header_map.items():
                    if "fract_x" in header_name:
                        x_idx = col_idx
                    elif "fract_y" in header_name:
                        y_idx = col_idx
                    elif "fract_z" in header_name:
                        z_idx = col_idx

                if x_idx is None or y_idx is None or z_idx is None:
                    continue

                if x_idx >= len(parts) or y_idx >= len(parts) or z_idx >= len(parts):
                    continue

                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])

                symbols.append(symbol)
                scaled_positions.append([x, y, z])

            except (ValueError, IndexError, KeyError):
                continue

    if not symbols:
        raise ValueError(f"Failed to parse atoms from {filename}")

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=pbc,
    )

    return atoms


# -----------------------------
# symmetry helpers
# -----------------------------
def ase_to_spglib_cell(atoms: Atoms):
    lattice = np.array(atoms.cell)
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    return (lattice, positions, numbers)


def get_spglib_dataset(atoms: Atoms, symprec=1e-2, angle_tolerance=5.0):
    try:
        cell = ase_to_spglib_cell(atoms)
        ds = spglib.get_symmetry_dataset(
            cell,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        return ds
    except Exception:
        return None


def robust_spglib_with_level(
    atoms: Atoms,
    symprec1=1e-2,
    angtol1=5.0,
    symprec2=5e-2,
    angtol2=10.0,
    symprec3=1e-1,
    angtol3=15.0,
):
    ds = get_spglib_dataset(atoms, symprec=symprec1, angle_tolerance=angtol1)
    if ds is not None:
        return ds, 1

    ds = get_spglib_dataset(atoms, symprec=symprec2, angle_tolerance=angtol2)
    if ds is not None:
        return ds, 2

    ds = get_spglib_dataset(atoms, symprec=symprec3, angle_tolerance=angtol3)
    if ds is not None:
        return ds, 3

    return None, 0


def crystal_system_id(spacegroup_number: int) -> int:
    sg = int(spacegroup_number)

    if sg < 1 or sg > 230:
        return -1
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

    return 6


def lattice_type_from_cs(cs: int) -> int:
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


def frac_mod1(x):
    return x - np.floor(x)


def frac_diff(a, b):
    d = a - b
    d = d - np.round(d)
    return d


def find_op_id_for_atom(rep_frac, target_frac, rotations, translations, tol=2e-3):
    K = rotations.shape[0]

    for k in range(K):
        cand = rotations[k].dot(rep_frac) + translations[k]
        cand = frac_mod1(cand)
        d = frac_diff(cand, target_frac)

        if np.max(np.abs(d)) < tol:
            return k

    for k in range(K):
        cand = rotations[k].dot(rep_frac) + translations[k]
        cand = frac_mod1(cand)
        d = frac_diff(cand, target_frac)

        if np.max(np.abs(d)) < 1e-2:
            return k

    return -1


# -----------------------------
# lattice params helper
# -----------------------------
def cell_to_abc_angles(cell_3x3: np.ndarray):
    a_vec, b_vec, c_vec = cell_3x3[0], cell_3x3[1], cell_3x3[2]

    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)

    def angle(u, v):
        cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        cosang = np.clip(cosang, -1.0, 1.0)
        return math.acos(cosang)

    alpha = angle(b_vec, c_vec)
    beta = angle(a_vec, c_vec)
    gamma = angle(a_vec, b_vec)

    return np.array([a, b, c, alpha, beta, gamma], dtype=np.float32)


# -----------------------------
# Wyckoff affine inference
# -----------------------------
def infer_wyckoff_affine_for_rep(
    rep_frac: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    tol=2e-3,
):
    rep = rep_frac.astype(np.float64)

    stab_rows = []

    for k in range(rotations.shape[0]):
        R = rotations[k].astype(np.int64)
        t = translations[k].astype(np.float64)

        cand = R.dot(rep) + t
        cand = frac_mod1(cand)

        if np.max(np.abs(frac_diff(cand, rep))) < tol:
            n = np.round(R.dot(rep) + t - rep).astype(np.float64)
            M = (R - np.eye(3)).astype(np.float64)
            bvec = (-t + n).astype(np.float64)

            for i in range(3):
                row = np.concatenate([M[i], [bvec[i]]], axis=0)
                if np.linalg.norm(row[:3]) > 1e-12:
                    stab_rows.append(row)

    if len(stab_rows) == 0:
        A = np.eye(3, dtype=np.float32)
        b = rep_frac.astype(np.float32)
        dof_mask = np.ones((3,), dtype=np.float32)
        return A, b, dof_mask

    stab = np.stack(stab_rows, axis=0)
    M = stab[:, :3]

    _, S, Vt = np.linalg.svd(M, full_matrices=True)

    rank = int((S > 1e-8).sum())
    d = 3 - rank

    if d < 0:
        d = 0

    if d == 0:
        A = np.zeros((3, 3), dtype=np.float32)
        dof_mask = np.zeros((3,), dtype=np.float32)
    else:
        basis = Vt[rank:, :].T
        A = np.zeros((3, 3), dtype=np.float32)
        A[:, :d] = basis.astype(np.float32)

        dof_mask = np.zeros((3,), dtype=np.float32)
        dof_mask[:d] = 1.0

    b = rep_frac.astype(np.float32)

    return A, b, dof_mask


def build_wyckoff_package_from_dataset(ds, frac_all: np.ndarray, tol_map=2e-3):
    sg = int(ds["number"])
    cs = crystal_system_id(sg)

    rotations = np.array(ds["rotations"], dtype=np.int64)
    translations = np.array(ds["translations"], dtype=np.float32)

    if rotations.shape[0] < 1:
        raise RuntimeError("No symmetry ops")

    equiv = np.array(ds["equivalent_atoms"], dtype=np.int64)

    if equiv.shape[0] != frac_all.shape[0]:
        raise RuntimeError("equivalent_atoms size mismatch")

    unique_classes = np.unique(equiv)
    class_to_wyc = {c: j for j, c in enumerate(unique_classes.tolist())}

    atom_wyc_local = np.array([class_to_wyc[c] for c in equiv], dtype=np.int64)
    n_wyc = unique_classes.shape[0]

    rep_index = np.full((n_wyc,), -1, dtype=np.int64)

    for ai in range(frac_all.shape[0]):
        w = atom_wyc_local[ai]
        if rep_index[w] < 0:
            rep_index[w] = ai

    if np.any(rep_index < 0):
        raise RuntimeError("rep_index build failed")

    rep_frac = frac_all[rep_index].astype(np.float32)

    atom_op_id = np.full((frac_all.shape[0],), -1, dtype=np.int64)

    for ai in range(frac_all.shape[0]):
        w = atom_wyc_local[ai]
        rep_ai = rep_index[w]

        op = find_op_id_for_atom(
            rep_frac=frac_all[rep_ai],
            target_frac=frac_all[ai],
            rotations=rotations,
            translations=translations,
            tol=tol_map,
        )

        atom_op_id[ai] = op

    if np.any(atom_op_id < 0):
        raise RuntimeError("op id mapping failed")

    wyc_A = np.zeros((n_wyc, 3, 3), dtype=np.float32)
    wyc_b = np.zeros((n_wyc, 3), dtype=np.float32)
    wyc_dof_mask = np.zeros((n_wyc, 3), dtype=np.float32)

    for wi in range(n_wyc):
        rep = rep_frac[wi]

        A, b, m = infer_wyckoff_affine_for_rep(
            rep_frac=rep,
            rotations=rotations,
            translations=translations,
            tol=2e-3,
        )

        wyc_A[wi] = A
        wyc_b[wi] = b
        wyc_dof_mask[wi] = m

    return {
        "sg": sg,
        "cs": cs,
        "rotations": rotations,
        "translations": translations,
        "atom_wyc_local": atom_wyc_local,
        "atom_op_id": atom_op_id,
        "rep_index": rep_index,
        "rep_frac": rep_frac,
        "wyc_A": wyc_A,
        "wyc_b": wyc_b,
        "wyc_dof_mask": wyc_dof_mask,
    }


def compute_y_type(sg_u: int, sg_r: int, cs_u: int, cs_r: int) -> int:
    if int(sg_u) == int(sg_r):
        return 0

    return 1 if int(cs_u) == int(cs_r) else 2


# -----------------------------
# graph fallback
# -----------------------------
def build_pair_graph_fallback(a2g: AtomsToGraphs, atoms_u: Atoms, atoms_r: Atoms):
    data = a2g.convert_single(atoms_u)

    positions_u = data.pos_u
    cell_r = torch.Tensor(atoms_r.get_cell())
    positions_r = torch.Tensor(atoms_r.get_positions())

    unwrapped_positions_r = a2g.unwrap_cartesian_positions(
        positions_u,
        positions_r,
        cell_r,
    )

    atoms_r.set_positions(unwrapped_positions_r)
    positions_r = torch.Tensor(atoms_r.get_positions())

    data.cell_r = cell_r.view(1, 3, 3)
    data.pos_r = positions_r

    return data


# -----------------------------
# worker
# -----------------------------
def write_data(mp_args):
    (
        a2g,
        pairs,
        data_indices,
        worker_id,
        worker_db_path,
        symprec1,
        angtol1,
        symprec2,
        angtol2,
        symprec3,
        angtol3,
        out_dir,
    ) = mp_args

    if os.path.exists(worker_db_path):
        os.remove(worker_db_path)

    conn = open_write_db(worker_db_path)
    cur = conn.cursor()

    idx = 0
    skipped = 0

    skip_reasons = {
        k: 0
        for k in [
            "missing_files",
            "too_few_atoms",
            "atom_count_mismatch",
            "composition_mismatch",
            "atomic_order_mismatch",
            "neighbor_mismatch",
            "neighbor_mismatch_fallback_fail",
            "parse_error",
            "sym_fail",
            "wyc_fail",
            "symmetry_changed",
            "other_error",
        ]
    }

    meta_rows: List[Tuple[int, str, int, int, int, int, int, int, int]] = []

    pbar = tqdm(
        data_indices,
        desc=f"Worker {worker_id}",
        position=worker_id,
        leave=True,
    )

    cur.execute("BEGIN")

    for index in pbar:
        item = pairs[index]
        atoms_id = str(item["atoms_id"])

        unrelaxed_path = item["initial_path"]
        relaxed_path = item["relax_path"]

        if not os.path.exists(relaxed_path) or not os.path.exists(unrelaxed_path):
            skip_reasons["missing_files"] += 1
            skipped += 1
            continue

        try:
            atoms_r = parse_cif_smart(relaxed_path)
            atoms_u = parse_cif_smart(unrelaxed_path)

            if len(atoms_r) < 3:
                skip_reasons["too_few_atoms"] += 1
                skipped += 1
                continue

            if len(atoms_r) != len(atoms_u):
                skip_reasons["atom_count_mismatch"] += 1
                skipped += 1
                continue

            if atoms_u.get_chemical_formula() != atoms_r.get_chemical_formula():
                skip_reasons["composition_mismatch"] += 1
                skipped += 1
                continue

            nums_u_np = atoms_u.get_atomic_numbers().astype(np.int64)
            nums_r_np = atoms_r.get_atomic_numbers().astype(np.int64)

            if not np.array_equal(nums_u_np, nums_r_np):
                skip_reasons["atomic_order_mismatch"] += 1
                skipped += 1
                continue

            frac_u = atoms_u.get_scaled_positions().astype(np.float32)
            frac_r = atoms_r.get_scaled_positions().astype(np.float32)

            ds_u, sym_level_u = robust_spglib_with_level(
                atoms_u,
                symprec1=symprec1,
                angtol1=angtol1,
                symprec2=symprec2,
                angtol2=angtol2,
                symprec3=symprec3,
                angtol3=angtol3,
            )

            ds_r, sym_level_r = robust_spglib_with_level(
                atoms_r,
                symprec1=symprec1,
                angtol1=angtol1,
                symprec2=symprec2,
                angtol2=angtol2,
                symprec3=symprec3,
                angtol3=angtol3,
            )

            if ds_u is None or ds_r is None:
                skip_reasons["sym_fail"] += 1
                skipped += 1
                continue

            try:
                pkg_u = build_wyckoff_package_from_dataset(ds_u, frac_u)
                pkg_r = build_wyckoff_package_from_dataset(ds_r, frac_r)
            except Exception:
                skip_reasons["wyc_fail"] += 1
                skipped += 1
                continue

            sg_u = int(pkg_u["sg"])
            sg_r = int(pkg_r["sg"])
            cs_u = int(pkg_u["cs"])
            cs_r = int(pkg_r["cs"])

            y_type = compute_y_type(sg_u, sg_r, cs_u, cs_r)

            if y_type != 0:
                skip_reasons["symmetry_changed"] += 1
                skipped += 1
                continue

            y_change = 0.0

            try:
                data = a2g.convert_pairs(atoms_u, atoms_r)
            except RuntimeError as e:
                if "must match the size" in str(e) or "size" in str(e).lower():
                    skip_reasons["neighbor_mismatch"] += 1
                    try:
                        data = build_pair_graph_fallback(a2g, atoms_u, atoms_r)
                    except Exception:
                        skip_reasons["neighbor_mismatch_fallback_fail"] += 1
                        skipped += 1
                        continue
                else:
                    raise

            nums = torch.from_numpy(nums_u_np).long()

            data.atomic_numbers = nums
            data.z = nums
            data.cif_id = atoms_id

            data.pos_frac_u = torch.from_numpy(frac_u)
            data.pos_frac_r = torch.from_numpy(frac_r)

            lt = lattice_type_from_cs(cs_u if cs_u != -1 else cs_r)

            cell_param_u = cell_to_abc_angles(np.array(atoms_u.cell))
            cell_param_r = cell_to_abc_angles(np.array(atoms_r.cell))

            data.lattice_type = torch.tensor([lt], dtype=torch.long)
            data.cell_param_u = torch.from_numpy(cell_param_u).unsqueeze(0).float()
            data.cell_param_r = torch.from_numpy(cell_param_r).unsqueeze(0).float()

            data.sg_u = torch.tensor([sg_u], dtype=torch.long)
            data.sg_r = torch.tensor([sg_r], dtype=torch.long)
            data.cs_u = torch.tensor([cs_u], dtype=torch.long)
            data.cs_r = torch.tensor([cs_r], dtype=torch.long)
            data.y_change = torch.tensor([y_change], dtype=torch.float32)
            data.y_type = torch.tensor([y_type], dtype=torch.long)

            rep_idx_u = pkg_u["rep_index"]
            rep_frac_target_u = frac_r[rep_idx_u].astype(np.float32)

            data.symm_R_u = torch.from_numpy(pkg_u["rotations"]).long()
            data.symm_t_u = torch.from_numpy(pkg_u["translations"]).float()
            data.atom_op_id_u = torch.from_numpy(pkg_u["atom_op_id"]).long()
            data.atom_wyc_u = torch.from_numpy(pkg_u["atom_wyc_local"]).long()
            data.wyc_A_u = torch.from_numpy(pkg_u["wyc_A"]).float()
            data.wyc_b_u = torch.from_numpy(pkg_u["wyc_b"]).float()
            data.wyc_dof_mask_u = torch.from_numpy(pkg_u["wyc_dof_mask"]).float()
            data.wyc_rep_target_u = torch.from_numpy(rep_frac_target_u).float()

            rep_idx_r = pkg_r["rep_index"]
            rep_frac_target_r = frac_r[rep_idx_r].astype(np.float32)

            data.symm_R_r = torch.from_numpy(pkg_r["rotations"]).long()
            data.symm_t_r = torch.from_numpy(pkg_r["translations"]).float()
            data.atom_op_id_r = torch.from_numpy(pkg_r["atom_op_id"]).long()
            data.atom_wyc_r = torch.from_numpy(pkg_r["atom_wyc_local"]).long()
            data.wyc_A_r = torch.from_numpy(pkg_r["wyc_A"]).float()
            data.wyc_b_r = torch.from_numpy(pkg_r["wyc_b"]).float()
            data.wyc_dof_mask_r = torch.from_numpy(pkg_r["wyc_dof_mask"]).float()
            data.wyc_rep_target_r = torch.from_numpy(rep_frac_target_r).float()

            if (
                hasattr(data, "pos_r")
                and hasattr(data, "cell_r")
                and hasattr(data, "edge_index")
                and hasattr(data, "cell_offsets")
            ):
                try:
                    pos_r_cart = data.pos_r.detach().cpu().numpy()
                    cell_r_cart = data.cell_r.detach().cpu().numpy().reshape(3, 3)
                    edge_index = data.edge_index.detach().cpu().numpy()
                    cell_offsets = data.cell_offsets.detach().cpu().numpy().astype(np.int64)

                    j = edge_index[0]
                    i = edge_index[1]

                    vecs = (pos_r_cart[j] + cell_offsets @ cell_r_cart) - pos_r_cart[i]
                    dist = np.linalg.norm(vecs, axis=1).astype(np.float32)

                    data.edge_dist_r = torch.from_numpy(dist)
                except Exception:
                    pass

            write_sample(cur, idx, data)

            meta_rows.append(
                (
                    idx,
                    atoms_id,
                    y_type,
                    sg_u,
                    sg_r,
                    cs_u,
                    cs_r,
                    sym_level_u,
                    sym_level_r,
                )
            )

            idx += 1

            if idx % 1000 == 0:
                conn.commit()
                cur.execute("BEGIN")

        except ValueError as e:
            if "Failed to parse" in str(e):
                skip_reasons["parse_error"] += 1
                skipped += 1
                continue

            skip_reasons["other_error"] += 1
            skipped += 1
            pbar.write(f"[ERROR] {atoms_id}: {str(e)[:120]}")
            continue

        except Exception as e:
            skip_reasons["other_error"] += 1
            skipped += 1
            pbar.write(f"[ERROR] {atoms_id}: {str(e)[:120]}")
            continue

    conn.commit()

    conn.execute(
        "INSERT OR REPLACE INTO info(key, value) VALUES (?, ?)",
        ("num_samples", str(idx)),
    )
    conn.commit()
    conn.close()

    meta_path = os.path.join(out_dir, f"meta_worker_{worker_id:04d}.pkl")

    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "worker_id": worker_id,
                "worker_db_path": worker_db_path,
                "length": idx,
                "rows": meta_rows,
                "skip_reasons": skip_reasons,
                "skipped": skipped,
            },
            f,
            protocol=-1,
        )

    print(f"\n[Worker {worker_id}] Finished:")
    print(f"  Written: {idx}")
    print(f"  Skipped: {skipped}")

    if skipped > 0:
        print("  Skip reasons:")
        for reason, count in skip_reasons.items():
            if count > 0:
                print(f"    - {reason}: {count}")


def _random_split(
    total: int,
    ratios: Tuple[float, float, float],
    seed: int,
):
    rng = np.random.default_rng(seed)

    r_train, r_val, r_test = ratios

    if abs((r_train + r_val + r_test) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    indices = np.arange(total, dtype=np.int64)
    rng.shuffle(indices)

    n_train = int(round(total * r_train))
    n_val = int(round(total * r_val))
    n_test = total - n_train - n_val

    if n_test < 0:
        n_test = 0
        n_val = total - n_train

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]

    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_name", type=str, default="processed_sqlite")
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--initial_dir", type=str, default="INITIAL")
    parser.add_argument("--relax_dir", type=str, default="RELAX")
    parser.add_argument("--suffix", type=str, default=".cif")

    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--max_neigh", type=int, default=50)
    parser.add_argument("--max_displace", type=int, default=20)

    parser.add_argument("--symprec1", type=float, default=1e-2)
    parser.add_argument("--angtol1", type=float, default=5.0)

    parser.add_argument("--symprec2", type=float, default=5e-2)
    parser.add_argument("--angtol2", type=float, default=10.0)

    parser.add_argument("--symprec3", type=float, default=1e-1)
    parser.add_argument("--angtol3", type=float, default=15.0)

    parser.add_argument("--split_ratios", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--db_name", type=str, default="data.sqlite")

    args = parser.parse_args()

    data_root = args.data_root
    num_workers = int(args.num_workers)

    pairs = collect_initial_relax_pairs(
        data_root=data_root,
        initial_dir=args.initial_dir,
        relax_dir=args.relax_dir,
        suffix=args.suffix,
    )

    out_dir = os.path.join(data_root, args.out_name)
    save_path = Path(out_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    tmp_dir = os.path.join(out_dir, "_tmp_workers")

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir, exist_ok=True)

    for old_meta in Path(out_dir).glob("meta_worker_*.pkl"):
        old_meta.unlink()

    for stale_name in [
        "idx_keep.npy",
        "idx_soft.npy",
        "idx_hard.npy",
        "data.pt",
        "shards.json",
        args.db_name,
    ]:
        stale_path = Path(out_dir) / stale_name
        if stale_path.exists():
            stale_path.unlink()

    for stale_pt in Path(out_dir).glob("data_*.pt"):
        stale_pt.unlink()

    a2g = AtomsToGraphs(
        radius=float(args.radius),
        max_neigh=int(args.max_neigh),
        max_displace=int(args.max_displace),
    )

    data_len = len(pairs)
    data_indices = np.arange(data_len)
    mp_data_indices = np.array_split(data_indices, num_workers)

    worker_db_paths = [
        os.path.join(tmp_dir, f"worker_{i:04d}.sqlite")
        for i in range(num_workers)
    ]

    mp_args = [
        (
            a2g,
            pairs,
            mp_data_indices[i],
            i,
            worker_db_paths[i],
            float(args.symprec1),
            float(args.angtol1),
            float(args.symprec2),
            float(args.angtol2),
            float(args.symprec3),
            float(args.angtol3),
            out_dir,
        )
        for i in range(num_workers)
    ]

    print(f"\nProcessing: {args.out_name}")
    print(f"Storage format: SQLite, db_name={args.db_name}")
    print("Filter: symmetry unchanged only")

    pool = mp.Pool(num_workers)
    pool.map(write_data, mp_args)
    pool.close()
    pool.join()

    meta_files = sorted(Path(out_dir).glob("meta_worker_*.pkl"))

    if len(meta_files) == 0:
        raise RuntimeError("No meta_worker_*.pkl found. Preprocess likely failed.")

    worker_meta = []

    for mf in meta_files:
        with open(mf, "rb") as f:
            worker_meta.append(pickle.load(f))

    worker_meta = sorted(worker_meta, key=lambda x: int(x["worker_id"]))

    all_reasons = {}

    for wm in worker_meta:
        for k, v in wm.get("skip_reasons", {}).items():
            all_reasons[k] = all_reasons.get(k, 0) + int(v)

    skip_df = pd.DataFrame(
        [
            {"reason": k, "count": v}
            for k, v in sorted(all_reasons.items(), key=lambda x: -x[1])
        ]
    )

    skip_df.to_csv(os.path.join(out_dir, "skip_summary.csv"), index=False)
    print("✓ Saved skip_summary.csv")

    total, meta_out_rows = merge_worker_sqlite_to_final(
        worker_meta=worker_meta,
        out_dir=out_dir,
        db_name=args.db_name,
    )

    meta_df = pd.DataFrame(
        meta_out_rows,
        columns=[
            "global_idx",
            "atoms_id",
            "y_type",
            "sg_u",
            "sg_r",
            "cs_u",
            "cs_r",
            "sym_level_u",
            "sym_level_r",
        ],
    )

    meta_df = meta_df.sort_values("global_idx")
    meta_df.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    print("✓ Saved meta.csv")

    ratios = tuple(float(x) for x in args.split_ratios.split(","))

    if len(ratios) != 3:
        raise ValueError("--split_ratios must be 'train,val,test' like 0.8,0.1,0.1")

    train_idx, val_idx, test_idx = _random_split(
        total=total,
        ratios=ratios,
        seed=int(args.seed),
    )

    np.save(os.path.join(out_dir, "split_train.npy"), train_idx)
    np.save(os.path.join(out_dir, "split_val.npy"), val_idx)
    np.save(os.path.join(out_dir, "split_test.npy"), test_idx)

    print(
        f"✓ Saved split indices: "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    print("\n[Split stats]")
    print(f"  train: {len(train_idx)}")
    print(f"  val  : {len(val_idx)}")
    print(f"  test : {len(test_idx)}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    for mf in Path(out_dir).glob("meta_worker_*.pkl"):
        mf.unlink()

    print("\n🎉 Preprocessing completed!")


if __name__ == "__main__":
    main()
