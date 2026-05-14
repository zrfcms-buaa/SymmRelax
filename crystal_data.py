# crystal_data.py

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
import torch_geometric


def fix_pyg_data(data: Data):
    if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
        return Data(**{k: v for k, v in data.__dict__.items() if v is not None})
    return data


class CrystalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        db_name="data.sqlite",
        transform=None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.db_name = db_name
        self.transform = transform

        self.db_path = self.data_dir / self.db_name

        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")

        self.conn = None
        self.cursor = None

        if split in ("train", "val", "test"):
            split_path = self.data_dir / f"split_{split}.npy"

            if not split_path.exists():
                raise FileNotFoundError(f"Split file not found: {split_path}")

            self.indices = np.load(split_path).astype(np.int64)

        elif split in ("all", None):
            self.indices = np.arange(self._read_num_samples(), dtype=np.int64)

        else:
            raise ValueError("split must be one of: train, val, test, all")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["conn"] = None
        state["cursor"] = None
        return state

    def __len__(self):
        return len(self.indices)

    def _connect(self):
        if self.conn is not None:
            return self.conn

        uri = f"file:{self.db_path}?mode=ro"

        self.conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=120,
            check_same_thread=False,
        )

        self.conn.execute("PRAGMA query_only=ON")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-65536")
        self.conn.execute("PRAGMA mmap_size=30000000000")

        self.cursor = self.conn.cursor()
        return self.conn

    def _read_num_samples(self):
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            timeout=120,
        )

        try:
            row = conn.execute(
                "SELECT value FROM info WHERE key='num_samples'"
            ).fetchone()

            if row is not None:
                return int(row[0])

            row = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
            return int(row[0])

        finally:
            conn.close()

    def __getitem__(self, idx):
        global_idx = int(self.indices[idx])

        self._connect()

        row = self.cursor.execute(
            "SELECT data FROM samples WHERE id=?",
            (global_idx,),
        ).fetchone()

        if row is None:
            raise KeyError(f"Missing sample id={global_idx}")

        data = pickle.loads(row[0])
        data = fix_pyg_data(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, str(global_idx)

    def close(self):
        if self.cursor is not None:
            self.cursor.close()
            self.cursor = None

        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _safe_cat(data_list, attr: str, dtype=None, view=None, default=None):
    vals = []

    for d in data_list:
        if not hasattr(d, attr) or getattr(d, attr) is None:
            return default
        vals.append(getattr(d, attr))

    try:
        out = torch.cat(vals, dim=0)
    except Exception:
        return default

    if dtype is not None:
        out = out.to(dtype=dtype)

    if view is not None:
        out = out.view(view)

    return out


def _pad_ops(data_list, attr_R: str, attr_t: str):
    for d in data_list:
        if (
            not hasattr(d, attr_R)
            or getattr(d, attr_R) is None
            or not hasattr(d, attr_t)
            or getattr(d, attr_t) is None
        ):
            return None, None

    B = len(data_list)
    Ks = [int(getattr(d, attr_R).size(0)) for d in data_list]
    Kmax = max(Ks) if len(Ks) else 0

    R = torch.zeros((B, Kmax, 3, 3), dtype=torch.long)
    t = torch.zeros((B, Kmax, 3), dtype=torch.float32)

    for bi, d in enumerate(data_list):
        k = int(getattr(d, attr_R).size(0))

        if k > 0:
            R[bi, :k] = getattr(d, attr_R).long()
            t[bi, :k] = getattr(d, attr_t).float()

    return R, t


def _stack_scalar(data_list, attr: str, B: int, dtype: torch.dtype, default_val=0):
    out = []

    for d in data_list:
        if hasattr(d, attr) and getattr(d, attr) is not None:
            v = getattr(d, attr).view(-1)

            if v.numel() == 0:
                out.append(torch.tensor([default_val], dtype=dtype))
            else:
                out.append(v[:1].to(dtype=dtype))
        else:
            out.append(torch.tensor([default_val], dtype=dtype))

    return torch.cat(out, dim=0).view(B)


def make_batch(samples, otf_graph: bool = False):
    data_list, _ = map(list, zip(*samples))

    batch = Batch.from_data_list(data_list)
    B = len(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for d in data_list:
                n_neighbors.append(int(d.edge_index.shape[1]))
            batch.neighbors = torch.tensor(n_neighbors, dtype=torch.long)
        except Exception:
            pass

    batch.atom_batch = batch.batch

    batch.lattice_type = _stack_scalar(
        data_list,
        "lattice_type",
        B,
        torch.long,
        default_val=0,
    )

    batch.cell_param_u = _safe_cat(
        data_list,
        "cell_param_u",
        dtype=torch.float32,
        view=(B, 6),
        default=torch.zeros((B, 6), dtype=torch.float32),
    )

    batch.cell_param_r = _safe_cat(
        data_list,
        "cell_param_r",
        dtype=torch.float32,
        view=(B, 6),
        default=torch.zeros((B, 6), dtype=torch.float32),
    )

    batch.energy_r = _stack_scalar(
        data_list,
        "energy_r",
        B,
        torch.float32,
        default_val=0.0,
    )

    batch.sg_u = _stack_scalar(data_list, "sg_u", B, torch.long, default_val=0)
    batch.sg_r = _stack_scalar(data_list, "sg_r", B, torch.long, default_val=0)

    batch.cs_u = _stack_scalar(data_list, "cs_u", B, torch.long, default_val=-1)
    batch.cs_r = _stack_scalar(data_list, "cs_r", B, torch.long, default_val=-1)

    has_y_type = all(
        hasattr(d, "y_type") and getattr(d, "y_type") is not None
        for d in data_list
    )

    if has_y_type:
        batch.y_type = _stack_scalar(
            data_list,
            "y_type",
            B,
            torch.long,
            default_val=0,
        )
        batch.y_change = (batch.y_type > 0).float()
        batch.y_soft = (batch.y_type == 1).float()
        batch.y_hard = (batch.y_type == 2).float()
    else:
        batch.y_change = _stack_scalar(
            data_list,
            "y_change",
            B,
            torch.float32,
            default_val=0.0,
        )
        batch.y_type = (batch.y_change > 0).long()
        batch.y_soft = (batch.y_type == 1).float()
        batch.y_hard = (batch.y_type == 2).float()

    edge_dist_r = _safe_cat(
        data_list,
        "edge_dist_r",
        dtype=torch.float32,
        view=-1,
        default=None,
    )

    if edge_dist_r is not None:
        batch.edge_dist_r = edge_dist_r

    has_u_pkg = all(
        hasattr(d, "wyc_b_u") and getattr(d, "wyc_b_u") is not None
        for d in data_list
    )

    if has_u_pkg:
        wyc_u_offsets = []
        n_wyc_u = []
        total_wyc_u = 0

        for d in data_list:
            n = int(d.wyc_b_u.size(0))
            wyc_u_offsets.append(total_wyc_u)
            n_wyc_u.append(n)
            total_wyc_u += n

        batch.wyc_A_u = _safe_cat(
            data_list,
            "wyc_A_u",
            dtype=torch.float32,
            default=torch.zeros((0, 3, 3), dtype=torch.float32),
        )

        batch.wyc_b_u = _safe_cat(
            data_list,
            "wyc_b_u",
            dtype=torch.float32,
            default=torch.zeros((0, 3), dtype=torch.float32),
        )

        batch.wyc_dof_mask_u = _safe_cat(
            data_list,
            "wyc_dof_mask_u",
            dtype=torch.float32,
            default=torch.zeros((0, 3), dtype=torch.float32),
        )

        batch.wyc_rep_target_u = _safe_cat(
            data_list,
            "wyc_rep_target_u",
            dtype=torch.float32,
            default=torch.zeros((0, 3), dtype=torch.float32),
        )

        atom_wyc_global_u = []

        for bi, d in enumerate(data_list):
            if hasattr(d, "atom_wyc_u") and d.atom_wyc_u is not None:
                atom_wyc_global_u.append(
                    d.atom_wyc_u.long().view(-1) + int(wyc_u_offsets[bi])
                )
            else:
                atom_wyc_global_u.append(
                    torch.zeros((int(d.num_nodes),), dtype=torch.long)
                )

        batch.atom_wyc_global_u = torch.cat(atom_wyc_global_u, dim=0).long()

        if hasattr(batch, "atom_op_id_u") and batch.atom_op_id_u is not None:
            batch.atom_op_id_u = batch.atom_op_id_u.long()

        if total_wyc_u > 0:
            batch.wyc_batch_u = torch.cat(
                [
                    torch.full((int(n_wyc_u[bi]),), bi, dtype=torch.long)
                    for bi in range(B)
                ],
                dim=0,
            )
        else:
            batch.wyc_batch_u = torch.zeros((0,), dtype=torch.long)

        batch.wyc_u_offsets = torch.tensor(wyc_u_offsets, dtype=torch.long)
        batch.n_wyc_u = torch.tensor(n_wyc_u, dtype=torch.long)

        symm_R_u, symm_t_u = _pad_ops(data_list, "symm_R_u", "symm_t_u")

        if symm_R_u is not None:
            batch.symm_R_u = symm_R_u
            batch.symm_t_u = symm_t_u

    has_r_pkg = all(
        hasattr(d, "wyc_b_r") and getattr(d, "wyc_b_r") is not None
        for d in data_list
    )

    if has_r_pkg:
        wyc_r_offsets = []
        n_wyc_r = []
        total_wyc_r = 0

        for d in data_list:
            n = int(d.wyc_b_r.size(0))
            wyc_r_offsets.append(total_wyc_r)
            n_wyc_r.append(n)
            total_wyc_r += n

        batch.wyc_A_r = _safe_cat(
            data_list,
            "wyc_A_r",
            dtype=torch.float32,
            default=torch.zeros((0, 3, 3), dtype=torch.float32),
        )

        batch.wyc_b_r = _safe_cat(
            data_list,
            "wyc_b_r",
            dtype=torch.float32,
            default=torch.zeros((0, 3), dtype=torch.float32),
        )

        batch.wyc_dof_mask_r = _safe_cat(
            data_list,
            "wyc_dof_mask_r",
            dtype=torch.float32,
            default=torch.zeros((0, 3), dtype=torch.float32),
        )

        batch.wyc_rep_target_r = _safe_cat(
            data_list,
            "wyc_rep_target_r",
            dtype=torch.float32,
            default=torch.zeros((0, 3), dtype=torch.float32),
        )

        atom_wyc_global_r = []

        for bi, d in enumerate(data_list):
            if hasattr(d, "atom_wyc_r") and d.atom_wyc_r is not None:
                atom_wyc_global_r.append(
                    d.atom_wyc_r.long().view(-1) + int(wyc_r_offsets[bi])
                )
            else:
                atom_wyc_global_r.append(
                    torch.zeros((int(d.num_nodes),), dtype=torch.long)
                )

        batch.atom_wyc_global_r = torch.cat(atom_wyc_global_r, dim=0).long()

        if hasattr(batch, "atom_op_id_r") and batch.atom_op_id_r is not None:
            batch.atom_op_id_r = batch.atom_op_id_r.long()

        if total_wyc_r > 0:
            batch.wyc_batch_r = torch.cat(
                [
                    torch.full((int(n_wyc_r[bi]),), bi, dtype=torch.long)
                    for bi in range(B)
                ],
                dim=0,
            )
        else:
            batch.wyc_batch_r = torch.zeros((0,), dtype=torch.long)

        batch.wyc_r_offsets = torch.tensor(wyc_r_offsets, dtype=torch.long)
        batch.n_wyc_r = torch.tensor(n_wyc_r, dtype=torch.long)

        symm_R_r, symm_t_r = _pad_ops(data_list, "symm_R_r", "symm_t_r")

        if symm_R_r is not None:
            batch.symm_R_r = symm_R_r
            batch.symm_t_r = symm_t_r

    return batch
