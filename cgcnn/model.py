from __future__ import print_function, division

import warnings
import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: torch.Tensor shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: torch.Tensor shape (N, atom_fea_len)
          Atom hidden features after convolution
        """
        N, M = nbr_fea_idx.shape

        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, H)
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea
            ],
            dim=2
        )  # (N, M, 2H+E)

        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)

        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CGCNNChargeTransferLayer(nn.Module):
    """
    Charge-transfer layer adapted for CGCNN tensor format.

    Inputs
    ------
    x : (N, H)
        Atom hidden features
    z : (N,)
        Atomic numbers
    nbr_fea : (N, M, E)
        Neighbor bond features
    nbr_fea_idx : (N, M)
        Neighbor indices
    crystal_atom_idx : list[LongTensor]
        Mapping crystal -> atom indices
    nbr_mask : (N, M), optional
        Valid neighbor mask, 1/True for real neighbors, 0/False for padded neighbors
    """
    def __init__(
        self,
        hidden_channels,
        edge_feat_channels,
        num_elements=118,
        chi0=0.4,
        tau=0.15,
        q_clip=2.5,
        learnable_chi=True,
        z_index_mode="atomic_number",
    ):
        super(CGCNNChargeTransferLayer, self).__init__()
        self.hidden_channels = int(hidden_channels)
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
                "CGCNNChargeTransferLayer: using fallback electronegativity "
                "table with learnable_chi=False.",
                stacklevel=2,
            )

        self.register_buffer("chi0", torch.tensor(float(chi0)))
        self.register_buffer("tau", torch.tensor(float(tau)))

        in_dim = hidden_channels * 2 + edge_feat_channels + 3
        hidden_mid = max(hidden_channels // 2, 16)

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_channels),
            nn.Softplus(),
            nn.Linear(hidden_channels, hidden_mid),
            nn.Softplus(),
            nn.Linear(hidden_mid, 1),
        )

        self.node_fuse = nn.Sequential(
            nn.Linear(hidden_channels + 2, hidden_channels),
            nn.Softplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # zero-init for stable start
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)
        nn.init.zeros_(self.node_fuse[-1].weight)
        nn.init.zeros_(self.node_fuse[-1].bias)

    @staticmethod
    def _build_pauling_table(num_elements):
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

    def _z_to_index(self, z):
        z = z.long().view(-1)
        if self.z_index_mode == "atomic_number":
            return z.clamp(1, self.num_elements)
        elif self.z_index_mode == "zero_based":
            return (z + 1).clamp(1, self.num_elements)
        else:
            raise ValueError(
                "Invalid z_index_mode={}, expected 'atomic_number' or 'zero_based'.".format(
                    self.z_index_mode
                )
            )

    @staticmethod
    def _build_batch_from_crystal_atom_idx(crystal_atom_idx, N, device):
        batch = torch.empty(N, dtype=torch.long, device=device)
        for c, idx in enumerate(crystal_atom_idx):
            batch[idx.to(device)] = c
        return batch

    @staticmethod
    def _neutralize_per_crystal(q, crystal_atom_idx, device):
        q = q.clone()
        for idx in crystal_atom_idx:
            idx = idx.to(device)
            q[idx] = q[idx] - q[idx].mean()
        return q

    def forward(self, x, z, nbr_fea, nbr_fea_idx, crystal_atom_idx, nbr_mask=None):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (N, H)
        z : torch.LongTensor, shape (N,)
        nbr_fea : torch.Tensor, shape (N, M, E)
        nbr_fea_idx : torch.LongTensor, shape (N, M)
        crystal_atom_idx : list[torch.LongTensor]
        nbr_mask : torch.BoolTensor or torch.Tensor, shape (N, M), optional

        Returns
        -------
        x_ct : torch.Tensor, shape (N, H)
        q : torch.Tensor, shape (N,)
        stats : dict
        """
        N, M = nbr_fea_idx.shape
        device = x.device
        dtype = x.dtype

        z_idx = self._z_to_index(z)
        chi = self.chi(z_idx).squeeze(-1).to(dtype=dtype)   # (N,)

        x_i = x.unsqueeze(1).expand(N, M, x.size(-1))       # (N, M, H)
        x_j = x[nbr_fea_idx, :]                             # (N, M, H)

        chi_i = chi.unsqueeze(1).expand(N, M)               # (N, M)
        chi_j = chi[nbr_fea_idx]                            # (N, M)
        dchi = chi_j - chi_i                                # neighbor -> center

        gate = torch.sigmoid(
            (dchi.abs() - self.chi0.to(dtype)) / (self.tau.to(dtype) + 1e-8)
        )                                                   # (N, M)

        edge_in = torch.cat(
            [
                x_i,
                x_j,
                nbr_fea.to(dtype),
                dchi.unsqueeze(-1),
                chi_i.unsqueeze(-1),
                chi_j.unsqueeze(-1),
            ],
            dim=-1,
        )                                                   # (N, M, 2H+E+3)

        dq_ji = self.edge_mlp(edge_in).squeeze(-1)          # (N, M)

        if nbr_mask is not None:
            mask = nbr_mask.to(dtype=dtype, device=device)
            gate = gate * mask
            dq_ji = dq_ji * mask

        q_raw = torch.sum(gate * dq_ji, dim=1)              # (N,)

        batch = self._build_batch_from_crystal_atom_idx(
            crystal_atom_idx, N, device
        )

        q = self._neutralize_per_crystal(q_raw, crystal_atom_idx, device)

        if self.q_clip > 0:
            q = self.q_clip * torch.tanh(q / self.q_clip)

        q = self._neutralize_per_crystal(q, crystal_atom_idx, device)

        x_delta = self.node_fuse(
            torch.cat([x, q.unsqueeze(-1), chi.unsqueeze(-1)], dim=-1)
        )
        x_ct = x + x_delta

        q_sum_per_crystal = []
        for idx in crystal_atom_idx:
            idx = idx.to(device)
            q_sum_per_crystal.append(q[idx].sum())
        q_sum_per_crystal = torch.stack(q_sum_per_crystal, dim=0)

        stats = {
            "chi_atom": chi,
            "q_pred": q,
            "q_sum_per_crystal": q_sum_per_crystal,
            "q_neutral_l1": q_sum_per_crystal.abs().mean().detach(),
            "q_neutral_max": q_sum_per_crystal.abs().max().detach(),
            "gate_mean": gate.mean().detach(),
            "gate_saturated_frac": ((gate > 0.99) | (gate < 0.01)).float().mean().detach(),
            "q_abs_mean": q.abs().mean().detach(),
            "batch_index": batch,
        }
        return x_ct, q, stats


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties, with optional charge-transfer module.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False,
                 use_charge_transfer=True,
                 ct_insert_after=None,
                 num_elements=118,
                 ct_chi0=0.4,
                 ct_tau=0.15,
                 ct_q_clip=2.5,
                 ct_learnable_chi=True,
                 ct_z_index_mode="atomic_number"):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers.
        n_conv: int
          Number of convolutional layers.
        h_fea_len: int
          Number of hidden features after pooling.
        n_h: int
          Number of hidden layers after pooling.
        classification: bool
          Classification or regression.
        use_charge_transfer: bool
          Whether to enable charge-transfer block.
        ct_insert_after: int or None
          Insert CT after this conv layer index (1-based). If None, defaults to n_conv.
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.use_charge_transfer = bool(use_charge_transfer)

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])

        if ct_insert_after is None:
            ct_insert_after = n_conv
        ct_insert_after = int(ct_insert_after)
        self.ct_insert_after = max(1, min(max(1, n_conv), ct_insert_after))

        if self.use_charge_transfer:
            self.charge_transfer = CGCNNChargeTransferLayer(
                hidden_channels=atom_fea_len,
                edge_feat_channels=nbr_fea_len,
                num_elements=num_elements,
                chi0=ct_chi0,
                tau=ct_tau,
                q_clip=ct_q_clip,
                learnable_chi=ct_learnable_chi,
                z_index_mode=ct_z_index_mode,
            )

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList([
                nn.Linear(h_fea_len, h_fea_len)
                for _ in range(n_h - 1)
            ])
            self.softpluses = nn.ModuleList([
                nn.Softplus()
                for _ in range(n_h - 1)
            ])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
                z=None, nbr_mask=None, return_ct_stats=False):
        """
        Forward pass

        Parameters
        ----------
        atom_fea : torch.Tensor shape (N, orig_atom_fea_len)
        nbr_fea : torch.Tensor shape (N, M, nbr_fea_len)
        nbr_fea_idx : torch.LongTensor shape (N, M)
        crystal_atom_idx : list of torch.LongTensor
        z : torch.LongTensor shape (N,), optional
          Atomic numbers. Required if use_charge_transfer=True.
        nbr_mask : torch.BoolTensor shape (N, M), optional
          Valid neighbor mask.
        return_ct_stats : bool
          Whether to return charge-transfer diagnostics.

        Returns
        -------
        out : torch.Tensor
          Prediction tensor.
        or
        out, ct_stats : tuple
          If return_ct_stats=True and CT is enabled.
        """
        atom_fea = self.embedding(atom_fea)

        q_pred = None
        ct_stats = None
        ct_inserted = False

        for i, conv_func in enumerate(self.convs):
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

            if self.use_charge_transfer and (not ct_inserted) and ((i + 1) == self.ct_insert_after):
                if z is None:
                    raise ValueError("z must be provided when use_charge_transfer=True")
                atom_fea, q_pred, ct_stats = self.charge_transfer(
                    atom_fea, z, nbr_fea, nbr_fea_idx, crystal_atom_idx, nbr_mask=nbr_mask
                )
                ct_inserted = True

        if self.use_charge_transfer and (not ct_inserted):
            if z is None:
                raise ValueError("z must be provided when use_charge_transfer=True")
            atom_fea, q_pred, ct_stats = self.charge_transfer(
                atom_fea, z, nbr_fea, nbr_fea_idx, crystal_atom_idx, nbr_mask=nbr_mask
            )

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)

        if return_ct_stats and (ct_stats is not None):
            return out, ct_stats
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pool the atom features to crystal features.

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------
        torch.Tensor shape (N0, atom_fea_len)
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.data.shape[0]
        summed_fea = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]
        return torch.cat(summed_fea, dim=0)
