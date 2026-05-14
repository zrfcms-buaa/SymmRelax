from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False,
                              train_sampler=None,
                              **kwargs):

    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print('[Warning] train_ratio is None, using 1 - val_ratio - '
                  'test_ratio = {} as training data.'.format(train_ratio))
        else:
            assert train_ratio + val_ratio + test_ratio <= 1

    indices = list(range(total_size))

    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)

    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)

    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)

    train_indices = indices[:train_size]
    val_indices = indices[-(valid_size + test_size):-test_size]
    test_indices = indices[-test_size:]

    if train_sampler is not None:
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn, pin_memory=pin_memory)
    else:
        train_sampler_default = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler_default,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn, pin_memory=pin_memory)

    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)

    if return_test:
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)

    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal properties.

    Parameters
    ----------
    dataset_list: list of tuples for each data point.
      ((atom_fea, nbr_fea, nbr_fea_idx, z, nbr_mask), target, cif_id)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      z: torch.LongTensor shape (n_i,)
      nbr_mask: torch.BoolTensor shape (n_i, M)
      target: torch.Tensor shape (1,)
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = number of crystals in batch

    inputs: tuple
      batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      crystal_atom_idx: list of torch.LongTensor of length N0
      batch_z: torch.LongTensor shape (N,)
      batch_nbr_mask: torch.BoolTensor shape (N, M)

    target: torch.Tensor shape (N0, 1)
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    batch_z, batch_nbr_mask = [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []

    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, z, nbr_mask), target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]

        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        batch_z.append(z)
        batch_nbr_mask.append(nbr_mask)

        new_idx = torch.arange(n_i, dtype=torch.long) + base_idx
        crystal_atom_idx.append(new_idx)

        batch_target.append(target)
        batch_cif_ids.append(cif_id)

        base_idx += n_i

    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_fea_idx, dim=0),
        crystal_atom_idx,
        torch.cat(batch_z, dim=0),
        torch.cat(batch_nbr_mask, dim=0),
    ), torch.stack(batch_target, dim=0), batch_cif_ids


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        Parameters
        ----------
        distances: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomInitializer(object):
    """
    Base class for initializing the vector representation for atoms.

    Use one AtomInitializer per dataset.
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping element number -> feature vector.
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {
            int(key): value for key, value in elem_embedding.items()
        }
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    └── ...

    id_prop.csv: a CSV file with two columns:
        1) unique crystal ID
        2) target property

    atom_init.json: maps element number -> initial atom feature vector

    Returns
    -------
    (atom_fea, nbr_fea, nbr_fea_idx, z, nbr_mask), target, cif_id
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=12, dmin=0, step=0.2,
                 random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius

        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'

        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'

        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + '.cif'))

        # atom features
        atom_fea = np.vstack([
            self.ari.get_atom_fea(crystal[i].specie.number)
            for i in range(len(crystal))
        ])
        atom_fea = torch.tensor(atom_fea, dtype=torch.float)

        # atomic numbers for charge-transfer module
        z = torch.tensor(
            [crystal[i].specie.number for i in range(len(crystal))],
            dtype=torch.long
        )

        # neighbors
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea, nbr_mask = [], [], []

        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    '{} not find enough neighbors to build graph. '
                    'If it happens frequently, consider increase radius.'.format(cif_id)
                )
                valid_n = len(nbr)

                nbr_idx_row = list(map(lambda x: x[2], nbr)) + \
                              [0] * (self.max_num_nbr - valid_n)

                nbr_dist_row = list(map(lambda x: x[1], nbr)) + \
                               [self.radius + 1.] * (self.max_num_nbr - valid_n)

                nbr_mask_row = [1] * valid_n + [0] * (self.max_num_nbr - valid_n)
            else:
                nbr = nbr[:self.max_num_nbr]
                nbr_idx_row = list(map(lambda x: x[2], nbr))
                nbr_dist_row = list(map(lambda x: x[1], nbr))
                nbr_mask_row = [1] * self.max_num_nbr

            nbr_fea_idx.append(nbr_idx_row)
            nbr_fea.append(nbr_dist_row)
            nbr_mask.append(nbr_mask_row)

        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)
        nbr_mask = np.array(nbr_mask)

        nbr_fea = self.gdf.expand(nbr_fea)

        nbr_fea = torch.tensor(nbr_fea, dtype=torch.float)
        nbr_fea_idx = torch.tensor(nbr_fea_idx, dtype=torch.long)
        nbr_mask = torch.tensor(nbr_mask, dtype=torch.bool)
        target = torch.tensor([float(target)], dtype=torch.float)

        return (atom_fea, nbr_fea, nbr_fea_idx, z, nbr_mask), target, cif_id
