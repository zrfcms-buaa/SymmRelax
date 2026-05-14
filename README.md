# SymmRelax

SymmRelax is a symmetry-preserving crystal pre-relaxation framework for high-throughput materials discovery.

It uses a symmetry-aware graph neural network to preserve crystallographic constraints during crystal structure pre-relaxation. During inference, an optional CGCNN energy predictor can be used for candidate reranking.

## Workflow

```text
paired CIF files
    ↓
preprocess_sqlite.py
    ↓
SQLite dataset
    ↓
train.py
    ↓
SymmRelax geometry model
    ↓
infer_symmrelax.py
    ↓
relaxed CIF predictions
    ↓
evaluate_structures.py
```

## Project Structure

```text
SymmRelax/
├── configs/
│   ├── train_config.json
│   └── infer_config.json
├── cgcnn/
│   ├── data.py
│   └── model.py
├── _cgcnn_tmp/
│   └── atom_init.json             # required when CGCNN energy prediction is enabled
├── crystal_data.py                # SQLite dataset and PyG batch construction
├── symm_graph.py                  # crystal graph construction and graph utilities
├── SymmRelax.py                   # model definition
├── preprocess_sqlite.py           # preprocess paired CIF files into SQLite dataset
├── train.py                       # training script
├── infer_symmrelax.py             # inference script
├── evaluate_structures.py         # optional evaluation script
├── README.md
└── .gitignore
```

Large data and model files are not stored directly in the Git repository. They are provided through GitHub Releases.

## Release Assets

Download the following files from GitHub Releases:

```text
binary_data.tar.gz
ckpt_symmrelax.pt
model_best.pth.tar
```

After downloading, place them as follows:

```text
SymmRelax/
├── data/
│   └── binary_data.tar.gz
└── models/
    ├── ckpt_symmrelax.pt
    └── model_best.pth.tar
```

If the folders do not exist, create them first:

```bash
mkdir -p data models
```

## Environment

The code was tested with:

```text
Python 3.9.21
PyTorch 2.5.1 + CUDA 12.1
PyTorch Geometric 2.6.1
```

Recommended conda environment:

```bash
conda create -n symmrelax python=3.9 -y
conda activate symmrelax

conda install pytorch==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg=2.6.1 -c pyg

conda install -c conda-forge \
  numpy scipy pandas tqdm joblib \
  ase pymatgen spglib
```

Core Python dependencies:

```text
numpy
scipy
pandas
tqdm
joblib
ase
pymatgen
spglib
torch
torch-geometric
torch-scatter
```

Optional dependency:

```text
cgcnn
```

`cgcnn` is required only when CGCNN energy prediction is enabled in `configs/infer_config.json`.

## Data Preparation

The paired CIF files are provided as:

```text
data/binary_data.tar.gz
```

The archive contains two folders:

```text
INITIAL/
RELAX/
```

Extract the archive into `binary_data/`:

```bash
mkdir -p binary_data
tar -xzf data/binary_data.tar.gz -C binary_data
```

After extraction, the directory should look like:

```text
binary_data/
├── INITIAL/
│   ├── sample_001.cif
│   ├── sample_002.cif
│   └── ...
└── RELAX/
    ├── sample_001.cif
    ├── sample_002.cif
    └── ...
```

The file names in `INITIAL/` and `RELAX/` should match.

Then run preprocessing:

```bash
python preprocess_sqlite.py \
  --data_root binary_data \
  --out_name processed_sqlite \
  --num_workers 8
```

This will generate the processed SQLite dataset under:

```text
binary_data/processed_sqlite/
```

Typical generated files include:

```text
data.sqlite
meta.csv
split_train.npy
split_val.npy
split_test.npy
```

## Training

Run distributed training with 8 GPUs:

```bash
torchrun --nproc_per_node=8 train.py \
  --root binary_data/processed_sqlite \
  --out models/ckpt_symmrelax.pt \
  --config configs/train_config.json \
  --steps_per_epoch 800 \
  --epochs1 50 \
  --epochs2 70
```

The geometry model checkpoint will be saved as:

```text
models/ckpt_symmrelax.pt
```

## Inference

Run inference using the trained SymmRelax geometry checkpoint:

```bash
python infer_symmrelax.py \
  --root binary_data/processed_sqlite \
  --geom_ckpt models/ckpt_symmrelax.pt \
  --out_dir infer_symmrelax \
  --config configs/infer_config.json
```

The inference script reads the optional CGCNN energy model path from:

```text
configs/infer_config.json
```

## Energy Model for Inference

The CGCNN energy model is optional. If energy prediction is enabled, the inference script depends on:

```text
cgcnn/
models/model_best.pth.tar
_cgcnn_tmp/atom_init.json
```

The recommended model directory is:

```text
models/
├── ckpt_symmrelax.pt
└── model_best.pth.tar
```

In `configs/infer_config.json`, enable the energy predictor like this:

```json
{
  "energy": {
    "enabled": true,
    "energy_ckpt": "models/model_best.pth.tar",
    "cgcnn_tmp_root": "_cgcnn_tmp",
    "energy_device": "cuda:0",
    "energy_max_num_nbr": 12,
    "energy_radius": 12.0,
    "beta_E": 0.8,
    "energy_topM": 2,
    "energy_only_tiebreak": true,
    "energy_tiebreak_margin": 0.25
  }
}
```

To disable the energy predictor:

```json
{
  "energy": {
    "enabled": false,
    "energy_ckpt": ""
  }
}
```

When `energy.enabled` is `true`, make sure:

```text
cgcnn/ exists and can be imported
models/model_best.pth.tar exists
_cgcnn_tmp/atom_init.json exists
```

The inference command does not need an extra `--energy_ckpt` argument because the energy checkpoint path is read from `configs/infer_config.json`.

## Inference Output

The inference results will be saved under:

```text
infer_symmrelax/
```

Typical output files and folders include:

```text
infer_symmrelax/
├── init/
├── pred/
├── dft/
├── summary.json
└── per_sample.csv
```

## Evaluation

After inference, predicted structures can be evaluated against the reference relaxed structures:

```bash
python evaluate_structures.py \
  --pred_name infer_symmrelax/pred \
  --dft_name infer_symmrelax/dft \
  --meta_csv binary_data/processed_sqlite/meta.csv \
  --split_npy binary_data/processed_sqlite/split_test.npy \
  --per_sample_csv infer_symmrelax/per_sample.csv \
  --out_csv infer_symmrelax/eval.csv \
  --summary_json infer_symmrelax/eval_summary.json
```

## Recommended `.gitignore`

```gitignore
__pycache__/
*.pyc

binary_data/
data/*.tar.gz

models/*.pt
models/*.pth
models/*.pth.tar

*.sqlite

infer_symmrelax/

_cgcnn_tmp/*
!_cgcnn_tmp/atom_init.json
```

## Notes

- `preprocess_sqlite.py` converts paired CIF files into a SQLite dataset.
- `train.py` trains the SymmRelax geometry model.
- `infer_symmrelax.py` performs structure relaxation inference.
- `infer_symmrelax.py` can optionally use a CGCNN energy predictor for candidate reranking.
- `evaluate_structures.py` computes optional structure-level evaluation metrics.
- `configs/` stores JSON configuration files.
- `models/` stores downloaded or trained model checkpoints.
- `data/` stores the downloaded dataset archive.
