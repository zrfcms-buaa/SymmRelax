# SymmRelax

SymmRelax is a symmetry-preserving crystal pre-relaxation framework for high-throughput materials discovery.

This repository provides the source code for the manuscript:

**Symmetry Preserving Pre-Relaxation Framework for High-Throughput and Reliable Crystal Screening**

SymmRelax refines unrelaxed crystal structures while preserving crystallographic constraints. It predicts crystal-system-constrained lattice parameters and Wyckoff-compatible representative coordinates, followed by exact symmetry expansion and geometry refinement.

## Data, checkpoints and source data

Large data files and trained model checkpoints are not stored directly in this Git repository. They are available from Zenodo:

**Zenodo:** https://doi.org/10.5281/zenodo.20456970

The Zenodo release contains:

```text
processed_sqlite.tar.xz
SourceData.zip
ckpt_symmrelax.pt
model_best.pth.tar
README_Data.txt
```

where:

- `processed_sqlite.tar.xz` contains the processed SQLite graph dataset used for SymmRelax training, validation and testing.
- `SourceData.zip` contains source data underlying the figures, tables, Extended Data and selected Supplementary Tables.
- `ckpt_symmrelax.pt` is the trained SymmRelax geometry checkpoint.
- `model_best.pth.tar` is the trained checkpoint used for the downstream or optional energy-prediction workflow.

## Repository structure

```text
SymmRelax/
├── configs/
│   ├── train_config.json
│   └── infer_config.json
├── cgcnn/
│   ├── data.py
│   └── model.py
├── _cgcnn_tmp/
│   └── atom_init.json
├── crystal_data.py
├── symm_graph.py
├── SymmRelax.py
├── preprocess_sqlite.py
├── train.py
├── infer_symmrelax.py
├── evaluate_structures.py
├── README.md
└── .gitignore
```

## Environment

The code was tested with:

```text
Python 3.9.21
PyTorch 2.5.1 + CUDA 12.1
PyTorch Geometric 2.6.1
```

A recommended conda environment is:

```bash
conda create -n symmrelax python=3.9 -y
conda activate symmrelax

conda install pytorch==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg=2.6.1 -c pyg

conda install -c conda-forge \
  numpy scipy pandas tqdm joblib \
  ase pymatgen spglib
```

Core dependencies:

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

## Preparing the processed dataset

Download `processed_sqlite.tar.xz` from the Zenodo release and place it under `data/`:

```bash
mkdir -p data models
```

Then extract it:

```bash
tar -xJf data/processed_sqlite.tar.xz -C data
```

After extraction, the dataset directory should look like:

```text
data/processed_sqlite/
├── data.sqlite
├── meta.csv
├── skip_summary.csv
├── split_train.npy
├── split_val.npy
└── split_test.npy
```

## Training

Run distributed training with 8 GPUs:

```bash
torchrun --nproc_per_node=8 train.py \
  --root data/processed_sqlite \
  --out models/ckpt_symmrelax.pt \
  --config configs/train_config.json \
  --steps_per_epoch 800 \
  --epochs1 50 \
  --epochs2 70
```

The trained geometry checkpoint will be saved as:

```text
models/ckpt_symmrelax.pt
```

## Inference

Download `ckpt_symmrelax.pt` from Zenodo and place it under:

```text
models/ckpt_symmrelax.pt
```

Run inference:

```bash
python infer_symmrelax.py \
  --root data/processed_sqlite \
  --geom_ckpt models/ckpt_symmrelax.pt \
  --out_dir infer_symmrelax \
  --config configs/infer_config.json
```

Typical inference outputs are:

```text
infer_symmrelax/
├── init/
├── pred/
├── dft/
├── summary.json
└── per_sample.csv
```

## Optional energy-prediction workflow

The inference workflow can optionally use an additional trained checkpoint for candidate reranking. Download `model_best.pth.tar` from Zenodo and place it under:

```text
models/model_best.pth.tar
```

The energy-prediction settings are controlled in:

```text
configs/infer_config.json
```

To disable this optional component, set:

```json
{
  "energy": {
    "enabled": false,
    "energy_ckpt": ""
  }
}
```

## Evaluation

After inference, predicted structures can be evaluated against DFT-relaxed references:

```bash
python evaluate_structures.py \
  --pred_name infer_symmrelax/pred \
  --dft_name infer_symmrelax/dft \
  --meta_csv data/processed_sqlite/meta.csv \
  --split_npy data/processed_sqlite/split_test.npy \
  --per_sample_csv infer_symmrelax/per_sample.csv \
  --out_csv infer_symmrelax/eval.csv \
  --summary_json infer_symmrelax/eval_summary.json
```

## Preprocessing from paired structures

The processed SQLite dataset can also be regenerated from paired initial and DFT-relaxed structures when such paired structures are available locally. The expected input layout is:

```text
binary_data/
├── INITIAL/
└── RELAX/
```

Run:

```bash
python preprocess_sqlite.py \
  --data_root binary_data \
  --out_name processed_sqlite \
  --num_workers 8
```

This generates:

```text
binary_data/processed_sqlite/
├── data.sqlite
├── meta.csv
├── skip_summary.csv
├── split_train.npy
├── split_val.npy
└── split_test.npy
```

## Source data

Source data underlying the figures, tables, Extended Data and selected Supplementary Tables are provided in the Zenodo release as:

```text
SourceData.zip
```

Fig. 2 is a conceptual schematic and has no associated numerical source data.

## License

The source code is released under the license provided in this repository. The processed dataset, trained checkpoints and source data are available through the Zenodo release.
