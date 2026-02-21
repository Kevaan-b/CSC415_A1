# CSC415 A1: DG via ER Reproduction

This repository reproduces part of:

- Paper: Zhao et al. (2020) - Domain Generalization via Entropy Regularization (NeurIPS)
- Original code: https://github.com/sshan-zhao/DG_via_ER
- Dataset mirror used for PACS: https://github.com/MachineLearning2020/Homework3-PACS

## 1) Clone this repo and install dependencies

```bash
git clone https://github.com/Kevaan-b/CSC415_A1.git
cd CSC415_A1
pip install -r requirements.txt
```

## 2) Add the PACS dataset

For convenience, I use the PACS mirror repo above.

From the repo root:

```bash
git clone --depth 1 https://github.com/MachineLearning2020/Homework3-PACS.git
mkdir -p dataset
cp -a Homework3-PACS/PACS dataset/PACS
```

Expected structure:

```text
dataset/PACS/art_painting/...
dataset/PACS/cartoon/...
dataset/PACS/photo/...
dataset/PACS/sketch/...
```

## 3) Run experiments

```bash
python a1_experiments.py
```

This script runs the configured reproduction and ablation batches and writes logs/results under experiment folders.

## File Summary

### What I kept the same from the original repo

- `models/__init__.py`
- `models/aux_models.py`
- `models/model_factory.py`
- `utils.py`
- `datalist/PACS/*`

### What I modified due to version differences

- `models/resnet.py`
- `train.py`
- `data/dataset.py`

### What I added for convenience of running experiments

- `a1_experiments.py`: Runs the reproduction and no-augmentation ablation loops and organizes outputs by run folder.
- `tsne_plot.py`: Creates domain t-SNE plots from saved checkpoints.
- `requirements.txt`: pip dependencies used for local setup.
- `precomputed_results/`: Stores run details of experiments ran before hand without model checkpoints.
