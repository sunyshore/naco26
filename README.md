# Niching-based Evolutionary Feature Selection for Animal Sound Classification

We build a feature selection pipeline for wildlife audio classification on the BirdCLEF+ 2026 dataset, with attention to rare species that standard methods tend to miss.

---

## Random Forest Experiment

### What's in the repo

  - `src/` — Python package with the pipeline:
  - `data.py` — builds the species manifest and train/val/test splits
  - `features.py` — extracts audio features using librosa
  - `baselines.py` — all-features and mRMR baselines, Random Forest training
  - `ec.py` — genetic algorithm with fitness sharing and weighted fitness
  - `evaluate.py` — evaluation on the test set, committee voting, plotting
  - `main.ipynb` — notebook that runs the full pipeline end to end

### Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Register a Jupyter kernel if using VS Code or JupyterLab:

```bash
python -m ipykernel install --user --name naco --display-name "Python (NACO)"
```

### Pointing at the dataset

The BirdCLEF+ 2026 dataset is not included in this repository. To obtain it, create a Kaggle account, accept the competition rules at https://www.kaggle.com/competitions/birdclef-2026, and download the dataset.

Place it so the folder structure looks like:

```
data/
  raw/
    train.csv
    train_audio/
```

Or set an environment variable pointing to wherever you placed it:

```bash
export BIRDCLEF_DIR=/path/to/your/birdclef-2026
```

On Windows use:

```
set BIRDCLEF_DIR=C:\path\to\your\birdclef-2026
```

### Running the notebook

Open the project folder in VS Code or launch JupyterLab. Open `main.ipynb` and pick the `Python (NACO)` kernel. Run all cells top to bottom.

The notebook runs five stages automatically:

| Stage | What it does |
|-------|-------------|
| 1 | Builds the species manifest from `train.csv` |
| 2 | Extracts audio features and caches to `data/features.pkl` |
| 3 | Runs the all-features and mRMR baselines |
| 4 | Runs the GA experiments across all seeds |
| 5 | Generates the results table and comparison figure |

Feature extraction is cached after the first run. To change the number of seeds or the sharing threshold, edit the config cell at the top of Stage 4:

```python
SEEDS = [42, 43, 44, 45, 46]
THRESHOLDS = [0.9]
```

Results are saved to `results/runs/<RUN_NAME>/` and include:
- `results.csv` — per-seed metrics for all methods
- `summary.csv` — mean and std across seeds
- `config.json` — full run configuration
- `histories.pkl` — per-generation fitness and diversity curves

### Sample output

After a completed 5-seed run the final results table shows:

| Method | Accuracy | Macro-F1 |
|--------|----------|----------|
| All features | 0.826 ± 0.010 | 0.657 ± 0.043 |
| mRMR top-15 | 0.778 ± 0.016 | 0.599 ± 0.057 |
| Plain EC, best | 0.809 ± 0.025 | 0.705 ± 0.025 |
| Plain EC, committee | 0.810 ± 0.006 | 0.691 ± 0.031 |
| Niched EC, best | 0.810 ± 0.008 | 0.676 ± 0.030 |
| Niched EC, committee | 0.828 ± 0.007 | 0.698 ± 0.032 |
| Niched EC, weighted best | 0.836 ± 0.009 | 0.703 ± 0.032 |
| **Niched EC, weighted committee** | **0.844 ± 0.015** | **0.711 ± 0.049** |

The comparison figure is saved to `results/figures/comparison.png`.

---

## MLP Experiment

To run, import any of the MLP notebooks directly into a Kaggle environment, then add the BirdCLEF+ 2026 data. For the 19-species MLP, ensure the T4 GPU accelerator is selected.

- `MLP-5` — 5 species experiment
- `MLP-19` — 19 species experiment

The entire notebook should run fully. The following output denotes the start of evolutionary computation:

```
Inner classifier swapped to PyTorch MLPClassifier(16,) solver=Adam max_iter=200
Using GPU: True (device: cuda)
Run name: kaggle_mlp_geo_temporal_pytorch_n10_seeds5_thresh2_20260612_011424
Run dir : /kaggle/working/results/runs/kaggle_mlp_geo_temporal_pytorch_n10_seeds5_thresh2_20260612_011424
=== Seed 42 ===
  baselines...
```

![gpu and data settings](./images/gpu.png)
