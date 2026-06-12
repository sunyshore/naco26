# naco26

## MLP
To run, import any of the MLP notebooks directly into a Kaggle environment, then add the BirdCLEF+ 2026 data. For 19-species MLP, ensure the T4 GPU accelerator is selected.
- MLP-5 is the 5 species experiment
- MLP-19 is the 19 species experiment
- MLPGEO-19 is the 19 species experiment with geotemporal features

The entire notebook should run fully; the following output denotes the start of evolutionary computation.
```
Inner classifier swapped to PyTorch MLPClassifier(16,) solver=Adam max_iter=200
Using GPU: True (device: cuda)

Run name: kaggle_mlp_geo_temporal_pytorch_n10_seeds5_thresh2_20260612_011424
Run dir : /kaggle/working/results/runs/kaggle_mlp_geo_temporal_pytorch_n10_seeds5_thresh2_20260612_011424

=== Seed 42 ===
  baselines...
```

![gpu and data settings](./images/gpu.png)
