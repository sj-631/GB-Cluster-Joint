# GB Cluster–Joint

Generalized Bayesian clustering with an autoencoder: alternate neural-network training and Gibbs partition updates, then summarize posterior clustering and uncertainty.

## Install

Use Python 3.10+ from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Default training (including Simulation I) requires R with `Rscript` on `PATH` for Hartigan–Wong initialization and SALSO estimation. Install these R packages:

```r
install.packages(c("salso", "RcppCNPy", "jsonlite", "digest"))
```

## Simulation I

Generate data using the Simulation I mechanism, then train:

```sh
python examples/generate_simulation1_data.py \
  --output-dir data/simulation1 --n 800 --p 1000 \
  --n-clusters 4 --gamma 0.4 --seed 123
python train.py \
  --preset presets/simulation1_current.json \
  --x-path data/simulation1/X.npy --y-path data/simulation1/y.npy \
  --output-dir outputs/joint_seed125 \
  --n-clusters 4 --seed 125 --device cpu
```

The generator standardizes columns using sample standard deviations (`ddof=1`). For your own data, supply a finite numeric `X.npy` with observations in rows and the same preprocessing. Optional `y.npy` labels are for evaluation only; omit `--y-path` when unavailable.

Defaults (CLI and Simulation I preset):

- Linear autoencoder `p → 6 → p`; Adam; batch size 64; CPU; latent scale projection enabled.
- 100 pretraining epochs and 100 joint iterations; learning rates `0.005 / 0.001`.
- Lambda is calibrated at pretraining epoch 10 (`r_0=1`) using K-means on the current latent representation, then fixed. Pretraining continues through epoch 100.
- After full pretraining, latent Hartigan–Wong (100 starts) initializes Gibbs. Initial eta uses `lambda_R * R + C`, with multiplier 1.
- Each training iteration uses `B=10` Gibbs sweeps and retains the last `M=5`: sweeps 6–10.
- Final sampling: 500 draws, no extra burn-in; SALSO under VI loss.

