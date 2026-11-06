"""Generate Simulation I data (50 signal coordinates plus Gaussian noise).

The manuscript uses n=800; --n changes only the sample size. Saved X.npy
is column-standardized using sample SDs, matching the experiment pipeline.
Labels are drawn once with equal probabilities, not forced to be balanced.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def generate_data(n=800, p=1000, n_clusters=4, gamma=0.4, seed=123):
    if n_clusters < 2 or n < n_clusters:
        raise ValueError("Require n >= n_clusters >= 2.")
    if p < 50:
        raise ValueError("Simulation I requires p >= 50.")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive.")
    if seed < 0:
        raise ValueError("seed must be nonnegative.")

    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_clusters, size=n, dtype=np.int64)
    alpha = np.pi / 4.0 + 2.0 * np.pi * np.arange(
        n_clusters, dtype=np.float64
    ) / float(n_clusters)
    means = np.empty((n_clusters, 50), dtype=np.float64)
    means[:, :25] = math.sqrt(2.0) * gamma * np.cos(alpha)[:, None]
    means[:, 25:] = math.sqrt(2.0) * gamma * np.sin(alpha)[:, None]
    x_raw = rng.normal(0.0, 1.0, size=(n, p))
    x_raw[:, :50] += means[y]
    column_mean = x_raw.mean(axis=0, keepdims=True)
    column_sd = x_raw.std(axis=0, ddof=1, keepdims=True)
    if not np.isfinite(column_sd).all() or np.any(column_sd <= 0):
        raise ValueError("Generated data contain an invalid column SD.")
    x = ((x_raw - column_mean) / column_sd).astype(np.float32)
    description = {
        "design": "Simulation I mechanism; manuscript reference n=800",
        "n": n, "p": p, "K": n_clusters, "gamma": gamma,
        "data_seed": seed,
        "rng": "numpy.random.Generator(PCG64)",
        "numpy_version": np.__version__,
        "labels": "iid Categorical(1/K,...,1/K), saved as 0,...,K-1; no resampling",
        "class_counts": np.bincount(y, minlength=n_clusters).tolist(),
        "signal_columns_zero_based": [0, 49],
        "signal_mean_formula": "sqrt(2)*gamma*[cos(alpha_k)*1_25, sin(alpha_k)*1_25]; alpha_k=pi/4+2*pi*k/K for zero-based k",
        "raw_conditional_covariance": "I_p; remaining p-50 coordinates are independent N(0,1) noise",
        "preprocessing": "column z-score using all n observations and sample SD (ddof=1)",
        "saved_dtypes": {"X.npy": "float32", "y.npy": "int64"},
    }
    return x, y, description


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--p", type=int, default=1000)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    try:
        x, y, description = generate_data(
            args.n, args.p, args.n_clusters, args.gamma, args.seed
        )
    except ValueError as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "X.npy", x, allow_pickle=False)
    np.save(output / "y.npy", y, allow_pickle=False)
    (output / "description.json").write_text(
        json.dumps(description, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Simulation I data saved to {output}")
    print(f"X shape: {x.shape}; class counts: {description['class_counts']}")


if __name__ == "__main__":
    main()
