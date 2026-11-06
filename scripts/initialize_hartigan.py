"""Create a Hartigan--Wong partition of supplied features, without reference labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--x-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--seed", type=int, default=125)
    parser.add_argument("--n-starts", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--standardize", choices=["none", "zscore"], default="none")
    args = parser.parse_args(argv)
    if min(args.n_clusters, args.n_starts, args.max_iterations) < 1:
        parser.error("Cluster, restart, and iteration counts must be positive.")
    if not 0 <= args.seed <= np.iinfo(np.int32).max:
        parser.error("Seed must be a non-negative signed 32-bit integer.")
    source = args.x_path.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output directory: {output}")
    x = np.load(source, allow_pickle=False)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        parser.error("X must be a finite nonempty two-dimensional numeric array.")
    if not 2 <= args.n_clusters < len(x):
        parser.error("Hartigan--Wong requires 2 <= n_clusters < n_observations.")
    if args.standardize == "zscore":
        mean, scale = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        scale = np.where(scale < 1e-8, 1.0, scale)
        x = (x - mean) / scale
    # Match the float32 features used by the trainer, then transfer exactly to R.
    x = x.astype(np.float32).astype(np.float64)
    if not np.isfinite(x).all():
        parser.error("Input conversion to float32 produced non-finite features.")
    rscript = shutil.which("Rscript")
    if rscript is None:
        parser.error("Rscript is missing; install R and RcppCNPy, jsonlite, digest first.")
    bridge = Path(__file__).with_name("initialize_hartigan.R")
    check = subprocess.run([rscript, "--vanilla", str(bridge), "--check"],
                           text=True, capture_output=True, timeout=30)
    if check.returncode:
        raise RuntimeError("Hartigan--Wong dependency check failed:\n" + check.stderr)
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="hw-input-", dir=output) as temporary:
        temporary = Path(temporary)
        converted = temporary / "X_float64.npy"
        np.save(converted, x, allow_pickle=False)
        request = {
            "converted_path": str(converted), "converted_md5": file_hash(converted, "md5"),
            "output_dir": str(output), "n": x.shape[0], "p": x.shape[1],
            "K": args.n_clusters, "seed": args.seed, "nstart": args.n_starts,
            "iter_max": args.max_iterations,
        }
        request_path = temporary / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        fitted = subprocess.run([rscript, "--vanilla", str(bridge), str(request_path)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=1800)
        (output / "r_output.log").write_text(fitted.stdout, encoding="utf-8")
        if fitted.returncode:
            raise RuntimeError(f"R initialization failed; inspect {output / 'r_output.log'}")
    labels = np.load(output / "labels.npy", allow_pickle=False)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if labels.shape != (len(x),) or not np.array_equal(np.unique(labels), np.arange(args.n_clusters)):
        raise RuntimeError("R returned an invalid initialization partition.")
    sse = sum(np.square(x[labels == k] - x[labels == k].mean(0)).sum()
              for k in range(args.n_clusters))
    if not np.isclose(sse, result["total_withinss"], rtol=1e-12, atol=1e-7):
        raise RuntimeError("Independent within-cluster SSE check failed.")
    result.update({
        "x_path": str(source), "x_sha256": file_hash(source),
        "labels_sha256": file_hash(output / "labels.npy"),
        "standardize": args.standardize,
        "fitting_dtype": "float64 exact conversion of trainer float32 input",
        "independently_recomputed_sse": float(sse),
        "reference_labels_used": False,
    })
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {output / 'labels.npy'}; within-cluster SSE={sse:.8g}")
    return result


if __name__ == "__main__":
    main()
