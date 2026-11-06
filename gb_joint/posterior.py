from __future__ import annotations

import hashlib
import subprocess
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def variation_of_information(first: np.ndarray, second: np.ndarray) -> float:
    """Variation of information using natural logarithms."""
    first = np.asarray(first)
    second = np.asarray(second)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Partitions must be aligned one-dimensional arrays.")
    first = relabel_consecutive(first)
    second = relabel_consecutive(second)
    if first.shape != second.shape:
        raise ValueError("Partitions must be aligned one-dimensional arrays.")
    n = first.size
    if n == 0:
        raise ValueError("Partitions must be nonempty.")

    k_first = int(first.max()) + 1
    k_second = int(second.max()) + 1
    contingency = np.zeros((k_first, k_second), dtype=np.int64)
    np.add.at(contingency, (first, second), 1)
    row_counts = contingency.sum(axis=1)
    column_counts = contingency.sum(axis=0)

    row_probabilities = row_counts / n
    column_probabilities = column_counts / n
    entropy_first = -float(np.sum(row_probabilities * np.log(row_probabilities)))
    entropy_second = -float(
        np.sum(column_probabilities * np.log(column_probabilities))
    )

    nonzero_rows, nonzero_columns = np.nonzero(contingency)
    joint_counts = contingency[nonzero_rows, nonzero_columns].astype(np.float64)
    mutual_information = float(
        np.sum(
            (joint_counts / n)
            * np.log(
                n
                * joint_counts
                / (
                    row_counts[nonzero_rows]
                    * column_counts[nonzero_columns]
                )
            )
        )
    )
    value = entropy_first + entropy_second - 2.0 * mutual_information
    return float(max(value, 0.0))


def empirical_expected_vi(candidate: np.ndarray, samples: np.ndarray) -> float:
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (S, n).")
    return float(
        np.mean([variation_of_information(candidate, sample) for sample in samples])
    )


def vi_medoid_representative_partition(
    samples: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("samples must have shape (S, n) with S >= 1.")
    if samples.shape[1] < 1:
        raise ValueError("The retained partitions must be nonempty.")

    relabeled = np.stack(
        [relabel_consecutive(sample) for sample in samples], axis=0
    )
    cluster_counts: list[np.ndarray] = []
    entropies = np.empty(relabeled.shape[0], dtype=np.float64)
    for index, sample in enumerate(relabeled):
        counts = np.bincount(sample)
        cluster_counts.append(counts)
        probabilities = counts.astype(np.float64) / sample.size
        entropies[index] = -float(
            np.sum(probabilities * np.log(probabilities))
        )

    losses = np.zeros(relabeled.shape[0], dtype=np.float64)
    for first_index in range(relabeled.shape[0]):
        for second_index in range(first_index):
            first_counts = cluster_counts[first_index]
            second_counts = cluster_counts[second_index]
            second_k = second_counts.size
            joint_counts = np.bincount(
                relabeled[first_index] * second_k + relabeled[second_index],
                minlength=first_counts.size * second_k,
            ).reshape(first_counts.size, second_k)
            nonzero_first, nonzero_second = np.nonzero(joint_counts)
            positive_joint = joint_counts[
                nonzero_first, nonzero_second
            ].astype(np.float64)
            mutual_information = float(
                np.sum(
                    (positive_joint / samples.shape[1])
                    * np.log(
                        samples.shape[1]
                        * positive_joint
                        / (
                            first_counts[nonzero_first]
                            * second_counts[nonzero_second]
                        )
                    )
                )
            )
            distance = max(
                entropies[first_index]
                + entropies[second_index]
                - 2.0 * mutual_information,
                0.0,
            )
            losses[first_index] += distance
            losses[second_index] += distance
    losses /= relabeled.shape[0]
    best_index = int(np.argmin(losses))
    return (
        relabeled[best_index],
        best_index,
        float(losses[best_index]),
    )


@dataclass(frozen=True)
class PosteriorSummaryResult:
    method: str
    realized_clusters: int
    empirical_expected_vi: float
    candidate_index: int | None
    candidate_space: str
    salso_n_runs: int | None = None
    salso_stdout: str | None = None
    salso_metadata: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_salso_metadata(stdout: str) -> dict[str, object]:
    """Read SALSO_META records from the R bridge output."""
    integer_fields = {
        "draws", "observations", "draw_clusters", "requested_max_clusters",
        "effective_max_clusters", "realized_clusters", "requested_n_runs",
        "actual_n_runs", "n_cores", "seed",
    }
    result: dict[str, object] = {}
    for line in stdout.splitlines():
        if not line.startswith("SALSO_META\t"):
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3 or parts[1] in result:
            raise RuntimeError("Malformed or duplicate SALSO metadata.")
        _, key, value = parts
        try:
            if key in integer_fields:
                result[key] = int(value)
            elif key in {"expected_vi_bits", "expected_vi_nats"}:
                result[key] = float(value)
            else:
                result[key] = value
        except ValueError as error:
            raise RuntimeError(f"Invalid SALSO metadata field {key!r}.") from error
    return result


def ensure_salso_available(script_path: Path) -> dict[str, object]:
    """Check R, the bridge, and required R packages before training.

    Missing SALSO dependencies raise an error; the summary method is not changed.
    """
    script_path = Path(script_path).expanduser().resolve()
    if not script_path.is_file():
        raise RuntimeError(f"SALSO bridge script does not exist: {script_path}")
    executable = shutil.which("Rscript")
    if executable is None:
        raise RuntimeError(
            "SALSO requires Rscript on PATH and the R packages 'salso' and "
            "'RcppCNPy'. Install these dependencies before training; "
            "no VI-medoid fallback is used."
        )
    command = [executable, "--vanilla", str(script_path), "--check"]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        diagnostic = getattr(error, "stderr", None) or getattr(error, "stdout", None)
        raise RuntimeError(
            "SALSO dependency check failed. Rscript and the R packages 'salso' "
            "and 'RcppCNPy' must be available; no VI-medoid fallback is used. "
            f"Diagnostic: {diagnostic or error}"
        ) from error
    metadata = parse_salso_metadata(completed.stdout)
    if metadata.get("status") != "available" or not all(
        metadata.get(key) for key in ("r_version", "salso_version", "rcppcnpy_version")
    ):
        raise RuntimeError("SALSO dependency check returned incomplete metadata.")
    metadata.update(rscript=str(Path(executable).resolve()), script_path=str(script_path))
    return metadata


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if value < (0 if allow_zero else 1) or value > np.iinfo(np.int32).max:
        raise ValueError(f"{name} is outside the supported R integer range.")
    return value


def _canonical_integer_draws(samples: np.ndarray, n_clusters: int) -> np.ndarray:
    """Canonicalize by first occurrence, invariant to arbitrary label names."""
    if samples.ndim != 2 or min(samples.shape) < 1:
        raise ValueError("Partition samples must be a nonempty (S, n) matrix.")
    if not np.issubdtype(samples.dtype, np.integer):
        raise ValueError("Partition samples must have integer-valued integer dtype.")
    if samples.shape[1] > np.iinfo(np.int32).max:
        raise ValueError("Too many observations for SALSO's R integer labels.")
    canonical = np.empty(samples.shape, dtype=np.int32)
    for i, sample in enumerate(samples):
        _, first_indices, inverse = np.unique(
            sample, return_index=True, return_inverse=True,
        )
        if first_indices.size != n_clusters:
            raise ValueError(
                f"Draw {i} has {first_indices.size} clusters; expected {n_clusters}."
            )
        order = np.argsort(first_indices)
        first_occurrence_labels = np.empty(order.size, dtype=np.int32)
        first_occurrence_labels[order] = np.arange(order.size, dtype=np.int32)
        canonical[i] = first_occurrence_labels[inverse]
    return canonical


def run_salso_vi(
    *,
    samples_path: Path,
    output_path: Path,
    n_clusters: int,
    n_runs: int = 64,
    n_cores: int = 1,
    seed: int,
    script_path: Path,
    max_clusters: int | None = None,
) -> tuple[np.ndarray, str]:
    """Run randomized SALSO search under variation-of-information loss.
    """
    n_clusters = _positive_integer(n_clusters, "n_clusters")
    n_runs = _positive_integer(n_runs, "n_runs")
    n_cores = _positive_integer(n_cores, "n_cores")
    seed = _positive_integer(seed, "seed", allow_zero=True)
    samples_path = Path(samples_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite a SALSO output: {output_path}")
    samples = _canonical_integer_draws(
        np.load(samples_path, allow_pickle=False), n_clusters,
    )
    requested_max = samples.shape[1] if max_clusters is None else _positive_integer(
        max_clusters, "max_clusters", allow_zero=True,
    )
    if requested_max > samples.shape[1]:
        raise ValueError("max_clusters cannot exceed the number of observations.")
    effective_max = n_clusters if requested_max == 0 else requested_max
    dependencies = ensure_salso_available(script_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # RcppCNPy's integer reader can depend on native C long width. Transfer
    # exact integer labels as float64, verify the entire payload in R, and
    # convert to R integers only after validation. Keep the input file unchanged.
    with tempfile.TemporaryDirectory(prefix="salso-", dir=output_path.parent) as temporary:
        transfer_input = Path(temporary) / "draws_float64.npy"
        transfer_output = Path(temporary) / "representative.npy"
        transfer_samples = np.ascontiguousarray(samples, dtype="<f8")
        transfer_md5 = hashlib.md5(transfer_samples.tobytes(order="C")).hexdigest()
        np.save(transfer_input, transfer_samples, allow_pickle=False)
        command = [
            str(dependencies["rscript"]), "--vanilla", str(dependencies["script_path"]),
            str(transfer_input), str(transfer_output), str(n_clusters),
            str(n_runs), str(n_cores), str(seed), str(requested_max), transfer_md5,
        ]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            diagnostic = getattr(error, "stderr", None) or getattr(error, "stdout", None)
            raise RuntimeError(
                f"SALSO-VI search failed; no VI-medoid fallback is used. "
                f"Diagnostic: {diagnostic or error}"
            ) from error
        if not transfer_output.is_file():
            raise RuntimeError("SALSO completed without creating its partition output.")
        labels = np.load(transfer_output, allow_pickle=False)
        if labels.ndim != 1 or labels.shape[0] != samples.shape[1]:
            raise RuntimeError("SALSO returned a partition with the wrong shape.")
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise RuntimeError("SALSO returned invalid non-integer or negative labels.")
        unique = np.unique(labels)
        if not np.array_equal(unique, np.arange(unique.size)) or unique.size > effective_max:
            raise RuntimeError("SALSO returned nonconsecutive or out-of-range labels.")
        labels = labels.astype(np.int64, copy=False)
        metadata = parse_salso_metadata(completed.stdout)
        expected_metadata = {
            "status": "completed", "draws": samples.shape[0],
            "observations": samples.shape[1], "draw_clusters": n_clusters,
            "requested_max_clusters": requested_max, "effective_max_clusters": effective_max,
            "realized_clusters": unique.size, "requested_n_runs": n_runs,
            "n_cores": n_cores, "seed": seed,
            "input_row_major_payload_md5": transfer_md5,
        }
        if any(metadata.get(k) != v for k, v in expected_metadata.items()):
            raise RuntimeError("SALSO metadata does not match the requested search.")
        if not isinstance(metadata.get("actual_n_runs"), int) or metadata["actual_n_runs"] < 1:
            raise RuntimeError("SALSO did not report any completed search runs.")
        reported_bits = metadata.get("expected_vi_bits", float("nan"))
        expected_nats = empirical_expected_vi(labels, samples)
        if not np.isfinite(reported_bits) or not np.isclose(
            reported_bits * np.log(2.0), expected_nats, rtol=1e-8, atol=1e-8,
        ):
            raise RuntimeError("SALSO's reported VI disagrees with the saved partition.")
        # Exclusive creation protects earlier results even if another writer
        # appears between the initial existence check and the completed search.
        with output_path.open("xb") as handle:
            np.save(handle, labels, allow_pickle=False)
    stdout = completed.stdout + completed.stderr
    stdout += f"\nSALSO_META\texpected_vi_nats\t{expected_nats:.17g}\n"
    return labels, stdout
