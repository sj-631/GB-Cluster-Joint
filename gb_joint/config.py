from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal

from .architectures import SUPPORTED_ARCHITECTURES


Architecture = Literal["linear", "mlp", "mlp_ln_gelu", "dcec_conv", "mnist_conv"]
StandardizeMode = Literal["none", "zscore"]
PosteriorSummary = Literal["salso", "vi_medoid", "dahl"]
PsmMode = Literal["auto", "full", "none"]
Optimizer = Literal["sgd", "adam"]


@dataclass
class TrainConfig:


    x_path: Path
    y_path: Path | None
    output_dir: Path

    n_clusters: int = 4
    z_dim: int = 6
    hidden_dim: int = 128
    dropout: float = 0.0
    architecture: Architecture = "linear"

    batch_size: int = 64
    pretrain_epochs: int = 100
    outer_iterations: int = 100
    joint_epochs_per_outer: int = 1
    gibbs_sweeps_per_outer: int = 10
    mc_partitions: int = 5
    # A sequential chain retains the last M of B sweeps, not spaced draws.
    # MC sample count is separate from the number of persistent chains.
    partition_sampler: str = "single_chain_sequential"
    initial_chain_burnin_sweeps: int = 0
    partition_initialization: str = "latent_kmeans"
    initial_partition_path: Path | None = None
    # Optional final-warmup override for the actual Gibbs start. A supplied
    # partition still serves before/intermediate-warmup calibration if selected.
    postwarm_partition_initialization: str = "latent_hartigan"
    hartigan_n_starts: int = 100
    # Optional linear-AE scale constraint; changes the clustering loss and prior.
    # Optimizer states are unchanged by the projection.
    latent_scale_projection: bool = True

    posterior_burnin_sweeps: int = 0
    posterior_samples: int = 500
    posterior_thinning: int = 1
    posterior_summary: PosteriorSummary = "salso"
    salso_n_runs: int = 64
    salso_n_cores: int = 1
    psm_mode: PsmMode = "auto"
    psm_max_n: int = 2500

    # Calibrate lambda_R at the selected fixed network checkpoint so that
    # median(lambda_R * ||grad R|| / ||grad C||) = r0.
    target_gradient_ratio: float = 1.0
    gradient_epsilon: float = 1e-12
    calibration_batches: int | None = None
    lambda_calibration_stage: str = "after_warmup"
    # Calibrate once after t=10 completed warmup epochs, then keep lambda fixed.
    # None follows the stage setting; an explicit epoch takes precedence.
    lambda_calibration_epoch: int | None = 10
    eta_initialization: str = "pretrained_loss"

    eta_a: float = 1.0
    eta_b: float = 1.0
    eta_shape_scale: float = 1.0
    eta_multiplier: float = 1.0
    optimizer: Optimizer = "adam"
    # Adam defaults. SGD requires separately chosen learning rates.
    lr_pretrain: float = 0.005
    lr_joint: float = 0.001
    encoder_prior_precision: float = 1e-4
    decoder_prior_precision: float = 1e-4

    standardize: StandardizeMode = "none"
    seed: int = 125
    device: str = "cpu"
    num_workers: int = 0
    verbose: bool = True
    save_diagnostics: bool = False
    hidden_dims: list[int] | None = None  # MLP widths; otherwise use hidden_dim.

    @property
    def n_chains(self) -> int:
        return 1 if self.partition_sampler == "single_chain_sequential" else self.mc_partitions

    @property
    def resolved_lambda_calibration_epoch(self) -> int:
        if self.lambda_calibration_epoch is not None:
            return self.lambda_calibration_epoch
        return 0 if self.lambda_calibration_stage == "before_warmup" else self.pretrain_epochs

    @property
    def resolved_lambda_calibration_stage(self) -> str:
        if self.lambda_calibration_epoch is None:
            return self.lambda_calibration_stage
        if self.lambda_calibration_epoch == 0:
            return "before_warmup"
        if self.lambda_calibration_epoch == self.pretrain_epochs:
            return "after_warmup"
        return "during_warmup"

    def validate(self) -> None:
        integer_positive = {
            "n_clusters": self.n_clusters,
            "z_dim": self.z_dim,
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
            "outer_iterations": self.outer_iterations,
            "joint_epochs_per_outer": self.joint_epochs_per_outer,
            "gibbs_sweeps_per_outer": self.gibbs_sweeps_per_outer,
            "mc_partitions": self.mc_partitions,
            "posterior_samples": self.posterior_samples,
            "posterior_thinning": self.posterior_thinning,
            "salso_n_runs": self.salso_n_runs,
            "salso_n_cores": self.salso_n_cores,
        }
        for name, value in integer_positive.items():
            if name in {"gibbs_sweeps_per_outer", "mc_partitions"} and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"{name} must be a positive integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if self.partition_sampler not in {"independent_chains", "single_chain_sequential"}:
            raise ValueError("Unknown partition_sampler.")
        if self.partition_initialization not in {"latent_kmeans", "provided"}:
            raise ValueError("partition_initialization must be 'latent_kmeans' or 'provided'.")
        if self.partition_initialization == "provided":
            supplied_partition_unused = (
                self.postwarm_partition_initialization == "latent_hartigan"
                and self.resolved_lambda_calibration_stage == "after_warmup"
            )
            if self.initial_partition_path is None and not supplied_partition_unused:
                raise ValueError(
                    "initial_partition_path is required for partition_initialization='provided', "
                    "unless final latent Hartigan--Wong supplies both calibration and Gibbs initialization."
                )
        elif self.initial_partition_path is not None:
            raise ValueError("initial_partition_path requires partition_initialization='provided'.")
        if self.postwarm_partition_initialization not in {"unchanged", "latent_hartigan"}:
            raise ValueError("postwarm_partition_initialization must be 'unchanged' or 'latent_hartigan'.")
        if (
            not isinstance(self.hartigan_n_starts, int)
            or isinstance(self.hartigan_n_starts, bool)
            or self.hartigan_n_starts < 1
        ):
            raise ValueError("hartigan_n_starts must be a positive integer.")
        if self.postwarm_partition_initialization == "latent_hartigan" and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
            or not 0 <= self.seed <= 2**31 - 1
        ):
            raise ValueError("Hartigan--Wong requires a non-negative signed 32-bit seed.")
        if not isinstance(self.latent_scale_projection, bool):
            raise ValueError("latent_scale_projection must be a boolean.")
        if self.latent_scale_projection and self.architecture != "linear":
            raise ValueError("latent_scale_projection currently supports only architecture='linear'.")
        if (
            not isinstance(self.pretrain_epochs, int)
            or isinstance(self.pretrain_epochs, bool)
            or self.pretrain_epochs < 0
        ):
            raise ValueError("pretrain_epochs must be a non-negative integer.")
        if self.lambda_calibration_stage not in {"before_warmup", "after_warmup"}:
            raise ValueError("Unknown lambda_calibration_stage.")
        if self.lambda_calibration_epoch is not None and (
            not isinstance(self.lambda_calibration_epoch, int)
            or isinstance(self.lambda_calibration_epoch, bool)
            or not 0 <= self.lambda_calibration_epoch <= self.pretrain_epochs
        ):
            raise ValueError(
                "lambda_calibration_epoch must be an integer between 0 and pretrain_epochs, or None."
            )
        if self.eta_initialization == "prior_mean":
            raise ValueError("Use eta_initialization='zero_loss': this option is not the prior mean.")
        if self.eta_initialization not in {"zero_loss", "pretrained_loss"}:
            raise ValueError("Unknown eta_initialization.")
        if self.postwarm_partition_initialization == "latent_hartigan" and self.eta_initialization != "pretrained_loss":
            raise ValueError("latent_hartigan requires eta_initialization='pretrained_loss'.")
        if self.partition_sampler == "single_chain_sequential":
            if self.gibbs_sweeps_per_outer < self.mc_partitions:
                raise ValueError("Sequential sweeps B must be at least mc_partitions M.")
        for name, value in {
            "initial_chain_burnin_sweeps": self.initial_chain_burnin_sweeps,
            "posterior_burnin_sweeps": self.posterior_burnin_sweeps,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.posterior_samples < self.mc_partitions:
            raise ValueError("posterior_samples must be at least mc_partitions.")
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                "architecture must be 'linear', 'mlp', 'mlp_ln_gelu', "
                "'dcec_conv', or 'mnist_conv'."
            )
        if self.hidden_dims is not None:
            if self.architecture != "mlp":
                raise ValueError("hidden_dims is supported only for architecture='mlp'.")
            if not isinstance(self.hidden_dims, (list, tuple)) or not self.hidden_dims or any(
                not isinstance(width, int) or isinstance(width, bool) or width < 1
                for width in self.hidden_dims
            ):
                raise ValueError("hidden_dims must be a nonempty sequence of positive integers.")
        if self.standardize not in {"none", "zscore"}:
            raise ValueError("standardize must be 'none' or 'zscore'.")
        if self.posterior_summary not in {"salso", "vi_medoid", "dahl"}:
            raise ValueError("posterior_summary must be 'salso', 'vi_medoid' or 'dahl'.")
        if self.optimizer not in {"sgd", "adam"}:
            raise ValueError("optimizer must be 'sgd' or 'adam'.")
        if self.psm_mode not in {"auto", "full", "none"}:
            raise ValueError("psm_mode must be 'auto', 'full', or 'none'.")
        if self.psm_max_n < 1:
            raise ValueError("psm_max_n must be positive.")
        if self.posterior_summary == "dahl" and self.psm_mode == "none":
            raise ValueError("Dahl's summary requires a posterior similarity matrix.")

        finite_positive = {
            "target_gradient_ratio": self.target_gradient_ratio,
            "gradient_epsilon": self.gradient_epsilon,
            "eta_a": self.eta_a,
            "eta_b": self.eta_b,
            "eta_shape_scale": self.eta_shape_scale,
            "eta_multiplier": self.eta_multiplier,
            "lr_pretrain": self.lr_pretrain,
            "lr_joint": self.lr_joint,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.calibration_batches is not None and self.calibration_batches < 1:
            raise ValueError("calibration_batches must be positive or None.")
        if self.dropout != 0:
            raise ValueError(
                "dropout must be 0: this implementation uses a deterministic "
                "encoder/decoder map for centroid and gradient calculations."
            )
        if self.architecture in {"dcec_conv", "mnist_conv"}:
            if self.standardize != "none":
                raise ValueError(f"{self.architecture} requires unstandardized [0, 1] pixels.")
        for name, value in {
            "encoder_prior_precision": self.encoder_prior_precision,
            "decoder_prior_precision": self.decoder_prior_precision,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["hidden_dims"] = list(self.hidden_dims) if self.hidden_dims is not None else None
        values["x_path"] = str(Path(self.x_path).expanduser().resolve())
        values["y_path"] = (
            str(Path(self.y_path).expanduser().resolve())
            if self.y_path is not None
            else None
        )
        values["output_dir"] = str(Path(self.output_dir).expanduser().resolve())
        values["initial_partition_path"] = (
            str(Path(self.initial_partition_path).expanduser().resolve())
            if self.initial_partition_path is not None else None
        )
        return values
