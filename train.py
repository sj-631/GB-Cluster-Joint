from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from gb_joint.architectures import SUPPORTED_ARCHITECTURES
from gb_joint.config import TrainConfig
from gb_joint.data import load_npy_dataset
from gb_joint.trainer import TotalLossTrainer
from gb_joint.utils import save_json, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="GB Cluster--Joint: lambda_R * reconstruction SSE + clustering SSE.",
    )
    defaults = {field.name: field.default for field in fields(TrainConfig)}

    def option(name: str, **kwargs) -> None:
        parser.add_argument("--" + name.replace("_", "-"), default=defaults[name], **kwargs)

    parser.add_argument("--preset", type=Path, help="JSON settings; explicit CLI options take precedence.")
    parser.add_argument("--x-path", type=Path, required=True)
    parser.add_argument("--y-path", type=Path, help="Optional reference labels, for evaluation only.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    option("architecture", choices=SUPPORTED_ARCHITECTURES)
    option("hidden_dims", type=int, nargs="+", help="MLP hidden widths, e.g. 256 64; overrides hidden-dim.")
    option("dropout", type=float, help="Must be zero for deterministic network maps.")
    for name in (
        "z_dim", "hidden_dim", "batch_size", "pretrain_epochs", "outer_iterations",
        "joint_epochs_per_outer", "hartigan_n_starts", "initial_chain_burnin_sweeps",
        "posterior_samples", "posterior_thinning", "salso_n_runs", "salso_n_cores",
        "psm_max_n", "calibration_batches", "seed", "num_workers",
    ):
        option(name, type=int)
    for name in (
        "gradient_epsilon", "eta_a", "eta_b", "lr_pretrain", "lr_joint",
        "encoder_prior_precision", "decoder_prior_precision",
    ):
        option(name, type=float)

    option("gibbs_sweeps_per_outer", type=int, help="B sweeps per joint iteration.")
    option("mc_partitions", type=int, help="M retained states; the sequential chain retains sweeps B-M+1 through B.")
    option("partition_initialization", choices=["latent_kmeans", "provided"],
           help="Calibration/Gibbs partition source, unless overridden after warmup.")
    option("initial_partition_path", type=Path,
           help="Provided NPY partition, not reference labels. Required for early calibration under the provided policy.")
    option("postwarm_partition_initialization", choices=["unchanged", "latent_hartigan"],
           help="Override the Gibbs start with final latent Hartigan--Wong; also used for after-warmup calibration.")
    option("latent_scale_projection", action=argparse.BooleanOptionalAction,
           help="Project pooled latent variance to one (linear architecture, CPU only).")
    option("partition_sampler", choices=["independent_chains", "single_chain_sequential"])
    option("posterior_burnin_sweeps", type=int,
           help="Extra final-stage burn-in; continue the last training-chain state.")
    option("posterior_summary", choices=["salso", "vi_medoid", "dahl"])
    option("psm_mode", choices=["auto", "full", "none"],
           help="Save the co-clustering matrix; auto skips n above psm-max-n.")

    option("target_gradient_ratio", type=float, help="r0: target weighted reconstruction/clustering gradient ratio.")
    option("lambda_calibration_stage", choices=["before_warmup", "after_warmup"],
           help="Used only when lambda-calibration-epoch is unset.")
    option("lambda_calibration_epoch", type=int,
           help="Calibrate once after t warmup epochs (default: 10), 0<=t<=T. This is not the total warmup length; t=0 precedes warmup. A null preset value follows the stage.")
    option("eta_initialization", choices=["pretrained_loss", "zero_loss"],
           help="Use actual pretrained loss, or the experimental zero-loss Gamma initialization.")
    option("eta_shape_scale", type=float, help="C in a_eta + C*n*(p+d)/2; non-unit values are sensitivity variants.")
    option("eta_multiplier", type=float, help="Sensitivity multiplier applied to eta for Gibbs and neural updates.")
    option("optimizer", choices=["sgd", "adam"], help="SGD requires both learning rates to be explicitly supplied.")
    option("standardize", choices=["none", "zscore"])
    option("device", type=str)
    option("save_diagnostics", action="store_true", help="Save intermediate checkpoints and detailed diagnostics.")
    parser.add_argument(
        "--num-threads", type=int, default=1,
        help="CPU numerical-library threads (default: one for reproducible runs).",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, object]]:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--preset", type=Path, default=None)
    preliminary_args, _ = preliminary.parse_known_args(argv)

    parser = build_parser()
    preset_values: dict[str, object] = {}
    if preliminary_args.preset is not None:
        preset_path = preliminary_args.preset.expanduser().resolve()
        if not preset_path.is_file():
            parser.error(f"Preset file does not exist: {preset_path}")
        with preset_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            parser.error("A preset must be a JSON object.")
        allowed = {
            action.dest
            for action in parser._actions
            if action.dest not in {"help", "preset"}
        }
        forbidden = {"x_path", "y_path", "output_dir", "initial_partition_path"}
        unknown = set(loaded) - allowed
        path_fields = set(loaded) & forbidden
        if unknown:
            parser.error(
                "Unknown preset field(s): " + ", ".join(sorted(unknown))
            )
        if path_fields:
            parser.error(
                "Data and output paths must be supplied explicitly, not in a "
                "preset: " + ", ".join(sorted(path_fields))
            )
        preset_values = loaded
        parser.set_defaults(**preset_values)
    args = parser.parse_args(argv)
    if args.optimizer == "sgd":
        explicit_options = {
            arg.split("=", 1)[0] for arg in (sys.argv[1:] if argv is None else argv)
        }
        missing = [
            field for field in ("lr_pretrain", "lr_joint")
            if field not in preset_values
            and "--" + field.replace("_", "-") not in explicit_options
        ]
        if missing:
            parser.error(
                "SGD requires explicitly supplied learning rates: "
                + ", ".join("--" + field.replace("_", "-") for field in missing)
                + " (or the corresponding preset fields)."
            )
    return args, preset_values


def main(argv: list[str] | None = None) -> dict[str, object]:
    args, preset_values = parse_args(argv)
    if args.num_threads < 1:
        raise SystemExit("--num-threads must be positive.")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(
            f"Refusing to write into an existing output directory: {output_dir}"
        )
    values = {field.name: getattr(args, field.name) for field in fields(TrainConfig)
              if field.name not in {"output_dir", "verbose"}}
    config = TrainConfig(**values, output_dir=output_dir, verbose=not args.quiet)
    config.validate()
    dataset, metadata = load_npy_dataset(
        config.x_path, config.y_path, standardize=config.standardize
    )
    if config.n_clusters > len(dataset):
        raise SystemExit("n_clusters cannot exceed the number of observations.")
    TotalLossTrainer.validate_dataset_contract(config, dataset)
    TotalLossTrainer.validate_runtime_dependencies(config)

    output_dir.mkdir(parents=True, exist_ok=False)
    if config.save_diagnostics:
        save_json(
            output_dir / "command.json",
            {
                "argv": [sys.executable, *(
                    sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]
                )],
                "cwd": os.getcwd(),
                "invoked_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    if config.save_diagnostics and args.preset is not None:
        preset_path = args.preset.expanduser().resolve()
        save_json(
            output_dir / "preset_manifest.json",
            {
                "path": str(preset_path),
                "sha256": sha256_file(preset_path),
                "values": preset_values,
                "command_line_overrides_take_precedence": True,
            },
        )
    if config.save_diagnostics:
        for name, array in metadata.items():
            np.save(output_dir / f"{name}.npy", array)
    elif metadata:
        np.savez(output_dir / "preprocessing.npz", **metadata)

    torch.set_num_threads(args.num_threads)
    save_json(output_dir / "execution.json", {
        "cpu_threads": args.num_threads,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "numerical_libraries_limited_by_threadpoolctl": True,
    })
    with threadpool_limits(limits=args.num_threads):
        result = TotalLossTrainer(config, dataset).fit()
    return result


if __name__ == "__main__":
    torch.set_num_interop_threads(1)
    main()
