"""Train and evaluate TriFlow-MoE on CAMELS-US."""

import argparse
import gzip
import json
import logging
import pickle
import random
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
from triflow_moe import config
from triflow_moe.camels_us import (
    DYNAMIC_COLUMNS,
    RAW_COLUMNS,
    STATIC_COLUMNS,
    CamelsUS,
    build_datasets,
    read_basin_ids,
)
from triflow_moe.metrics import evaluate
from triflow_moe.model import TriFlowMoE, infer_feature_order
from triflow_moe.training import predict, train

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASIN_FILE = ROOT / "data" / "basins_camels_us_531.txt"
LOGGER = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "CAMELS_US",
        help="CAMELS-US data directory (default: data/CAMELS_US)",
    )
    parser.add_argument(
        "--basin-file",
        type=Path,
        default=DEFAULT_BASIN_FILE,
        help="Gauge list (default: the included 531-basin benchmark)",
    )
    parser.add_argument(
        "--forcing-format",
        choices=("auto", "daymet", "legacy"),
        default="auto",
        help="Daymet text or prepared CSV forcing (auto prefers existing prepared CSVs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Parent directory for run outputs (default: results)",
    )
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:N (default: auto)"
    )
    devices.add_argument(
        "--gpu", type=int, help="CUDA device index, equivalent to --device cuda:N"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="Random seed (default: 325)",
    )
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=config.MAX_EPOCHS,
        help="Maximum number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=config.BATCH_SIZE,
        help="Training batch size (default: 256)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=positive_int,
        default=config.EVAL_BATCH_SIZE,
        help="Validation and test batch size (default: 512)",
    )
    parser.add_argument(
        "--debug-basins",
        type=positive_int,
        help="Run on the first N gauges for a local check",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    try:
        args.device = resolve_device(
            f"cuda:{args.gpu}" if args.gpu is not None else args.device
        )
        args.basins = read_basin_ids(args.basin_file)
        if args.debug_basins is not None:
            args.basins = args.basins[: args.debug_basins]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return torch.device(value)
    if value == "cuda":
        value = "cuda:0"
    kind, separator, index = value.partition(":")
    if kind != "cuda" or not separator or not index.isascii() or not index.isdecimal():
        raise ValueError(
            "Use --device auto, cpu, cuda, or cuda:N with a nonnegative index"
        )
    if not torch.cuda.is_available() or int(index) >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {value} is unavailable. Use --device cpu or a visible CUDA index."
        )
    return torch.device("cuda", int(index))


def write_results(results, run_dir: Path):
    with gzip.open(
        run_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline=""
    ) as stream:
        for index, (basin, frame) in enumerate(results.items()):
            predictions = frame.rename(
                columns={"y_obs": "y_obs_mm_d", "y_sim": "y_sim_mm_d"}
            )
            predictions = predictions.rename_axis("date").reset_index()
            predictions.insert(0, "basin_id", basin)
            predictions.to_csv(
                stream, index=False, header=(index == 0), date_format="%Y-%m-%d"
            )
    with (run_dir / "test_results.pickle").open("wb") as stream:
        pickle.dump(results, stream)
    metrics = evaluate(results)
    metrics.to_csv(run_dir / "metrics.csv")
    summary = (
        metrics.describe()
        .loc[["50%", "mean", "std"]]
        .rename(index={"50%": "median"})
        .T.rename_axis("metric")
    )
    summary.to_csv(run_dir / "metrics_summary.csv")
    return summary


def main(argv=None):
    args = parse_args(argv)
    archive = CamelsUS(args.data_dir, args.basins, args.forcing_format)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        mkdtemp(prefix=f"triflow_moe_seed_{args.seed}_", dir=args.output_dir.resolve())
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "run_progress.txt", encoding="utf-8"),
        ],
    )
    LOGGER.info(
        "TriFlow-MoE | seed %d | %s | %d basins | %s forcing",
        args.seed,
        args.device,
        len(args.basins),
        archive.forcing_format,
    )
    LOGGER.info("Run directory: %s", run_dir)
    splits, scaler = build_datasets(
        archive, config.PERIODS, config.SEQUENCE_LENGTH, config.WARMUP_DAYS
    )
    for name, basins in splits.items():
        LOGGER.info("%s windows: %d", name, sum(map(len, basins.values())))

    p_index = DYNAMIC_COLUMNS.index("prcp_daymet")
    parameters = {
        **config.MODEL_PARAMETERS,
        "input_size_dynamic": len(DYNAMIC_COLUMNS),
        "static_input_size": len(STATIC_COLUMNS),
        "feature_order": infer_feature_order(DYNAMIC_COLUMNS),
        "precip_mean": scaler["x_d_mean"][p_index],
        "precip_std": scaler["x_d_std"][p_index],
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = TriFlowMoE(parameters).to(args.device)
    run_config = {
        "model": "TriFlow-MoE",
        "seed": args.seed,
        "device": str(args.device),
        "pytorch_version": str(torch.__version__),
        "data_dir": str(args.data_dir.resolve()),
        "forcing_format": archive.forcing_format,
        "basin_ids": args.basins,
        "dynamic_columns": DYNAMIC_COLUMNS,
        "static_columns": STATIC_COLUMNS,
        "raw_columns": RAW_COLUMNS,
        "periods": config.PERIODS,
        "sequence_length": config.SEQUENCE_LENGTH,
        "raw_warmup_days": config.WARMUP_DAYS,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_epochs": args.epochs,
        "learning_rate": config.LEARNING_RATE,
        "lr_step_epochs": config.LR_STEP_EPOCHS,
        "lr_gamma": config.LR_GAMMA,
        "patience": config.PATIENCE,
        "gradient_clip_norm": config.GRAD_CLIP_NORM,
        "recharge_usage_floor": config.USAGE_FLOOR,
        "recharge_usage_weight": config.USAGE_WEIGHT,
    }
    for name, value in (
        ("model_config", parameters),
        ("run_config", run_config),
        ("scaler", scaler),
    ):
        (run_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )

    epoch, score = train(
        model,
        splits,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        device=args.device,
        checkpoint=run_dir / "best_model.pt",
    )
    LOGGER.info(
        "Testing checkpoint from epoch %d (validation median NSE %.5f)", epoch, score
    )
    results = predict(model, splits["test"], args.eval_batch_size, args.device)
    summary = write_results(results, run_dir)
    LOGGER.info(
        "Test median NSE %.5f | test median KGE %.5f",
        summary.loc["NSE", "median"],
        summary.loc["KGE", "median"],
    )
    LOGGER.info("Completed: %s", run_dir)


if __name__ == "__main__":
    main()
