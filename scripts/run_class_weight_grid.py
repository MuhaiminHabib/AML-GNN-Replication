import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from datasets.registry import load_dataset
from models.registry import build_model
from training.trainer import TrainConfig, set_seed, train_model


CLASS_WEIGHT_GRID = [0.25, 0.35, 0.50, 0.75, 1.00]


BASE_EXPERIMENTS = [
    {
        "base_name": "gcn_norm_undirected",
        "model": "gcn",
        "dataset": "elliptic",
        "normalize": True,
        "undirected": True,
        "direction_aware": False,
        "epochs": 100,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "hidden_channels": 128,
        "dropout": 0.5,
        "heads": 4,
        "threshold_tuning": False,
    },
    {
        "base_name": "graphsage_raw_undirected",
        "model": "graphsage",
        "dataset": "elliptic",
        "normalize": False,
        "undirected": True,
        "direction_aware": False,
        "epochs": 100,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "hidden_channels": 128,
        "dropout": 0.5,
        "heads": 4,
        "threshold_tuning": False,
    },
    {
        "base_name": "gatv2_dir_heads8_hidden32",
        "model": "gatv2_dir",
        "dataset": "elliptic",
        "normalize": False,
        "undirected": False,
        "direction_aware": True,
        "epochs": 250,
        "lr": 0.0005,
        "weight_decay": 5e-4,
        "hidden_channels": 32,
        "dropout": 0.2,
        "heads": 8,
        "threshold_tuning": False,
    },
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Seeds to run. Start with --seeds 42. Later use 42 7 21 123 2025.",
    )

    return parser.parse_args()


def validate_experiment(exp):
    if exp["undirected"] and exp["direction_aware"]:
        raise ValueError(
            f"{exp['base_name']} is invalid: "
            "undirected and direction_aware cannot both be True."
        )

    if exp["model"] == "gatv2_dir" and not exp["direction_aware"]:
        raise ValueError("gatv2_dir requires direction_aware=True.")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_root = PROJECT_ROOT / "data"

    all_rows = []

    for base_exp in BASE_EXPERIMENTS:
        validate_experiment(base_exp)

        for class_weight_strength in CLASS_WEIGHT_GRID:
            experiment_name = (
                f"{base_exp['base_name']}_cw"
                f"{str(class_weight_strength).replace('.', '')}"
            )

            print("\n" + "=" * 90)
            print(f"Experiment: {experiment_name}")
            print(f"Model: {base_exp['model']}")
            print(f"Class weight strength: {class_weight_strength}")
            print("=" * 90)

            for seed in args.seeds:
                print("\n" + "-" * 90)
                print(f"Running seed: {seed}")
                print("-" * 90)

                set_seed(seed)

                data, _ = load_dataset(
                    base_exp["dataset"],
                    data_root,
                    normalize=base_exp["normalize"],
                    make_undirected=base_exp["undirected"],
                    direction_aware=base_exp["direction_aware"],
                )

                model = build_model(
                    model_name=base_exp["model"],
                    in_channels=data.num_node_features,
                    hidden_channels=base_exp["hidden_channels"],
                    out_channels=2,
                    dropout=base_exp["dropout"],
                    heads=base_exp["heads"],
                )

                config = TrainConfig(
                    seed=seed,
                    epochs=base_exp["epochs"],
                    lr=base_exp["lr"],
                    weight_decay=base_exp["weight_decay"],
                    hidden_channels=base_exp["hidden_channels"],
                    dropout=base_exp["dropout"],
                    threshold_tuning=base_exp["threshold_tuning"],
                    heads=base_exp["heads"],
                    class_weight_strength=class_weight_strength,
                )

                checkpoint_path = (
                    PROJECT_ROOT
                    / "outputs"
                    / "checkpoints"
                    / base_exp["dataset"]
                    / "class_weight_grid"
                    / experiment_name
                    / f"seed_{seed}.pt"
                )

                result = train_model(
                    model=model,
                    data=data,
                    config=config,
                    device=device,
                    checkpoint_path=checkpoint_path,
                )

                metrics = result["test_metrics"]

                row = {
                    "experiment_name": experiment_name,
                    "base_name": base_exp["base_name"],
                    "dataset": base_exp["dataset"],
                    "model": base_exp["model"],
                    "seed": seed,
                    "normalize": base_exp["normalize"],
                    "undirected": base_exp["undirected"],
                    "direction_aware": base_exp["direction_aware"],
                    "class_weight_strength": class_weight_strength,
                    "threshold_tuning": base_exp["threshold_tuning"],
                    "best_epoch": result["best_epoch"],
                    "best_val_f1": result["best_val_f1"],
                    "threshold": metrics["threshold"],
                    "accuracy": metrics["accuracy"],
                    "illicit_precision": metrics["illicit_precision"],
                    "illicit_recall": metrics["illicit_recall"],
                    "illicit_f1": metrics["illicit_f1"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "epochs": base_exp["epochs"],
                    "lr": base_exp["lr"],
                    "weight_decay": base_exp["weight_decay"],
                    "hidden_channels": base_exp["hidden_channels"],
                    "dropout": base_exp["dropout"],
                    "heads": base_exp["heads"],
                }

                all_rows.append(row)

                print("\nSeed result:")
                print(f"experiment_name: {experiment_name}")
                print(f"model: {base_exp['model']}")
                print(f"class_weight_strength: {class_weight_strength}")
                print(f"best_epoch: {row['best_epoch']}")
                print(f"best_val_f1: {row['best_val_f1']:.4f}")
                print(f"test_illicit_f1: {row['illicit_f1']:.4f}")
                print(f"test_pr_auc: {row['pr_auc']:.4f}")

    output_dir = PROJECT_ROOT / "outputs" / "results" / "class_weight_grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "class_weight_grid_results.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nSaved class weight grid results to:")
    print(output_path)


if __name__ == "__main__":
    main()