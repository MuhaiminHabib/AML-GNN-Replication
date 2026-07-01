import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from datasets.registry import load_dataset
from models.registry import build_model
from training.trainer import TrainConfig, set_seed, train_model


SEEDS = [42, 7, 21, 123, 2025]


EXPERIMENTS = [
    {
        "experiment_name": "gcn_norm_undirected",
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
        "class_weight_strength": 1.0,
        "threshold_tuning": False,
    },
    {
        "experiment_name": "graphsage_raw_undirected_cw05",
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
        "class_weight_strength": 0.50,
        "threshold_tuning": False,
    },
    {
        "experiment_name": "gatv2_dir_heads8_hidden32_cw035",
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
        "class_weight_strength": 0.35,
        "threshold_tuning": False,
    },
]


def validate_experiment(experiment: dict) -> None:
    model_name = experiment["model"].lower()
    undirected = experiment.get("undirected", False)
    direction_aware = experiment.get("direction_aware", False)

    if undirected and direction_aware:
        raise ValueError(
            f"{experiment['experiment_name']} is invalid: "
            "use either undirected=True or direction_aware=True, not both."
        )

    if model_name == "gatv2_dir" and not direction_aware:
        raise ValueError(
            f"{experiment['experiment_name']} is invalid: "
            "gatv2_dir requires direction_aware=True."
        )


def run_one_experiment(experiment: dict, seed: int, device: torch.device) -> dict:
    set_seed(seed)

    dataset_name = experiment["dataset"]
    model_name = experiment["model"]

    data_root = PROJECT_ROOT / "data"

    data, _ = load_dataset(
        dataset_name,
        data_root,
        normalize=experiment["normalize"],
        make_undirected=experiment["undirected"],
        direction_aware=experiment["direction_aware"],
    )

    model = build_model(
        model_name=model_name,
        in_channels=data.num_node_features,
        hidden_channels=experiment["hidden_channels"],
        out_channels=2,
        dropout=experiment["dropout"],
        heads=experiment["heads"],
    )

    config = TrainConfig(
        seed=seed,
        epochs=experiment["epochs"],
        lr=experiment["lr"],
        weight_decay=experiment["weight_decay"],
        hidden_channels=experiment["hidden_channels"],
        dropout=experiment["dropout"],
        threshold_tuning=experiment["threshold_tuning"],
        heads=experiment["heads"],
        class_weight_strength=experiment["class_weight_strength"],
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / dataset_name
        / experiment["experiment_name"]
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

    return {
        "experiment_name": experiment["experiment_name"],
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "normalize": experiment["normalize"],
        "undirected": experiment["undirected"],
        "direction_aware": experiment["direction_aware"],
        "best_epoch": result["best_epoch"],
        "best_val_f1": result["best_val_f1"],
        "threshold": metrics["threshold"],
        "accuracy": metrics["accuracy"],
        "illicit_precision": metrics["illicit_precision"],
        "illicit_recall": metrics["illicit_recall"],
        "illicit_f1": metrics["illicit_f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "epochs": experiment["epochs"],
        "lr": experiment["lr"],
        "weight_decay": experiment["weight_decay"],
        "hidden_channels": experiment["hidden_channels"],
        "dropout": experiment["dropout"],
        "heads": experiment["heads"],
        "class_weight_strength": experiment["class_weight_strength"],
        "threshold_tuning": experiment["threshold_tuning"],
    }


def print_experiment_header(experiment: dict) -> None:
    print("\n" + "=" * 80)
    print(f"Experiment: {experiment['experiment_name']}")
    print(f"Dataset: {experiment['dataset']}")
    print(f"Model: {experiment['model']}")
    print(f"Normalize: {experiment['normalize']}")
    print(f"Undirected: {experiment['undirected']}")
    print(f"Direction-aware: {experiment['direction_aware']}")
    print(f"Class weight strength: {experiment['class_weight_strength']}")
    print(f"Threshold tuning: {experiment['threshold_tuning']}")
    print("=" * 80)


def print_seed_result(row: dict) -> None:
    print("\nSeed result:")
    print(f"experiment_name: {row['experiment_name']}")
    print(f"seed: {row['seed']}")
    print(f"best_epoch: {row['best_epoch']}")
    print(f"best_val_f1: {row['best_val_f1']:.4f}")
    print(f"accuracy: {row['accuracy']:.4f}")
    print(f"illicit_precision: {row['illicit_precision']:.4f}")
    print(f"illicit_recall: {row['illicit_recall']:.4f}")
    print(f"illicit_f1: {row['illicit_f1']:.4f}")
    print(f"roc_auc: {row['roc_auc']:.4f}")
    print(f"pr_auc: {row['pr_auc']:.4f}")
    print(f"threshold: {row['threshold']:.4f}")


def save_results(rows: list[dict]) -> Path:
    output_dir = PROJECT_ROOT / "outputs" / "results" / "multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "elliptic_multiseed_results.csv"

    if not rows:
        raise RuntimeError("No results were generated.")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    all_rows = []

    for experiment in EXPERIMENTS:
        validate_experiment(experiment)
        print_experiment_header(experiment)

        for seed in SEEDS:
            print("\n" + "-" * 80)
            print(f"Running seed: {seed}")
            print("-" * 80)

            row = run_one_experiment(
                experiment=experiment,
                seed=seed,
                device=device,
            )

            all_rows.append(row)
            print_seed_result(row)

    output_path = save_results(all_rows)

    print("\nSaved multi-seed results to:")
    print(output_path)


if __name__ == "__main__":
    main()