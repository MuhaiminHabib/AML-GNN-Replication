import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch

from datasets.registry import load_dataset
from models.registry import build_model
from training.trainer import TrainConfig, set_seed, train_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="elliptic")
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no_threshold_tuning",action="store_true",help="Disable validation-based threshold tuning and use threshold=0.5.")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--class_weight_strength", type=float, default=1.0)
    parser.add_argument("--no_normalize", action="store_true", help="Disable train-time feature normalization.")
    parser.add_argument("--undirected", action="store_true", help="Convert the graph to bidirectional/undirected message passing.")
    parser.add_argument("--direction_aware", action="store_true", help="Use bidirectional edges with edge direction attributes.",)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.direction_aware and args.undirected:
        raise ValueError("Use --direction_aware or --undirected, not both.")

    if args.model.lower() == "gatv2_dir" and not args.direction_aware:
        raise ValueError("gatv2_dir requires --direction_aware.")

    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        threshold_tuning=not args.no_threshold_tuning,
        heads=args.heads,
        class_weight_strength=args.class_weight_strength,
    )

    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_root = PROJECT_ROOT / "data"
    data, summary_fn = load_dataset(
        args.dataset,
        data_root,
        normalize=not args.no_normalize,
        make_undirected=args.undirected,
        direction_aware=args.direction_aware,
    )
    summary_fn(data)

    model = build_model(
        model_name=args.model,
        in_channels=data.num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        dropout=args.dropout,
        heads=args.heads,
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / args.dataset
        / args.model
        / f"seed_{args.seed}.pt"
    )

    result = train_model(
        model=model,
        data=data,
        config=config,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    print("\nFinal Test Metrics")
    print(f"dataset: {args.dataset}")
    print(f"model: {args.model}")
    print(f"seed: {args.seed}")
    print(f"normalize: {not args.no_normalize}")
    print(f"threshold_tuning: {not args.no_threshold_tuning}")
    print(f"undirected: {args.undirected}")
    print(f"direction_aware: {args.direction_aware}")

    for key, value in result["test_metrics"].items():
        print(f"{key}: {value:.4f}")

    result_path = (
        PROJECT_ROOT
        / "outputs"
        / "results"
        / args.dataset
        / args.model
        / f"seed_{args.seed}.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)

    result["dataset"] = args.dataset
    result["model"] = args.model
    result["seed"] = args.seed
    result["normalize"] = not args.no_normalize
    result["threshold_tuning"] = not args.no_threshold_tuning
    result["undirected"] = args.undirected
    result["direction_aware"] = args.direction_aware
    result["hyperparameters"] = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_channels": args.hidden_channels,
        "dropout": args.dropout,
        "heads": args.heads,
        "class_weight_strength": args.class_weight_strength,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"\nSaved result to: {result_path}")


if __name__ == "__main__":
    main()