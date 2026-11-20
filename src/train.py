
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import CLASS_NAMES, build_datasets, build_transforms, compute_class_weights
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())

    average_loss = total_loss / len(loader.dataset)

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )

    recalls = recall_score(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )

    metrics = {
        "loss": average_loss,
        "macro_f1": macro_f1,
    }

    for class_name, recall in zip(CLASS_NAMES, recalls):
        metrics[f"recall_{class_name}"] = float(recall)

    return metrics


def run_sanity_check(
    model: nn.Module,
    train_dataset,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    writer: SummaryWriter,
    metrics_path: Path,
    sanity_epochs: int,
    batch_size: int,
    checkpoint_path: Path,
) -> dict:
    """Overfit one fixed, non-augmented training batch."""

    _, evaluation_transform = build_transforms(image_size=224)
    train_dataset.transform = evaluation_transform

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    first_loss = None
    final_metrics = {}

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, sanity_epochs + 1):
            model.train()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()

            if first_loss is None:
                first_loss = loss.item()

            final_metrics = {
                "epoch": epoch,
                "sanity_loss": loss.item(),
                "sanity_accuracy": accuracy,
            }

            metrics_file.write(json.dumps(final_metrics) + "\n")
            metrics_file.flush()

            writer.add_scalar("sanity/loss", loss.item(), epoch)
            writer.add_scalar("sanity/accuracy", accuracy, epoch)

            print(
                f"Sanity epoch {epoch:03d}/{sanity_epochs} | "
                f"loss: {loss.item():.4f} | accuracy: {accuracy:.4f}"
            )

    final_metrics["first_sanity_loss"] = first_loss

    torch.save(
        {
            "epoch": sanity_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print(f"Sanity checkpoint saved to: {checkpoint_path}")

    return final_metrics


def run_training(
    model: nn.Module,
    loaders: dict,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    writer: SummaryWriter,
    metrics_path: Path,
    epochs: int,
    checkpoint_path: Path,
) -> dict:
    best_validation_f1 = float("-inf")

    final_metrics = {}

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, epochs + 1):
            model.train()
            training_loss = 0.0

            for images, labels in loaders["train"]:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                training_loss += loss.item() * labels.size(0)

            training_loss /= len(loaders["train"].dataset)

            validation_metrics = evaluate(
                model,
                loaders["validation"],
                criterion,
                device,
            )

            current_validation_f1 = validation_metrics["macro_f1"]

            if current_validation_f1 > best_validation_f1:
                best_validation_f1 = current_validation_f1

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_validation_f1": best_validation_f1,
                    },
                    checkpoint_path,
                )

                print(f"Best model checkpoint saved to: {checkpoint_path}")

            final_metrics = {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": validation_metrics["loss"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_recall_benign": validation_metrics["recall_benign"],
                "validation_recall_malignant": validation_metrics["recall_malignant"],
                "validation_recall_normal": validation_metrics["recall_normal"],
            }

            metrics_file.write(json.dumps(final_metrics) + "\n")
            metrics_file.flush()

            writer.add_scalar("loss/train", training_loss, epoch)
            writer.add_scalar("loss/validation", validation_metrics["loss"], epoch)
            writer.add_scalar(
                "metrics/validation_macro_f1",
                validation_metrics["macro_f1"],
                epoch,
            )

            for class_name in CLASS_NAMES:
                writer.add_scalar(
                    f"metrics/recall_{class_name}",
                    validation_metrics[f"recall_{class_name}"],
                    epoch,
                )

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train loss: {training_loss:.4f} | "
                f"validation loss: {validation_metrics['loss']:.4f} | "
                f"validation macro F1: {validation_metrics['macro_f1']:.4f}"
            )

    final_metrics["best_validation_macro_f1"] = best_validation_f1
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["sanity", "train"],
        default="sanity",
    )
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    set_seed(config["project"]["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    run_name = args.run_name or (
        f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    run_directory = Path(config["logging"]["log_dir"]) / run_name
    artifact_directory = Path(config["logging"]["artifact_dir"])

    run_directory.mkdir(parents=True, exist_ok=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)

    ####ADDED - start
    # Define the checkpoint file for the current run.
    checkpoint_path = artifact_directory / f"{run_name}_best_model.pth"
    ####ADDED - finish

    writer = SummaryWriter(log_dir=str(run_directory))

    hparams = {
        "model": config["model"]["name"],
        "image_size": config["data"]["image_size"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "seed": config["project"]["seed"],
    }

    writer.add_text(
        "hyperparameters",
        json.dumps(hparams, indent=2),
    )

    datasets = build_datasets(
        config["data"]["root"],
        config["data"]["image_size"],
    )

    class_weights = compute_class_weights(datasets["train"]).to(device)

    model = build_model(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    print(f"Device: {device}")
    print(f"Class weights: {class_weights.tolist()}")
    print(f"Run name: {run_name}")

    if args.mode == "sanity":
        metrics_path = artifact_directory / f"{run_name}_metrics.jsonl"

        final_metrics = run_sanity_check(
            model=model,
            train_dataset=datasets["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            metrics_path=metrics_path,
            sanity_epochs=config["training"]["sanity_epochs"],
            batch_size=config["training"]["batch_size"],
            checkpoint_path=checkpoint_path,
        )
    else:
        loaders = {
            split_name: DataLoader(
                dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=split_name == "train",
                num_workers=config["data"]["num_workers"],
                pin_memory=device.type == "cuda",
            )
            for split_name, dataset in datasets.items()
        }

        metrics_path = artifact_directory / f"{run_name}_metrics.jsonl"

        final_metrics = run_training(
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            metrics_path=metrics_path,
            epochs=config["training"]["epochs"],
            checkpoint_path=checkpoint_path,
        )

    summary_path = artifact_directory / f"{run_name}_summary.txt"

    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write(f"Mode: {args.mode}\n")
        summary_file.write(f"Device: {device}\n")
        summary_file.write(f"Run name: {run_name}\n")
        summary_file.write(
            "Class weights: "
            f"{class_weights.detach().cpu().tolist()}\n"
        )
        summary_file.write(json.dumps(final_metrics, indent=2) + "\n")

    writer.add_hparams(
        hparams,
        {"final_metric": final_metrics.get("sanity_accuracy", 0.0)},
    )
    writer.close()

    print(f"Metrics saved to: {metrics_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
