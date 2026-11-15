from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


CLASS_NAMES = ("benign", "malignant", "normal")


def _split_path(data_root: str | Path, split: str) -> Path:
    path = Path(data_root) / split

    if not path.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    missing_classes = [
        class_name for class_name in CLASS_NAMES
        if not (path / class_name).is_dir()
    ]

    if missing_classes:
        raise FileNotFoundError(
            f"Missing class folders in {path}: {missing_classes}"
        )

    return path


def build_transforms(image_size: int):
    """Return transforms for training and evaluation images."""

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    return train_transform, evaluation_transform


def build_datasets(data_root: str | Path, image_size: int) -> dict[str, Dataset]:
    """Load the provided train, validation, and test image folders."""

    train_transform, evaluation_transform = build_transforms(image_size)

    datasets = {
        "train": ImageFolder(
            _split_path(data_root, "train"),
            transform=train_transform,
        ),
        "validation": ImageFolder(
            _split_path(data_root, "validation"),
            transform=evaluation_transform,
        ),
        "test": ImageFolder(
            _split_path(data_root, "test"),
            transform=evaluation_transform,
        ),
    }

    expected_mapping = {
        class_name: index
        for index, class_name in enumerate(CLASS_NAMES)
    }

    if datasets["train"].class_to_idx != expected_mapping:
        raise ValueError(
            "Unexpected class mapping: "
            f"{datasets['train'].class_to_idx}"
        )

    return datasets


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """Compute inverse-frequency weights for weighted cross-entropy loss."""

    targets = torch.tensor(dataset.targets)
    counts = torch.bincount(targets, minlength=len(CLASS_NAMES)).float()

    if torch.any(counts == 0):
        raise ValueError(f"At least one class has no images: {counts.tolist()}")

    return counts.sum() / (len(CLASS_NAMES) * counts)
