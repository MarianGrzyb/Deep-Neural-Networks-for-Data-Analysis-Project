
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.dataset import CLASS_NAMES, build_transforms
from src.model import build_model


def load_inference_model(
    checkpoint_path: str | Path,
    config_path: str | Path = "configs/base.yaml",
):
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_config = config["model"]
    image_size = config["data"]["image_size"]

    model = build_model(
        num_classes=model_config["num_classes"],
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, evaluation_transform = build_transforms(
        image_size=image_size
    )

    return model, evaluation_transform, device


def predict_image(image: Image.Image, model, transform, device):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    results = {
        class_name: float(probabilities[index].item())
        for index, class_name in enumerate(CLASS_NAMES)
    }

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, results
