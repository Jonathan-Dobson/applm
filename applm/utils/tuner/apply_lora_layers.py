import types
from pathlib import Path
from mlx import nn
import json
from .linear_to_lora_layers import linear_to_lora_layers


def apply_lora_layers(model: nn.Module, adapter_path: str) -> nn.Module:
    """
    Apply LoRA layers to the model.

    Args:
        model (nn.Module): The neural network model.
        adapter_path (str): Path to the adapter configuration file.

    Returns:
        nn.Module: The updated model with LoRA layers applied.
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"The adapter path does not exist: {adapter_path}")
    with open(adapter_path / "adapter_config.json", "r") as fid:
        config = types.SimpleNamespace(**json.load(fid))
    linear_to_lora_layers(
        model,
        config.lora_layers,
        config.lora_parameters,
        getattr(config, "use_dora", False),
    )
    model.load_weights(str(adapter_path / "adapters.safetensors"), strict=False)
    return model
