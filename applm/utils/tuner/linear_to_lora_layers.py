from mlx import nn
from typing import Dict
from mlx.utils import tree_unflatten
from .dora import DoRALinear
from .LoRALinear import LoRALinear
from ..models.switch_layers.SwitchLinear import SwitchLinear
from ..models.switch_layers.QuantizedSwitchLinear import QuantizedSwitchLinear
from .LoRASwitchLinear import LoRASwitchLinear


def linear_to_lora_layers(
    model: nn.Module,
    num_lora_layers: int,
    config: Dict,
    use_dora: bool = False,
):
    """
    Convert some of the models linear layers to lora layers.

    Args:
        model (nn.Module): The neural network model.
        num_lora_layers (int): The number of blocks to convert to lora layers
        starting from the last layer.
        config (dict): More configuration parameters for LoRA, including the
          rank, scale, and optional layer keys.
        use_dora (bool): If True, uses DoRA instead of LoRA.
          Default: ``False``
    """

    num_layers = len(model.layers)

    if num_lora_layers < 0:
        num_lora_layers = num_layers

    if num_lora_layers > num_layers:
        raise ValueError(
            f"Requested {num_lora_layers} LoRA layers "
            f"but the model only has {num_layers} layers."
        )

    def to_lora(layer):
        if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
            LoRALayer = DoRALinear if use_dora else LoRALinear
        elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
            if use_dora:
                raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.")
            LoRALayer = LoRASwitchLinear
        else:
            raise ValueError(
                f"Can't convert layer of type {type(layer).__name__} to LoRA"
            )

        return LoRALayer.from_linear(
            layer,
            r=config["rank"],
            scale=config["scale"],
            dropout=config["dropout"],
        )

    keys = config.get("keys", None)
    if keys is not None:
        keys = set(keys)
    elif model.model_type in [
        "mistral",
        "llama",
        "phi",
        "mixtral",
        "stablelm",
        "qwen2",
        "qwen2_moe",
        "gemma",
        "gemma2",
        "starcoder2",
        "cohere",
        "minicpm",
    ]:
        keys = set(["self_attn.q_proj", "self_attn.v_proj"])
        if model.model_type == "mixtral":
            keys.add("block_sparse_moe.gate")
        if model.model_type == "qwen2_moe":
            keys.add("mlp.gate")
            keys.add("mlp.shared_expert_gate")

    elif model.model_type == "gpt_bigcode":
        keys = set(["attn.c_attn"])
    elif model.model_type == "gpt2":
        keys = set(["attn.c_attn"])
    elif model.model_type == "gpt_neox":
        keys = set(["attention.query_key_value"])
    elif model.model_type == "olmo":
        keys = set(["att_proj"])
    elif model.model_type == "openelm":
        keys = set(["attn.qkv_proj"])
    elif model.model_type == "phi3":
        keys = set(["self_attn.qkv_proj"])
    elif model.model_type == "phi-msft":
        keys = set(["mixer.Wqkv", "moe.gate"])
    elif model.model_type == "dbrx":
        keys = set(["norm_attn_norm.attn.Wqkv", "ffn.router.layer"])
    elif model.model_type == "internlm2":
        keys = set(["attention.wqkv", "attention.wo"])
    elif model.model_type == "deepseek_v2":
        keys = set(
            [
                "self_attn.q_proj",
                "self_attn.q_a_proj",
                "self_attn.q_b_proj",
                "self_attn.kv_a_proj_with_mqa",
                "self_attn.kv_b_proj",
            ]
        )
    else:
        raise ValueError(f"Lora does not support {model.model_type}")

    for l in model.layers[num_layers - num_lora_layers :]:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
        l.update_modules(tree_unflatten(lora_layers))
