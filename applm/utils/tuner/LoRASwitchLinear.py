# Copyright © 2024 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn


from ..models.switch_layers.SwitchLinear import SwitchLinear
from ..models.switch_layers.QuantizedSwitchLinear import QuantizedSwitchLinear


class LoRASwitchLinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Module,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ) -> "LoRASwitchLinear":
        lora_lin = LoRASwitchLinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            num_experts=linear.num_experts,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def to_linear(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, QuantizedSwitchLinear)

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )
        num_experts, output_dims, input_dims = weight.shape
        fused_linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        lora_b = (self.scale * self.lora_b).astype(dtype)
        lora_a = self.lora_a.reshape(num_experts, -1, input_dims).astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized and not de_quantize:
            fused_linear = fused_linear.to_quantized(linear.group_size, linear.bits)

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(r * num_experts, input_dims),
        )
        self.lora_b = mx.zeros(shape=(num_experts, output_dims, r))
        self.num_experts = num_experts

    def __call__(self, x, indices):
        shape = x.shape[:-3] + (self.num_experts, -1)

        y = self.linear(x, indices)
        z = (self.dropout(x) @ self.lora_a.T).reshape(shape)
        z = mx.take_along_axis(z, indices[..., None], axis=-2)
        z = z[..., None, :] @ self.lora_b[indices].swapaxes(-2, -1)

        return y + (self.scale * z).astype(x.dtype)
