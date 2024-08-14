import math

from mlx import core as mx
import mlx.nn as nn

from SwitchLinear import SwitchLinear


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.silu,
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        x_up = self.up_proj(x, indices)
        x_gate = self.gate_proj(x, indices)
        x = self.down_proj(self.activation(x_gate) * x_up, indices)

        return x.squeeze(-2)
