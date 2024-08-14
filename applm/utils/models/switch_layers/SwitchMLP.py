from mlx import core as mx
import mlx.nn as nn
from SwitchLinear import SwitchLinear


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.gelu_approx,
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        x = self.fc1(x, indices)
        x = self.activation(x)
        x = self.fc2(x, indices)

        return x.squeeze(-2)
