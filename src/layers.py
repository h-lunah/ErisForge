from einops import (
    einops,
)
import jaxtyping
import torch


class AblationDecoderLayer(torch.nn.Module):
    def __init__(self, original_layer, direction):
        super(AblationDecoderLayer, self).__init__()
        self.original_layer = original_layer
        self.objective_behaviour_dr = direction

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        ablated = self._direction_ablation_hook(
            activation=hidden_states,
            direction=self.objective_behaviour_dr.to(hidden_states.device),
        ).to(hidden_states.device)
        args = (ablated,) + args[1:]
        return self.original_layer.forward(*args, **kwargs)

    @staticmethod
    def _direction_ablation_hook(
            activation: jaxtyping.Float[torch.Tensor, "... d_act"],
            direction: jaxtyping.Float[torch.Tensor, "d_act"]
    ) -> torch.Tensor:
        proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
        return activation - proj


class AdditionDecoderLayer(torch.nn.Module):
    def __init__(self, original_layer, direction):
        super(AdditionDecoderLayer, self).__init__()
        self.original_layer = original_layer
        self.objective_behaviour_dir = direction

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        added = self._direction_addition_hook(
            activation=hidden_states,
            direction=self.objective_behaviour_dir.to(hidden_states.device),
        ).to(hidden_states.device)
        args = (added,) + args[1:]
        return self.original_layer.forward(*args, **kwargs)

    @staticmethod
    def _direction_addition_hook(
            activation: jaxtyping.Float[torch.Tensor, "... d_act"],
            direction: jaxtyping.Float[torch.Tensor, "d_act"]
    ) -> torch.Tensor:
        return activation + direction

