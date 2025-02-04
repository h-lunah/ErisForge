import torch
import einops

class AblationDecoderLayer(torch.nn.Module):
    def __init__(self, original_layer, direction):
        """
        Initialize the AblationDecoderLayer.
        :param original_layer: the original layer to be ablated
        :param direction: the direction of the ablation.
        """
        super(AblationDecoderLayer, self).__init__()
        self.original_layer = original_layer
        self.register_buffer("objective_behaviour_dr", direction)

    def forward(self, *args, **kwargs):
        """
        Forward method of the AblationDecoderLayer.
        :param args: the arguments to be passed to the original layer (first argument should be the activations).
        :param kwargs: additional keyword arguments for the original layer.
        :return: the output of the original layer after ablation.
        """
        hidden_states = args[0]
        direction = self.objective_behaviour_dr.to(hidden_states.device, non_blocking=True)
        direction = direction.to(hidden_states.dtype)
        ablated = self._direction_ablation_hook(
            activation=hidden_states,
            direction=direction,
        )
        args = (ablated,) + args[1:]
        return self.original_layer(*args, **kwargs)

    @staticmethod
    def _direction_ablation_hook(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Ablation hook for the AblationDecoderLayer.
        :param activation: the activation to be ablated.
        :param direction: the direction of the ablation (already on the correct device/dtype).
        :return: the ablated activation.
        """
        direction = direction.to(activation.dtype)
        proj = einops.einsum(
            activation,
            direction.view(-1, 1),
            "... d_act, d_act single -> ... single",
        ) * direction
        return activation - proj


class AdditionDecoderLayer(torch.nn.Module):
    def __init__(self, original_layer, direction):
        """
        Initialize the AdditionDecoderLayer.
        :param original_layer: the original layer to be added to.
        :param direction: the direction of the addition.
        """
        super(AdditionDecoderLayer, self).__init__()
        self.original_layer = original_layer
        self.register_buffer("objective_behaviour_dir", direction)

    def forward(self, *args, **kwargs):
        """
        Forward method of the AdditionDecoderLayer.
        :param args: the arguments to be passed to the original layer (first argument should be the activations).
        :param kwargs: additional keyword arguments for the original layer.
        :return: the output of the original layer after addition.
        """
        hidden_states = args[0]
        direction = self.objective_behaviour_dir.to(hidden_states.device, non_blocking=True)
        direction = direction.to(hidden_states.dtype)
        added = self._direction_addition_hook(
            activation=hidden_states,
            direction=direction,
        )
        args = (added,) + args[1:]
        return self.original_layer(*args, **kwargs)

    @staticmethod
    def _direction_addition_hook(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Addition hook for the AdditionDecoderLayer.
        :param activation: the activation to be added to.
        :param direction: the direction of the addition.
        :return: the modified activation.
        """
        direction = direction.to(activation.dtype)
        return activation + direction