from torch import nn


class AblationDecoderLayer(nn.Module):
    def __init__(self, original_layer, refusal_dir):
        super(AblationDecoderLayer, self).__init__()
        self.original_layer = original_layer
        self.refusal_dir = refusal_dir

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        ablated = self._direction_ablation_hook(hidden_states, self.refusal_dir.to(hidden_states.device)).to(
            hidden_states.device)
        args = (ablated,) + args[1:]
        return self.original_layer.forward(*args, **kwargs)
