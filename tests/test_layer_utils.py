import json
import os
import unittest
from unittest.mock import (
    MagicMock,
    patch,
)

from erisforge.utils.layer_utils import (
    get_layers_names_by_model,
    identify_model,
)


class TestLayerUtils(unittest.TestCase):
    def setUp(self):
        path = (
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            + "/erisforge/assets/llm_models.json"
        )
        with open(path, "r") as f:
            self.models = json.load(f)

    def test_get_layers_names_by_model_returns_correct_layers(self):
        for model_name, layers in self.models.items():
            layer_names = get_layers_names_by_model(model_name)
            self.assertDictEqual(d1=layer_names, d2=layers)

    def test_get_layers_names_by_model_raises_value_error_for_unknown_model(self):
        with self.assertRaises(ValueError):
            get_layers_names_by_model("unknown_model")

    def test_get_layers_names_by_model_returns_all_models_when_no_model_name_provided(
        self,
    ):
        all_models = get_layers_names_by_model()
        self.assertDictEqual(d1=all_models, d2=self.models)

    def test_identify_model_returns_correct_model_name(self):
        custom_model = MagicMock()
        custom_model.named_parameters.return_value = [
            ("layer1", None),
            ("layer2", None),
        ]
        with patch(
            "erisforge.utils.layer_utils.get_layers_names_by_model"
        ) as mock_get_layers:
            mock_get_layers.return_value = {
                "model_a": {"self_attention": "layer1", "mlp": "layer2"}
            }
            result = identify_model(custom_model)
            self.assertEqual(result, "model_a")

    def test_identify_model_raises_value_error_for_unknown_model(self):
        custom_model = MagicMock()
        custom_model.named_parameters.return_value = [
            ("layer5", None),
            ("layer6", None),
        ]
        with patch(
            "erisforge.utils.layer_utils.get_layers_names_by_model"
        ) as mock_get_layers:
            mock_get_layers.return_value = {
                "model_a": {"self_attention": "layer1", "mlp": "layer2"}
            }
            with self.assertRaises(ValueError):
                identify_model(custom_model)
