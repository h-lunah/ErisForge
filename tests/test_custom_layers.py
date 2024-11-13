import unittest

import torch

from erisforge.layers.layers import (
    AblationDecoderLayer,
    AdditionDecoderLayer,
)


class TestCustomLayers(unittest.TestCase):
    def setUp(self):
        self.original_layer = torch.nn.Linear(10, 10)
        self.direction = torch.randn(10)

    def test_ablation_decoder_layer_ablation(self):
        layer = AblationDecoderLayer(self.original_layer, self.direction)
        input_tensor = torch.randn(5, 10)
        output = layer(input_tensor)
        self.assertEqual(output.shape, (5, 10))

    def test_addition_decoder_layer_addition(self):
        layer = AdditionDecoderLayer(self.original_layer, self.direction)
        input_tensor = torch.randn(5, 10)
        output = layer(input_tensor)
        self.assertEqual(output.shape, (5, 10))

    def test_ablation_decoder_layer_no_ablation(self):
        layer = AblationDecoderLayer(self.original_layer, torch.zeros(10))
        input_tensor = torch.randn(5, 10)
        output = layer(input_tensor)
        self.assertTrue(torch.allclose(output, self.original_layer(input_tensor)))

    def test_addition_decoder_layer_no_addition(self):
        layer = AdditionDecoderLayer(self.original_layer, torch.zeros(10))
        input_tensor = torch.randn(5, 10)
        output = layer(input_tensor)
        self.assertTrue(torch.allclose(output, self.original_layer(input_tensor)))
