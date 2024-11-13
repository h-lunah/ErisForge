import os
import unittest
from unittest.mock import (
    MagicMock,
    patch,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from erisforge.eris_forge import (
    Forge,
)


class TestForge(unittest.TestCase):
    def setUp(self):
        self.forge = Forge()
        self.model_name = "microsoft/Phi-3-small-128k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = MagicMock(spec=AutoModelForCausalLM)
        self.model.to = MagicMock(return_value=self.model)
        self.model.generate = MagicMock(return_value=MagicMock())
        self.objective_behaviour_instructions = ["Generate a positive review."]
        self.anti_behaviour_instructions = ["Generate a negative review."]
        self.forge.load_instructions(
            self.objective_behaviour_instructions, self.anti_behaviour_instructions
        )

    @patch("erisforge.eris_forge.AutoTokenizer.from_pretrained")
    def test_tokenize_instructions(self, mock_tokenizer):
        mock_tokenizer.return_value = self.tokenizer
        tokenized_instructions = self.forge.tokenize_instructions(self.tokenizer)
        self.assertIn("objective_behaviour_tokens", tokenized_instructions)
        self.assertIn("antiobjective_tokens", tokenized_instructions)
        self.assertEqual(len(tokenized_instructions["objective_behaviour_tokens"]), 1)
        self.assertEqual(len(tokenized_instructions["antiobjective_tokens"]), 1)

    @patch("erisforge.eris_forge.AutoModelForCausalLM.from_pretrained")
    def test_compute_output(self, mock_model):
        mock_model.return_value = self.model
        tokenized_instructions = self.forge.tokenize_instructions(self.tokenizer)
        outputs = self.forge.compute_output(
            model=self.model,
            objective_behaviour_tokenized_instructions=tokenized_instructions[
                "objective_behaviour_tokens"
            ],
            anti_behaviour_tokenized_instructions=tokenized_instructions[
                "antiobjective_tokens"
            ],
        )
        self.assertIn("obj_beh", outputs)
        self.assertIn("anti_obj", outputs)
        self.assertEqual(len(outputs["obj_beh"]), 1)
        self.assertEqual(len(outputs["anti_obj"]), 1)
