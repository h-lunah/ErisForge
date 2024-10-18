import logging
import random
from typing import List, Dict

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM


class Forge:

    def __init__(self):
        self.max_iterations: int = 0
        self.positive_behaviour_instructions: List[str] = []
        self.negative_behaviour_instructions: List[str] = []
        if torch.backends.mps.is_available():
            logging.info("MPS is available.")
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            logging.info("CUDA is available.")
            self.device = torch.device("cuda")
        else:
            logging.info("CPU is available.")
            self.device = torch.device("cpu")


    def load_instructions(self, positive_behaviour_instructions: List[str], negative_behaviour_instructions: List[str]):
        logging.info(f"Loading instructions, positive: {len(positive_behaviour_instructions)}, negative: {len(negative_behaviour_instructions)}")
        self.positive_behaviour_instructions: List[str] = positive_behaviour_instructions
        self.negative_behaviour_instructions: List[str] = negative_behaviour_instructions
        self.max_iterations: int = len(positive_behaviour_instructions) + len(negative_behaviour_instructions)
        logging.info(f"Instructions loaded, positive: {len(positive_behaviour_instructions)}, negative: {len(negative_behaviour_instructions)}")

    @staticmethod
    def _tokenize(tokenizer: PreTrainedTokenizerBase, instruction: str, bar: tqdm | None = None) -> torch.Tensor:
        tokens: torch.Tensor = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": instruction}],
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if bar:
            bar.update(n=1)

        return tokens

    def tokenize_instructions(
            self,
            tokenizer: PreTrainedTokenizerBase | AutoTokenizer | str,
            max_n_positive_instruction:  int | None = None,
            max_n_negative_instruction:  int | None = None,
    ) -> Dict[str, List[Tensor]]:
        if isinstance(tokenizer, str):
            logging.info(f"Loading tokenizer from {tokenizer}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

        max_n_positive_instruction = min(len(self.positive_behaviour_instructions), max_n_positive_instruction) if max_n_positive_instruction else len(self.positive_behaviour_instructions)
        max_n_negative_instruction = min(len(self.negative_behaviour_instructions), max_n_negative_instruction) if max_n_negative_instruction else len(self.negative_behaviour_instructions)

        positive_behaviour_instructions = random.sample(self.positive_behaviour_instructions, max_n_positive_instruction)
        negative_behaviour_instructions = random.sample(self.negative_behaviour_instructions, max_n_negative_instruction)

        logging.info(f'For tokenization, using {max_n_positive_instruction/len(self.positive_behaviour_instructions)*100:.2f}% positive instructions.')
        logging.info(f'For tokenization, using {max_n_negative_instruction/len(self.negative_behaviour_instructions)*100:.2f}% negative instructions.')

        logging.info('Tokenizing Positive instructions...')
        with tqdm(total=max_n_positive_instruction, desc='Tokenizing Positive instructions') as bar:
            positive_instr_tokens: List[torch.Tensor] = [
                self._tokenize(tokenizer=tokenizer, instruction=positive_behaviour_instruction, bar=bar)
                for positive_behaviour_instruction in positive_behaviour_instructions
            ]

        logging.info('Tokenizing Negative instructions...')
        with tqdm(total=max_n_negative_instruction, desc='Tokenizing Negative instructions') as bar:
            negative_instr_tokens: List[torch.Tensor] = [
                self._tokenize(tokenizer=tokenizer, instruction=negative_behaviour_instruction, bar=bar)
                for negative_behaviour_instruction in negative_behaviour_instructions
            ]
        logging.info('Tokenization complete.')

        return {
            'positive_tokens': positive_instr_tokens,
            'negative_tokens': negative_instr_tokens
        }

    def _generate_new_tok(self, model: AutoModelForCausalLM, tokens: Tensor, bar: tqdm):
        bar.update(n=1)
        return model.generate(
            tokens.to(self.device),
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    def compute_positive_direction(
            self,
            model: AutoModelForCausalLM | str,
            positive_behaviour_tokenized_instructions: List[Tensor],
            negative_behaviour_instructions: List[Tensor],
            layer: str | int,
    ) -> Tensor:
        if isinstance(model, str):
            logging.info(f"Loading model from {model}")
            model: AutoModelForCausalLM  = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
        else:
            model.to(self.device)

        if isinstance(layer, str):
            if layer == "auto":
                layer = int(len(model.model.layers) * 0.6)
                logging.info(f"Using layer {layer} for computing positive direction.")
            elif layer=="best":
                logging.info("Searching for best layer for computing positive direction.")
                raise NotImplementedError("Search of best layer not implemented yet.")
            else:
                raise ValueError(f"Invalid layer value: {layer}, must be 'auto', 'best' or an integer.")
        else:
            logging.info(f"Using layer {layer} for computing positive direction.")

        logging.info("Generating tokens on positive instructions.")
        with tqdm(total=len(positive_behaviour_tokenized_instructions), desc="Generating tokens on positive instructions") as bar:
            positive_outputs = [
                self._generate_new_tok(model, positive_behaviour_tokenized_instruction, bar)
                for positive_behaviour_tokenized_instruction in positive_behaviour_tokenized_instructions
            ]
        logging.info('Completed generating tokens on positive instructions.')

        logging.info("Generating tokens on negative instructions.")
        with tqdm(total=len(negative_behaviour_instructions), desc="Generating tokens on negative instructions") as bar:
            negative_outputs = [
                self._generate_new_tok(model, negative_behaviour_instruction, bar)
                for negative_behaviour_instruction in negative_behaviour_instructions
            ]
        logging.info('Completed generating tokens on negative instructions.')

        logging.info("Computing positive direction.")
        positive_mean = torch.stack([output.hidden_states[0][layer][:, -1, :] for output in positive_outputs]).mean(dim=0)
        negative_mean = torch.stack([output.hidden_states[0][layer][:, -1, :] for output in negative_outputs]).mean(dim=0)

        positive_dir = positive_mean - negative_mean
        positive_dir = positive_dir / positive_dir.norm()

        return positive_dir


if __name__ == "__main__":
    MODEL = 'google/gemma-1.1-2b-it'
    with open("harmful.txt", "r") as f:
        pos = f.readlines()

    with open("harmless.txt", "r") as f:
        neg = f.readlines()

    logging.basicConfig(level=logging.INFO)
    forge = Forge()
    forge.load_instructions(positive_behaviour_instructions=pos, negative_behaviour_instructions=neg)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    d_toks = forge.tokenize_instructions(tokenizer=tokenizer, max_n_negative_instruction=100, max_n_positive_instruction=100)

    refusal_dir = forge.compute_positive_direction(
        model=MODEL,
        positive_behaviour_tokenized_instructions=d_toks['positive_tokens'],
        negative_behaviour_instructions=d_toks['negative_tokens'],
        layer="auto"
    )