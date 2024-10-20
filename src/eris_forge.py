import logging
import random
from typing import List, Dict, Any

import jaxtyping
import torch
from einops import einops
from torch import Tensor, nn
from tqdm import tqdm, trange
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM, TextStreamer
from transformers.generation import GenerateDecoderOnlyOutput

from src.layers import AblationDecoderLayer


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

    def _generate_new_tokens(
            self,
            model: AutoModelForCausalLM,
            tokens: Tensor,
            bar: tqdm | None = None,
            n_generated_tokens: int = 1,
            streamer: TextStreamer | None = None,
    ) -> GenerateDecoderOnlyOutput:
        if bar:
            bar.update(n=1)

        params = {
            "inputs": tokens.to(self.device),
            "use_cache": False,
            "max_new_tokens": n_generated_tokens,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
        }

        if streamer:
            params["streamer"] = streamer

        return model.generate(**params)

    def compute_output(
            self,
            model: AutoModelForCausalLM | str,
            positive_behaviour_tokenized_instructions: List[Tensor],
            negative_behaviour_tokenized_instructions: List[Tensor],
    ) -> Dict[str, List[GenerateDecoderOnlyOutput]]:
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

        logging.info("Generating tokens on positive instructions.")
        with tqdm(total=len(positive_behaviour_tokenized_instructions), desc="Generating tokens on positive instructions") as bar:
            positive_outputs = [
                self._generate_new_tokens(
                    model=model,
                    tokens=positive_behaviour_tokenized_instruction,
                    bar=bar,
                    n_generated_tokens= 10 if tokenizer else 1,
                )
                for positive_behaviour_tokenized_instruction in positive_behaviour_tokenized_instructions
            ]
        logging.info('Completed generating tokens on positive instructions.')

        logging.info("Generating tokens on negative instructions.")
        with tqdm(total=len(negative_behaviour_tokenized_instructions), desc="Generating tokens on negative instructions") as bar:
            negative_outputs = [
                self._generate_new_tokens(
                    model=model,
                    tokens=negative_behaviour_instruction,
                    bar=bar,
                    n_generated_tokens= 10 if tokenizer else 1,
                )
                for negative_behaviour_instruction in negative_behaviour_tokenized_instructions
            ]
        logging.info('Completed generating tokens on negative instructions.')

        return {
            'pos': positive_outputs,
            'neg': negative_outputs,
        }

    def compute_best_layer(
            self,
            model,

    ):
        pass

    def compute_positive_direction(
            self,
            positive_outputs: List[GenerateDecoderOnlyOutput],
            negative_outputs: List[GenerateDecoderOnlyOutput],
            layer: str,
    ) -> Tensor:
        positive_mean = torch.stack([output.hidden_states[0][layer][:, -1, :] for output in positive_outputs]).mean(dim=0)
        negative_mean = torch.stack([output.hidden_states[0][layer][:, -1, :] for output in negative_outputs]).mean(dim=0)

        positive_dir = positive_mean - negative_mean
        positive_dir = positive_dir / positive_dir.norm()

        return positive_dir

    @staticmethod
    def _direction_ablation_hook(
            activation: jaxtyping.Float[torch.Tensor, "... d_act"],
            direction: jaxtyping.Float[torch.Tensor, "d_act"]
    ) -> Tensor:
        proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
        return activation - proj


    def run_forged_model(
            self,
            model: AutoModelForCausalLM,
            refusal_dir: Tensor,
            tokenizer: PreTrainedTokenizerBase | AutoTokenizer,
            instructions: List[str],
            max_new_tokens: int = 100,
            stream: bool = False,
    ) -> List[List[Dict[str, Any]]]:

        for layer_idx in trange(len(model.model.layers), desc='Ablating model layers'):
            model.model.layers[layer_idx] = AblationDecoderLayer(
                original_layer=model.model.layers[layer_idx],
                refusal_dir=refusal_dir,
            )

        logging.info('Tokenizing instructions for newly forged model.')
        with tqdm(total=len(instructions), desc='Tokenizing instructions for newly forged model') as bar:
            instr_tokens: List[torch.Tensor] = [
                self._tokenize(tokenizer=tokenizer, instruction=instruction, bar=bar)
                for instruction in instructions
            ]

        logging.info('Generating tokens for newly forged model.')
        with tqdm(total=len(instructions), desc='Generating tokens for newly forged model') as bar:
            encoded_responses = [
                self._generate_new_tokens(
                    model=model,
                    tokens=instr_token,
                    bar=bar,
                    n_generated_tokens=max_new_tokens,
                    streamer=TextStreamer(tokenizer) if stream else None,
                )
                for instr_token in instr_tokens
            ]

        conversations: List[List[Dict[str, Any]]] = []
        for enc_resp, instr in encoded_responses, instructions:
            conversations.append(
                [
                    {"role": "user", "content": instr},
                    {"role": "assistant", "content": tokenizer.batch_decode(enc_resp, skip_special_tokens=True)}
                ]
            )

        return conversations


if __name__ == "__main__":
    MODEL = 'google/gemma-1.1-2b-it'
    with open("harmful.txt", "r") as f:
        pos = f.readlines()

    with open("harmless.txt", "r") as f:
        neg = f.readlines()
    max_inst = 1
    logging.basicConfig(level=logging.INFO)
    forge = Forge()
    forge.load_instructions(positive_behaviour_instructions=pos, negative_behaviour_instructions=neg)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    d_toks = forge.tokenize_instructions(tokenizer=tokenizer, max_n_negative_instruction=max_inst, max_n_positive_instruction=max_inst)