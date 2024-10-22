import logging
import random
from random import sample
from typing import List, Dict, Any, Type

import torch
from torch import Tensor, nn
from tqdm import tqdm, trange
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM, TextStreamer, PreTrainedModel
from transformers.generation import GenerateDecoderOnlyOutput

from src.layers import AblationDecoderLayer, AdditionDecoderLayer
from src.scorers.base_scorer import BaseScorer


class Forge:

    def __init__(self):
        self.max_toks = 1
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
                    n_generated_tokens=self.max_toks,
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
                    n_generated_tokens=self.max_toks,
                )
                for negative_behaviour_instruction in negative_behaviour_tokenized_instructions
            ]
        logging.info('Completed generating tokens on negative instructions.')

        return {
            'pos': positive_outputs,
            'neg': negative_outputs,
        }

    def compute_positive_metric(self):

    def find_approximate_best_positive_direction(
            self,
            model: AutoModelForCausalLM | PreTrainedModel,
            scorer: BaseScorer,
            eval_positive_instructions: List[str],
            eval_negative_instructions: List[str],
            min_layer: int | None = None,
            max_layer: int | None = None,
    ) -> Tensor:
        if min_layer is None:
            min_layer = max(int(len(model.model.layers) * 0.2), 1)
        if max_layer is None:
            max_layer = min(int(len(model.model.layers) * 0.8), len(model.model.layers)-2)

        score_x_layer = []

        with tqdm(total=len(eval_positive_instructions), desc='Tokenizing Positive Eval Instructions set') as bar:
            pos_toks = [
                self._tokenize(tokenizer=tokenizer, instruction=instr, bar=bar)
                for instr in eval_positive_instructions
            ]
        with tqdm(total=len(eval_negative_instructions), desc='Tokenizing Negative Eval Instructions set') as bar:
            neg_toks = [
                self._tokenize(tokenizer=tokenizer, instruction=instr, bar=bar)
                for instr in eval_negative_instructions
            ]

        d_out = self.compute_output(
            model=model,
            positive_behaviour_tokenized_instructions=pos_toks,
            negative_behaviour_tokenized_instructions=neg_toks,
        )

        for layer_idx in trange(min_layer, max_layer, desc='Finding best positive direction'):
            tmp_pos_dir = self.compute_positive_direction(
                positive_outputs=d_out['pos'],
                negative_outputs=d_out['neg'],
                layer=layer_idx,
            )

            conversations_ablated = self.run_forged_model(
                model=model,
                type_of_layer=AblationDecoderLayer,
                positive_dir=tmp_pos_dir,
                tokenizer=tokenizer,
                min_layer=min_layer,
                max_layer=max_layer,
                instructions=eval_positive_instructions,
                max_new_tokens=100,
                stream=False,
            )

            conversations_added = self.run_forged_model(
                model=model,
                type_of_layer=AdditionDecoderLayer,
                positive_dir=tmp_pos_dir,
                tokenizer=tokenizer,
                min_layer=layer_idx,
                max_layer=layer_idx+1,
                instructions=eval_negative_instructions,
                max_new_tokens=100,
                stream=False,
            )

            positive_score = sum(
                        [
                            scorer.score(
                                model_response=conv[-1]['content'],
                                user_query=conv[-2]['content'],
                            ) for conv in conversations_ablated
                        ]
                    )
            negative_score = 1 - sum(
                        [
                            scorer.score(
                                model_response=conv[-1]['content'],
                                user_query=conv[-2]['content'],
                            ) for conv in conversations_added
                        ]
                    )

            score_x_layer.append(
                {
                    'layer': layer_idx,
                    'score': (positive_score - negative_score)/2,
                    'dir': tmp_pos_dir,
                }
            )
        score_x_layer = sorted(score_x_layer, key=lambda x: x['score'], reverse=True)
        return score_x_layer[0]['dir']

    def _replace_layers(
            self,
            new_layer: Type[torch.nn.Module],
            max_layer: int,
            min_layer: int,
            model: AutoModelForCausalLM | PreTrainedModel,
            direction: Tensor,
    ):
        for layer_idx in trange(min_layer, max_layer, desc='Ablating model layers'):
            if isinstance(model.model.layers[layer_idx], AblationDecoderLayer) or isinstance(
                    model.model.layers[layer_idx], AdditionDecoderLayer):
                model.model.layers[layer_idx] = new_layer(
                    original_layer=model.model.layers[layer_idx].original_layer,
                    refusal_dir=direction,
                )
            else:
                model.model.layers[layer_idx] = new_layer(
                    original_layer=model.model.layers[layer_idx],
                    refusal_dir=direction,
                )
        return model

    def compute_positive_direction(
            self,
            positive_outputs: List[GenerateDecoderOnlyOutput],
            negative_outputs: List[GenerateDecoderOnlyOutput],
            layer: int | None = None,
    ) -> Tensor:
        if layer is None:
            layer = int(len(model.model.layers) * 0.6)
        positive_mean = torch.stack([output.hidden_states[0][layer][:, -self.max_toks:, :].mean(dim=1) for output in positive_outputs]).mean(dim=0)
        negative_mean = torch.stack([output.hidden_states[0][layer][:, -self.max_toks:, :].mean(dim=1) for output in negative_outputs]).mean(dim=0)

        positive_dir = positive_mean - negative_mean
        positive_dir = positive_dir / positive_dir.norm()

        return positive_dir


    def run_forged_model(
            self,
            model: AutoModelForCausalLM | PreTrainedModel,
            type_of_layer: Type[torch.nn.Module],
            positive_dir: Tensor,
            tokenizer: PreTrainedTokenizerBase | AutoTokenizer,
            min_layer: int,
            max_layer: int,
            instructions: List[str] | None = None,
            tokenized_instructions: List[Tensor] | None = None,
            max_new_tokens: int = 100,
            stream: bool = False,
    ) -> List[List[Dict[str, Any]]]:

        new_model = self._replace_layers(
            new_layer=type_of_layer,
            max_layer=max_layer,
            min_layer=min_layer,
            model=model,
            direction=positive_dir,
        )

        if tokenized_instructions:
            logging.info('Using provided tokenized instructions. No need to tokenize again.')
            instr_tokens = tokenized_instructions
        elif instructions:
            logging.info('Tokenizing instructions for newly forged model.')
            with tqdm(total=len(instructions), desc='Tokenizing instructions for newly forged model') as bar:
                instr_tokens: List[torch.Tensor] = [
                    self._tokenize(tokenizer=tokenizer, instruction=instruction, bar=bar)
                    for instruction in instructions
                ]
        else:
            raise ValueError('Either instructions or tokenized instructions must be provided.')


        logging.info('Generating tokens for newly forged model.')
        with tqdm(total=len(instructions), desc='Generating tokens for newly forged model') as bar:
            encoded_responses = [
                self._generate_new_tokens(
                    model=new_model,
                    tokens=instr_token,
                    bar=bar,
                    n_generated_tokens=max_new_tokens,
                    streamer=TextStreamer(tokenizer) if stream else None,
                )
                for instr_token in instr_tokens
            ]

        conversations: List[List[Dict[str, Any]]] = []
        for enc_resp, instr in zip(encoded_responses, instructions):
            conversations.append(
                [
                    {"role": "user", "content": instr},
                    {"role": "assistant", "content": tokenizer.decode(enc_resp.sequences[0].tolist(), skip_special_tokens=True)}
                ]
            )

        return conversations


if __name__ == "__main__":
    random.seed(42)
    MODEL = 'google/gemma-1.1-2b-it'
    with open("./harmful.txt", "r") as f:
        pos = f.readlines()

    with open("./harmless.txt", "r") as f:
        neg = f.readlines()

    max_inst = 100
    logging.basicConfig(level=logging.INFO)

    forge = Forge()
    forge.load_instructions(positive_behaviour_instructions=pos, negative_behaviour_instructions=neg)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(forge.device)

    d_toks = forge.tokenize_instructions(
        tokenizer=tokenizer,
        max_n_negative_instruction=max_inst,
        max_n_positive_instruction=max_inst,
    )

    d_instr = forge.compute_output(
        model=model,
        positive_behaviour_tokenized_instructions=d_toks['positive_tokens'],
        negative_behaviour_tokenized_instructions=d_toks['negative_tokens'],
    )

    refusal_dir = forge.compute_positive_direction(
        positive_outputs=d_instr['pos'],
        negative_outputs=d_instr['neg'],
    )

    conversations = forge.run_forged_model(
        model=model,
        positive_dir=refusal_dir,
        tokenizer=tokenizer,
        instructions=sample(population=pos, k=20),
        max_new_tokens=100,
        stream=False,
    )

    for conversation in conversations:
        print('='*20)
        for round in conversation:
            print(f'{round["role"]}: {round["content"]}')

