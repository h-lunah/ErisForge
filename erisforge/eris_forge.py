import logging
import random
from typing import (
    Any,
    Dict,
    List,
    Type,
)

import torch
from torch import (
    Tensor,
)
from tqdm import (
    tqdm,
    trange,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
)

from erisforge.layers.layers import (
    AblationDecoderLayer,
    AdditionDecoderLayer,
)
from erisforge.scorers.base_scorer import (
    BaseScorer,
)
from erisforge.utils.layer_utils import (
    get_layers_names_by_model,
    identify_model,
)


class Forge:
    def __init__(self):
        """
        Initializes the Forge object.
        """
        self.max_toks = 1
        self.max_iterations: int = 0
        self.objective_behaviour_instructions: List[str] = []
        self.anti_behaviour_instructions: List[str] = []
        if torch.backends.mps.is_available():
            logging.info("MPS is available.")
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            logging.info("CUDA is available.")
            self.device = torch.device("cuda")
        else:
            logging.info("CPU is available.")
            self.device = torch.device("cpu")

    def load_instructions(
        self,
        objective_behaviour_instructions: List[str],
        anti_behaviour_instructions: List[str],
    ) -> None:
        """
        Loads the instructions for the Forge object.
        :param objective_behaviour_instructions: List of the instructions asking the model to perform the behaviour you want.
        :param anti_behaviour_instructions: List of instructions that ask the model to perform random, usual, tasks.
        :return: None
        """
        logging.info(
            f"Loading instructions, objective_behaviour: {len(objective_behaviour_instructions)}, antiobjective: {len(anti_behaviour_instructions)}"
        )
        self.objective_behaviour_instructions: List[str] = (
            objective_behaviour_instructions
        )
        self.anti_behaviour_instructions: List[str] = anti_behaviour_instructions
        self.max_iterations: int = len(objective_behaviour_instructions) + len(
            anti_behaviour_instructions
        )
        logging.info(
            f"Instructions loaded, objective_behaviour: {len(objective_behaviour_instructions)}, antiobjective: {len(anti_behaviour_instructions)}"
        )

    @staticmethod
    def _tokenize(
        tokenizer: PreTrainedTokenizerBase,
        instruction: str,
        bar: tqdm | None = None,
    ) -> torch.Tensor:
        """
        Tokenizes the instruction.
        :param tokenizer: Tokenizer for a particular model.
        :param instruction: Instruction to be tokenized.
        :param bar: Progress bar object.
        :return: Tokenized instruction in the form of a tensor.
        """
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
        max_n_objective_behaviour_instruction: int | None = None,
        max_n_antiobjective_instruction: int | None = None,
    ) -> Dict[str, List[Tensor]]:
        """
        Tokenizes the instructions.
        :param tokenizer: Tokenizer for a particular model.
        :param max_n_objective_behaviour_instruction: Maximum number of objective_behaviour instructions to be tokenized.
        :param max_n_antiobjective_instruction: Maximum number of antiobjective instructions to be tokenized.
        :return: Dictionary containing tokenized objective_behaviour and antiobjective instructions.
        """
        if isinstance(tokenizer, str):
            logging.info(f"Loading tokenizer from {tokenizer}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

        max_n_objective_behaviour_instruction = (
            min(
                len(self.objective_behaviour_instructions),
                max_n_objective_behaviour_instruction,
            )
            if max_n_objective_behaviour_instruction
            else len(self.objective_behaviour_instructions)
        )
        max_n_antiobjective_instruction = (
            min(len(self.anti_behaviour_instructions), max_n_antiobjective_instruction)
            if max_n_antiobjective_instruction
            else len(self.anti_behaviour_instructions)
        )

        objective_behaviour_instructions = random.sample(
            self.objective_behaviour_instructions, max_n_objective_behaviour_instruction
        )
        anti_behaviour_instructions = random.sample(
            self.anti_behaviour_instructions, max_n_antiobjective_instruction
        )

        logging.info(
            f"For tokenization, using {max_n_objective_behaviour_instruction / len(self.objective_behaviour_instructions) * 100:.2f}% objective_behaviour instructions."
        )
        logging.info(
            f"For tokenization, using {max_n_antiobjective_instruction / len(self.anti_behaviour_instructions) * 100:.2f}% antiobjective instructions."
        )

        logging.info("Tokenizing objective_behaviour instructions...")
        with tqdm(
            total=max_n_objective_behaviour_instruction,
            desc="Tokenizing objective_behaviour instructions",
        ) as bar:
            objective_behaviour_instr_tokens: List[torch.Tensor] = [
                self._tokenize(
                    tokenizer=tokenizer,
                    instruction=objective_behaviour_instruction,
                    bar=bar,
                )
                for objective_behaviour_instruction in objective_behaviour_instructions
            ]

        logging.info("Tokenizing antiobjective instructions...")
        with tqdm(
            total=max_n_antiobjective_instruction,
            desc="Tokenizing antiobjective instructions",
        ) as bar:
            antiobjective_instr_tokens: List[torch.Tensor] = [
                self._tokenize(
                    tokenizer=tokenizer, instruction=anti_behaviour_instruction, bar=bar
                )
                for anti_behaviour_instruction in anti_behaviour_instructions
            ]
        logging.info("Tokenization complete.")

        return {
            "objective_behaviour_tokens": objective_behaviour_instr_tokens,
            "antiobjective_tokens": antiobjective_instr_tokens,
        }

    def _generate_new_tokens(
        self,
        model: AutoModelForCausalLM,
        tokens: Tensor,
        bar: tqdm | None = None,
        n_generated_tokens: int = 1,
        streamer: TextStreamer | None = None,
    ) -> GenerateDecoderOnlyOutput:
        """
        Generates new tokens given a prompt.
        :param model: A HuggingFace model.
        :param tokens: Tokenized instruction.
        :param bar: Progress bar object.
        :param n_generated_tokens: Number of tokens to generate.
        :param streamer: TextStreamer object, used for showing the text generation.
        :return: Generated tokens.
        """
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

        output = model.generate(**params)
        return output

    def compute_output(
        self,
        model: AutoModelForCausalLM | str,
        objective_behaviour_tokenized_instructions: List[Tensor],
        anti_behaviour_tokenized_instructions: List[Tensor],
    ) -> Dict[str, List[GenerateDecoderOnlyOutput]]:
        """
        Computes the output for the given instructions.
        :param model: A HuggingFace model.
        :param objective_behaviour_tokenized_instructions: Tokenized objective_behaviour instructions.
        :param anti_behaviour_tokenized_instructions: Tokenized antiobjective instructions.
        :return: Dictionary containing the outputs for the given instructions.
        """
        if isinstance(model, str):
            logging.info(f"Loading model from {model}")
            model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
        else:
            model.to(self.device)

        logging.info("Generating tokens on objective_behaviour instructions.")
        with tqdm(
            total=len(objective_behaviour_tokenized_instructions),
            desc="Generating tokens on objective_behaviour instructions",
        ) as bar:
            objective_behaviour_outputs = [
                self._generate_new_tokens(
                    model=model,
                    tokens=objective_behaviour_tokenized_instruction,
                    bar=bar,
                    n_generated_tokens=self.max_toks,
                )
                for objective_behaviour_tokenized_instruction in objective_behaviour_tokenized_instructions
            ]
        logging.info("Completed generating tokens on objective_behaviour instructions.")

        logging.info("Generating tokens on antiobjective instructions.")
        with tqdm(
            total=len(anti_behaviour_tokenized_instructions),
            desc="Generating tokens on antiobjective instructions",
        ) as bar:
            antiobjective_outputs = [
                self._generate_new_tokens(
                    model=model,
                    tokens=anti_behaviour_instruction,
                    bar=bar,
                    n_generated_tokens=self.max_toks,
                )
                for anti_behaviour_instruction in anti_behaviour_tokenized_instructions
            ]
        logging.info("Completed generating tokens on antiobjective instructions.")

        return {
            "obj_beh": objective_behaviour_outputs,
            "anti_obj": antiobjective_outputs,
        }

    def find_approximate_best_objective_behaviour_direction(
        self,
        model: AutoModelForCausalLM | PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | AutoTokenizer,
        scorer: BaseScorer,
        eval_objective_behaviour_instructions: List[str],
        eval_antiobjective_instructions: List[str],
        min_layer: int | None = None,
        max_layer: int | None = None,
    ) -> Tensor:
        """
        Finds the approximate best objective_behaviour direction, given the model, tokenizer, scorer, instructions and the layers.
        :param model: A HuggingFace model.
        :param tokenizer: Tokenizer for a particular model.
        :param scorer: Scorer object.
        :param eval_objective_behaviour_instructions: Instructions for evaluating the objective_behaviour.
        :param eval_antiobjective_instructions: Instructions for evaluating the antiobjective.
        :param min_layer: First layer to be considered for computing the best direction.
        :param max_layer: Last layer to be considered for computing the best direction.
        :return: The approximate best objective_behaviour direction.
        """
        if min_layer is None:
            min_layer = max(int(len(model.model.layers) * 0.2), 1)
        if max_layer is None:
            max_layer = min(
                int(len(model.model.layers) * 0.8), len(model.model.layers) - 2
            )

        logging.info(
            f"Using layers from {min_layer} to {max_layer} for computing best direction."
        )
        score_x_layer = []
        logging.info("Tokenizing evaluation instructions...")
        with tqdm(
            total=len(eval_objective_behaviour_instructions),
            desc="Tokenizing objective_behaviour Eval Instructions set",
        ) as bar:
            obj_beh_toks = [
                self._tokenize(tokenizer=tokenizer, instruction=instr, bar=bar)
                for instr in eval_objective_behaviour_instructions
            ]
        with tqdm(
            total=len(eval_antiobjective_instructions),
            desc="Tokenizing antiobjective Eval Instructions set",
        ) as bar:
            anti_obj_toks = [
                self._tokenize(tokenizer=tokenizer, instruction=instr, bar=bar)
                for instr in eval_antiobjective_instructions
            ]

        logging.info("Computing output for evaluation instructions...")
        d_out = self.compute_output(
            model=model,
            objective_behaviour_tokenized_instructions=obj_beh_toks,
            anti_behaviour_tokenized_instructions=anti_obj_toks,
        )

        self.free_memory([obj_beh_toks, anti_obj_toks])

        logging.info("Finding best objective_behaviour direction...")
        for layer_idx in trange(
            min_layer, max_layer, desc="Finding best objective_behaviour direction"
        ):
            tmp_obj_beh_dir = self.compute_objective_behaviour_direction(
                model=model,
                objective_behaviour_outputs=d_out["obj_beh"],
                antiobjective_outputs=d_out["anti_obj"],
                layer=layer_idx,
            )
            logging.info(
                f"Objective_behaviour direction computed for layer {layer_idx}."
            )
            logging.info("Ablating and adding layers to compute score...")
            conversations_ablated = self.run_forged_model(
                model=model,
                type_of_layer=AblationDecoderLayer,
                objective_behaviour_dir=tmp_obj_beh_dir,
                tokenizer=tokenizer,
                min_layer=min_layer,
                max_layer=max_layer,
                instructions=eval_objective_behaviour_instructions,
                max_new_tokens=100,
                stream=False,
            )

            logging.info("Ablation complete. Adding layers to compute score...")
            conversations_added = self.run_forged_model(
                model=model,
                type_of_layer=AdditionDecoderLayer,
                objective_behaviour_dir=tmp_obj_beh_dir,
                tokenizer=tokenizer,
                min_layer=layer_idx,
                max_layer=layer_idx + 1,
                instructions=eval_antiobjective_instructions,
                max_new_tokens=100,
                stream=False,
            )

            objective_behaviour_score = sum(
                [
                    scorer.score(
                        model_response=conv[-1]["content"],
                        user_query=conv[-2]["content"],
                    )
                    for conv in conversations_ablated
                ]
            )
            antiobjective_score = 1 - sum(
                [
                    scorer.score(
                        model_response=conv[-1]["content"],
                        user_query=conv[-2]["content"],
                    )
                    for conv in conversations_added
                ]
            )

            score_x_layer.append(
                {
                    "layer": layer_idx,
                    "score": (objective_behaviour_score - antiobjective_score) / 2,
                    "dir": tmp_obj_beh_dir,
                }
            )
        score_x_layer = sorted(score_x_layer, key=lambda x: x["score"], reverse=True)
        return score_x_layer[0]["dir"]

    def _replace_layers(
        self,
        new_layer: Type[torch.nn.Module],
        max_layer: int,
        min_layer: int,
        model: AutoModelForCausalLM | PreTrainedModel,
        direction: Tensor,
    ) -> AutoModelForCausalLM | PreTrainedModel:
        """
        Replaces the layers of the model.
        :param new_layer: Type of layer to be replaced.
        :param max_layer: Maximum layer to be replaced.
        :param min_layer: Minimum layer to be replaced.
        :param model: A HuggingFace model.
        :param direction: Direction tensor.
        :return: Model with replaced layers.
        """
        for layer_idx in trange(min_layer, max_layer, desc="Ablating model layers"):
            if isinstance(
                model.model.layers[layer_idx], AblationDecoderLayer
            ) or isinstance(model.model.layers[layer_idx], AdditionDecoderLayer):
                model.model.layers[layer_idx] = new_layer(
                    original_layer=model.model.layers[layer_idx].original_layer,
                    direction=direction,
                )
            else:
                model.model.layers[layer_idx] = new_layer(
                    original_layer=model.model.layers[layer_idx],
                    direction=direction,
                )
        return model

    def compute_objective_behaviour_direction(
        self,
        model: AutoModelForCausalLM | PreTrainedModel,
        objective_behaviour_outputs: List[GenerateDecoderOnlyOutput],
        antiobjective_outputs: List[GenerateDecoderOnlyOutput],
        layer: int | None = None,
    ) -> Tensor:
        """
        Computes the objective_behaviour direction given a layer.
        :param model: A HuggingFace model.
        :param objective_behaviour_outputs: Objective_behaviour outputs.
        :param antiobjective_outputs: Antiobjective outputs.
        :param layer: Layer to be considered for computing the objective_behaviour direction.
        :return: Objective_behaviour direction.
        """
        if layer is None:
            layer = int(len(model.model.layers) * 0.6)
        objective_behaviour_mean = torch.stack(
            [
                output.hidden_states[0][layer][:, -self.max_toks :, :].mean(dim=1)
                for output in objective_behaviour_outputs
            ]
        ).mean(dim=0)
        antiobjective_mean = torch.stack(
            [
                output.hidden_states[0][layer][:, -self.max_toks :, :].mean(dim=1)
                for output in antiobjective_outputs
            ]
        ).mean(dim=0)

        objective_behaviour_dir = objective_behaviour_mean - antiobjective_mean
        objective_behaviour_dir = (
            objective_behaviour_dir / objective_behaviour_dir.norm()
        )

        return objective_behaviour_dir

    def run_forged_model(
        self,
        model: AutoModelForCausalLM | PreTrainedModel,
        objective_behaviour_dir: Tensor,
        tokenizer: PreTrainedTokenizerBase | AutoTokenizer,
        type_of_layer: Type[torch.nn.Module] | None = None,
        min_layer: int | None = None,
        max_layer: int | None = None,
        instructions: List[str] | None = None,
        tokenized_instructions: List[Tensor] | None = None,
        max_new_tokens: int = 100,
        stream: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """
        Runs the forged model.
        :param model: A HuggingFace model.
        :param objective_behaviour_dir: Objective_behaviour direction.
        :param tokenizer: Tokenizer for a particular model.
        :param type_of_layer: Type of layer to be replaced.
        :param min_layer: Minimum layer to be replaced.
        :param max_layer: Maximum layer to be replaced.
        :param instructions: Instructions to be used for the forged model.
        :param tokenized_instructions: Tokenized instructions to be used for the forged model.
        :param max_new_tokens: Maximum number of tokens to be generated.
        :param stream: Whether to show as text the generation.
        :return: List of conversations.
        """

        if min_layer is None:
            min_layer = max(int(len(model.model.layers) * 0.2), 1)
        if max_layer is None:
            max_layer = min(
                int(len(model.model.layers) * 0.8), len(model.model.layers) - 2
            )

        new_model = self._replace_layers(
            new_layer=type_of_layer if type_of_layer else AblationDecoderLayer,
            max_layer=max_layer,
            min_layer=min_layer,
            model=model,
            direction=objective_behaviour_dir,
        )

        if tokenized_instructions:
            logging.info(
                "Using provided tokenized instructions. No need to tokenize again."
            )
            instr_tokens = tokenized_instructions
        elif instructions:
            logging.info("Tokenizing instructions for newly forged model.")
            with tqdm(
                total=len(instructions),
                desc="Tokenizing instructions for newly forged model",
            ) as bar:
                instr_tokens: List[torch.Tensor] = [
                    self._tokenize(
                        tokenizer=tokenizer, instruction=instruction, bar=bar
                    )
                    for instruction in instructions
                ]
        else:
            raise ValueError(
                "Either instructions or tokenized instructions must be provided."
            )

        logging.info("Generating tokens for newly forged model.")
        with tqdm(
            total=len(instructions), desc="Generating tokens for newly forged model"
        ) as bar:
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
                    {
                        "role": "assistant",
                        "content": tokenizer.decode(
                            enc_resp.sequences[0].tolist(), skip_special_tokens=True
                        ),
                    },
                ]
            )

        return conversations

    @staticmethod
    def _modify_tensor(
        tensor: Tensor,
        behaviour_dir: Tensor,
        scale_factor: float = 1.0,
    ) -> Tensor:
        """
        Modifies the tensor applying the behaviour direction.
        :param tensor: Tensor to be modified.
        :param behaviour_dir: Behaviour direction.
        :param scale_factor: Scale factor, must be between -1.0 and 1.0. If negative, induces the behaviour.
        :return: Modified tensor.
        """
        if abs(scale_factor) > 1.0:
            raise ValueError("The scale factor must be between -1.0 and 1.0.")

        tensor_float32 = tensor.to(torch.float32)
        refusal_dir_float32 = behaviour_dir.to(torch.float32)

        if refusal_dir_float32.dim() > 1:
            refusal_dir_float32 = refusal_dir_float32.view(-1)

        tensor_float32 -= scale_factor * torch.matmul(
            input=torch.outer(
                input=refusal_dir_float32,
                vec2=refusal_dir_float32,
            ),
            other=tensor_float32,
        )
        tensor_modified = tensor_float32.to(torch.bfloat16)

        return torch.nn.Parameter(tensor_modified)

    def save_model(
        self,
        model: AutoModelForCausalLM | PreTrainedModel,
        behaviour_dir: Tensor,
        scale_factor: float = 1.0,
        min_layer: int | None = None,
        max_layer: int | None = None,
        output_model_name: str = None,
        tokenizer: PreTrainedTokenizerBase | AutoTokenizer = None,
        to_hub: bool = False,
        model_architecture: str = "gemma",
    ) -> AutoModelForCausalLM | PreTrainedModel:
        """
        Modifies the layers and saves the model (to disk or to the HuggingFace Hub).
        :param model: A HuggingFace model.
        :param behaviour_dir: Behaviour direction.
        :param scale_factor: Scale factor, must be between -1.0 and 1.0. If negative, induces the behaviour.
        :param min_layer: Minimum layer to be modified.
        :param max_layer: Maximum layer to be modified.
        :param output_model_name: Name of the (new) model, useful if pushed to hub or saved somewhere.
        :param tokenizer: Tokenizer for a particular model, useful if pushed to hub or saved somewhere.
        :param to_hub: Whether to push the model to the HuggingFace Hub.
        :param model_architecture: Model architecture, needed to identify what are the layers to be modified. If not specified, will try to find the layers to be modified by trying all possibilities.
        :return: Modified model.
        """
        if abs(scale_factor) > 1.0:
            raise ValueError("The scale factor must be between -1.0 and 1.0.")
        if not model_architecture:
            logging.warning(
                "No model architecture provided. Trying to identify the model architecture based on the layer names."
            )
            model_architecture = identify_model(model.model)
        layer_names = get_layers_names_by_model(model_architecture.lower())
        custom_model = model.model

        if min_layer is None:
            min_layer = max(int(len(model.model.layers) * 0.2), 2)
        if max_layer is None:
            max_layer = min(
                int(len(model.model.layers) * 0.8), len(model.model.layers) - 3
            )

        for layer_idx in range(min_layer, max_layer):
            layer = custom_model.layers[layer_idx]
            for attr_path in layer_names.values():
                parts = attr_path.split(".")
                target = layer
                for part in parts[:-1]:
                    target = getattr(target, part)

                weight_attr = getattr(target, parts[-1])
                modified_weight = self._modify_tensor(
                    tensor=weight_attr,
                    behaviour_dir=behaviour_dir,
                    scale_factor=scale_factor,
                )
                setattr(target, parts[-1], torch.nn.Parameter(modified_weight))
        if output_model_name:
            model.save_pretrained(
                output_model_name,
                push_to_hub=to_hub,
            )
            if tokenizer:
                tokenizer.save_pretrained(
                    output_model_name,
                    push_to_hub=to_hub,
                )
        else:
            logging.warning(
                "No output model name provided. Model not saved to disk nor pushed to hub."
            )

        return model

    def free_memory(self, list_of_variables: List[Any]):
        """
        Frees the memory.
        :param list_of_variables: List of variables to be deleted.
        :return: None
        """
        logging.warning(f"Freeing memory for {len(list_of_variables)} variables.")
        del list_of_variables
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
