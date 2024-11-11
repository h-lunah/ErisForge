import random

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.eris_forge import (
    Forge,
)
from src.scorers.refusal_scorer.expression_refusal_scorer import (
    ExpressionRefusalScorer,
)

if __name__ == "__main__":
    random.seed(42)
    MODEL = 'google/gemma-1.1-2b-it'
    with open("../src/assets/harmful_instructions.txt", "r") as f:
        obj_beh = f.readlines()

    with open("../src/assets/harmless_instructions.txt", "r") as f:
        anti_obj = f.readlines()

    max_inst = 100

    forge = Forge()
    forge.load_instructions(objective_behaviour_instructions=obj_beh, anti_behaviour_instructions=anti_obj)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(forge.device)

    d_toks = forge.tokenize_instructions(
        tokenizer=tokenizer,
        max_n_antiobjective_instruction=max_inst,
        max_n_objective_behaviour_instruction=max_inst,
    )

    d_instr = forge.compute_output(
        model=model,
        objective_behaviour_tokenized_instructions=d_toks['objective_behaviour_tokens'],
        anti_behaviour_tokenized_instructions=d_toks['antiobjective_tokens'],
    )

    scorer = ExpressionRefusalScorer()

    forge.free_memory([d_toks, d_instr])

    # Currently, this piece of code probably causes a memory leak.

    # refusal_dir = forge.find_approximate_best_objective_behaviour_direction(
    #     model=model,
    #     tokenizer=tokenizer,
    #     scorer=scorer,
    #     eval_objective_behaviour_instructions=obj_beh[:max_inst],
    #     eval_antiobjective_instructions=anti_obj[:max_inst],
    #     min_layer=10,
    #     max_layer=13,
    # )

    refusal_dir = forge.compute_objective_behaviour_direction(
        model=model,
        objective_behaviour_outputs=d_instr['obj_beh'],
        antiobjective_outputs=d_instr['anti_obj'],
        layer=int(len(model.model.layers) * 0.65),
    )

    # conversations = forge.run_forged_model(
    #     model=model,
    #     objective_behaviour_dir=refusal_dir,
    #     tokenizer=tokenizer,
    #     instructions=random.sample(population=obj_beh, k=20),
    #     max_new_tokens=100,
    #     stream=False,
    # )
    #
    # for conversation in conversations:
    #     print('=' * 20)
    #     for round in conversation:
    #         print(f'{round["role"]}: {round["content"]}')
    #
    # forge.free_memory([conversations, d_toks, d_instr])

    forge.save_model(
        model=model,
        tokenizer=tokenizer,
        behaviour_dir=refusal_dir,
        output_model_name='corrupted_gemma_test',
        to_hub=False,
        model_architecture='gemma',
    )
