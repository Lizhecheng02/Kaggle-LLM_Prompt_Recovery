import numpy as np
import pandas as pd
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    get_polynomial_decay_schedule_with_warmup
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm


def get_completion(original_text, rewritten_text, model, tokenizer):
    prompt_template = """
<start_of_turn>user Assuming you are an expert in self-awareness and skilled at recovering prompts. Tell me the prompt I gave to you, only return the content of prompt.
Original Text: {original_text}
Rewritten Text: {rewritten_text}
Prompt:<end_of_turn>
<start_of_turn>model
"""
    prompt = prompt_template.format(
        original_text=original_text,
        rewritten_text=rewritten_text
    )
    encodeds = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True
    )

    model_inputs = encodeds.to("cuda")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=20,
        do_sample=True,
        # pad_token_id=tokenizer.eos_token_id,
        eos_token_id=235265
    )
    prompt_length = len(tokenizer.encode(prompt))
    decoded = tokenizer.decode(
        generated_ids[0][prompt_length:],
        skip_special_tokens=True
    )

    return decoded


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "/kaggle/input/gemma/transformers/7b-it/1"
tokenizer = AutoTokenizer.from_pretrained(
    model_id
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="auto"
)

adapter_file = "/kaggle/input/gemma-7b-it-fine-tuned/outputs/checkpoint-300"
merged_model = PeftModel.from_pretrained(base_model, adapter_file)
merged_model = merged_model.merge_and_unload()
print(merged_model)

test = pd.read_csv("datasets/test.csv")

rewrite_prompts = []

for idx, row in tqdm(test.iterrows(), total=len(test)):
    output = get_completion(
        row["original_text"],
        row["rewritten_text"],
        merged_model,
        tokenizer
    )
    print(output)
    rewrite_prompts.append(output)

test["rewrite_prompt"] = rewrite_prompts
test = test[["id", "rewrite_prompt"]]

test.to_csv("submission.csv", index=False)
submission = pd.read_csv("submission.csv")
submission.head()
