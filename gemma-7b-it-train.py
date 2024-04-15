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
from trl import SFTTrainer
from datasets import Dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "/kaggle/input/gemma/transformers/7b-it/1"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    add_eos_token=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.10,
    bias="none",
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(model, lora_config)

trainable, total = lora_model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable / total * 100: .4f}%")


def tokenize_and_truncate(text, max_length=200):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    truncated_text = tokenizer.decode(tokens)
    return truncated_text


df = pd.read_csv("datasets/1594prompts.csv")
df["original_text"] = df["original_text"].apply(tokenize_and_truncate)
df["rewritten_text"] = df["rewritten_text"].apply(tokenize_and_truncate)
df["token_count"] = df.apply(lambda row: len(tokenizer(row["rewrite_prompt"] + row["original_text"] + row["rewritten_text"])["input_ids"]), axis=1)
df = df.sample(frac=1, random_state=42)

train_size = int(0.80 * len(df))
test_size = len(df) - train_size
train_dataset = df[:train_size]
train_dataset = Dataset.from_pandas(train_dataset)
eval_dataset = df[train_size:]
eval_dataset = Dataset.from_pandas(eval_dataset)


def formatting_prompts_func(example):
    system_prompt = "Assuming you are an expert in self-awareness and skilled at recovering prompts. I will give you an Original Text and the Rewritten Text after I prompt you. You should tell me what prompt I most likely gave you. The prompt should contain less than 30 words, only return the prompt.\n"
    input_prompt = f"<start_of_turn>user {system_prompt}\nOriginal Text: {example['original_text'][0]}\nRewritten Text: {example['rewritten_text'][0]}\nPrompt:<end_of_turn>\n"
    output_prompt = f"<start_of_turn>model {example['rewrite_prompt'][0]}<end_of_turn>"
    full_text = system_prompt + input_prompt + output_prompt
    return [full_text]


args = TrainingArguments(
    output_dir=f"outputs",
    gradient_accumulation_steps=8,
    fp16=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_steps=25,
    num_train_epochs=3,
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=10,
    overwrite_output_dir=True,
    weight_decay=0.01,
    save_only_model=True,
    neftune_noise_alpha=1.0,
    learning_rate=2e-4,
    warmup_steps=50,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine"
)

torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_batch_size=1,
    formatting_func=formatting_prompts_func,
    max_seq_length=512,
    peft_config=lora_config,
    args=args
)

trainer.train()
new_model = "finetuned-gemma-7b"
trainer.model.save_pretrained(new_model)
