import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    get_polynomial_decay_schedule_with_warmup
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "google/gemma-7b"
access_token = "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"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=access_token
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="auto",
    token=access_token
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
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


df = pd.read_csv("datasets/train.csv")
df["original_text"] = df["original_text"].apply(tokenize_and_truncate)
df["rewritten_text"] = df["rewritten_text"].apply(tokenize_and_truncate)
df["token_count"] = df.apply(lambda row: len(tokenizer(row["rewrite_prompt"] + row["original_text"] + row["rewritten_text"])["input_ids"]), axis=1)
df = df.sample(frac=1, random_state=42)

train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_dataset = df[:train_size]
train_dataset = Dataset.from_pandas(train_dataset)
eval_dataset = df[train_size:]
eval_dataset = Dataset.from_pandas(eval_dataset)


def formatting_prompts_func(example):
    system_prompt = f"Assuming you are an expert in self-awareness and skilled at recovering prompts. I will give you an Original Text and the Rewritten Text after I prompt you. Please tell me what prompt I most likely gave you. The length of the prompt does not exceed thirty words.\n"
    input_prompt = f"Original Text: {example['original_text'][0]}\nRewritten Text: {example['rewritten_text'][0]}\n"
    output_prompt = f"Used Prompt: {example['rewrite_prompt'][0]}"
    full_text = system_prompt + input_prompt + output_prompt
    return [full_text]


args = TrainingArguments(
    output_dir=f"outputs",
    gradient_accumulation_steps=8,
    fp16=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_steps=100,
    num_train_epochs=3,
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,
    overwrite_output_dir=True,
    weight_decay=0.01,
    save_only_model=True,
    neftune_noise_alpha=1.0,
    learning_rate=1e-4,
    warmup_steps=50,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    #     max_steps=10
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
