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
from trl import SFTTrainer, DataCollatorForLanguageModeling
from datasets import Dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    # load_in_8bit=True
)

model_id = "google/gemma-7b-it"
access_token = "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"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=access_token
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    token=access_token
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
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
print(
    f"Trainable: {trainable} | total: {total} | Percentage: {trainable / total * 100: .4f}%")


def tokenize_and_truncate(text, max_length=150):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    truncated_text = tokenizer.decode(tokens)
    return truncated_text


df = pd.read_csv("datasets/train.csv")
df["original_text"] = df["original_text"].apply(tokenize_and_truncate)
df["rewritten_text"] = df["rewritten_text"].apply(tokenize_and_truncate)
df["token_count"] = df.apply(lambda row: len(tokenizer(
    row["rewrite_prompt"] + row["original_text"] + row["rewritten_text"])["input_ids"]), axis=1)
df = df.sample(frac=1, random_state=42)

train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_dataset = df[:train_size]
train_dataset = Dataset.from_pandas(train_dataset)
eval_dataset = df[train_size:]
eval_dataset = Dataset.from_pandas(eval_dataset)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example)):
        system_prompt = f"Assuming you are an expert in self-awareness and skilled at recovering prompts. I will give you an Original Text and the Rewritten Text after I prompt you. Please tell me what prompt I most likely gave you. The length of the prompt does not exceed twenty words.\n"
        input_prompt = f"### Original Text: {example['original_text']}\n### Rewritten Text: {example['rewritten_text']}\n"
        output_prompt = f"### Prompt: {example['rewrite_prompt']}"
        output_texts.append(system_prompt + input_prompt + output_prompt)
    return output_texts


args = TrainingArguments(
    output_dir=f"outputs",
    fp16=True,
    gradient_accumulation_steps=8,
    logging_steps=100,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    report_to="none",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,
    overwrite_output_dir=True,
    load_best_model_at_end=True,
    weight_decay=0.01,
    save_only_model=True,
    neftune_noise_alpha=1.0
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
gpu_count = torch.cuda.device_count()
print(f"Number of GPUs: {gpu_count}")
total_steps = args.num_train_epochs * \
    int(len(train_dataset) * 1.0 / gpu_count /
        args.per_device_train_batch_size / args.gradient_accumulation_steps)
scheduler = get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps) * 0.05,
    num_training_steps=total_steps,
    power=1.5,
    lr_end=3e-6
)

torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    peft_config=lora_config,
    dataset_num_proc=2,
    dataset_batch_size=100,
    max_seq_length=1024,
    optimizers=(optimizer, scheduler),
    args=args
)

trainer.train()
new_model = "finetuned-gemma-7b"
trainer.model.save_pretrained(new_model)
