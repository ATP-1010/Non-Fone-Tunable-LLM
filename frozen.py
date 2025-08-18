import os
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# ===== 1. Tokenizer =====
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# ===== 2. Model Config =====
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=1024,
    intermediate_size=2730,
    num_attention_heads=8,
    num_hidden_layers=16,
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    use_cache=False
)
model = LlamaForCausalLM(config)

# ===== 2.1 Freeze 20% parameters randomly =====
random.seed(42)  # 固定随机数种子，保证可复现
all_params = [name for name, p in model.named_parameters()]
num_freeze = int(0.5 * len(all_params))
freeze_params = random.sample(all_params, num_freeze)

for name, param in model.named_parameters():
    if name in freeze_params:
        param.requires_grad = False
        print(f"[Freeze] {name}")

# ===== 3. Load Dataset =====
print(">>> Loading The Pile dataset...")
dataset = load_dataset("DKYoon/SlimPajama-6B", split="train")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=2048)

tokenized_dataset = dataset.map(tokenize_fn, num_proc=32)

# ===== 4. Data Collator =====
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===== 5. Training Arguments =====
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=3,
    learning_rate=2e-4,
    warmup_steps=2000,
    weight_decay=0.01,
    fp16=True,
    deepspeed="deepspeed_config.json",
    eval_strategy="no",
    dataloader_drop_last=True,
    report_to="wandb",
    logging_dir="./logs",
    dataloader_num_workers=16,
    run_name="llama-pt"
)

# ===== 6. Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ===== 7. Start Training =====
if __name__ == "__main__":
    print(">>> Starting training...")
    trainer.train()
