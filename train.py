import os
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
# 可替换为 SentencePiece 自训练 tokenizer
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# ===== 2. Model Config =====
# TinyLLaMA 1.1B 配置
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

# ===== 3. Load Dataset =====
# 使用 streaming 模式读取 The Pile（不会爆内存）
print(">>> Loading The Pile dataset...")
dataset = load_dataset("EleutherAI/pile", split="train")

# Tokenization
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=2048)

tokenized_dataset = dataset.map(tokenize_fn)

# ===== 4. Data Collator =====
# causal LM，不使用 MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===== 5. Training Arguments =====
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,            # 每卡 batch size
    gradient_accumulation_steps=8,            # 8×8×8=512 global batch
    num_train_epochs=1,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=3,
    learning_rate=2e-4,
    warmup_steps=2000,
    weight_decay=0.01,
    fp16=True,
    deepspeed="deepspeed_config.json",        # ✅ DeepSpeed ZeRO3
    eval_strategy="no",
    dataloader_drop_last=True,
    report_to="tensorboard",                  # 可改成 wandb
    logging_dir="./logs"
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
