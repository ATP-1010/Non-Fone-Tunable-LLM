import os
import random
import json
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

# ===== 2.1 Freeze 30% neurons per Linear (structured by rows) & save mask =====
def build_neuron_mask(model, freeze_ratio=0.30, seed=42):
    """
    对每个 nn.Linear，随机选择 freeze_ratio 比例的输出神经元（weight 的行）作为私有 mask。
    返回: { "<module_path>.weight": [row_idx, ...], ... }
    """
    g = torch.Generator().manual_seed(seed)
    mask = {}
    for mod_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            out_features = module.weight.shape[0]
            k = max(1, int(out_features * freeze_ratio))
            idx = torch.randperm(out_features, generator=g)[:k]
            mask[f"{mod_name}.weight"] = idx.tolist()
            # 若存在 bias，同步记录
            if module.bias is not None:
                mask[f"{mod_name}.bias"] = idx.tolist()
    return mask

def apply_neuron_freeze_with_hooks(model, mask):
    """
    用 backward hook 把被 mask 的 Linear 输出行（以及 bias 对应索引）的梯度清零，等效冻结。
    不改 requires_grad，不影响你其余逻辑。
    """
    hooks = []

    for mod_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w_key = f"{mod_name}.weight"
            b_key = f"{mod_name}.bias"

            if w_key in mask:
                idx = torch.tensor(mask[w_key], dtype=torch.long)

                def make_w_hook(row_idx):
                    def hook(m, gin, gout):
                        if m.weight.grad is not None:
                            m.weight.grad[row_idx, :] = 0
                    return hook
                hooks.append(module.register_full_backward_hook(make_w_hook(idx)))

            if module.bias is not None and b_key in mask:
                b_idx = torch.tensor(mask[b_key], dtype=torch.long)

                def make_b_hook(row_idx):
                    def hook(m, gin, gout):
                        if m.bias is not None and m.bias.grad is not None:
                            m.bias.grad[row_idx] = 0
                    return hook
                hooks.append(module.register_full_backward_hook(make_b_hook(b_idx)))

    return hooks

# 构建并保存 mask（可复现实验）
neuron_mask = build_neuron_mask(model, freeze_ratio=0.30, seed=42)
os.makedirs("./checkpoints", exist_ok=True)
with open("./checkpoints/private_neuron_mask.json", "w") as f:
    json.dump(neuron_mask, f, indent=2)

# 应用冻结（预训练阶段恒定生效）
_ = apply_neuron_freeze_with_hooks(model, neuron_mask)

# ===== 3. Load Dataset =====
print(">>> Loading The Pile dataset...")
dataset = load_dataset("DKYoon/SlimPajama-6B", split="train")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=2048)

tokenized_dataset = dataset.map(tokenize_fn, num_proc=32)

# ===== 4. Data Collator =====
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===== 5. Training Arguments =====
# 提供 DeepSpeed 开关：默认启用；export USE_DEEPSPEED=0 可关闭
use_ds = os.environ.get("USE_DEEPSPEED", "1") != "0"
ds_cfg = "deepspeed_config.json" if use_ds else None

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
    deepspeed=ds_cfg,                    # ← 开关在这里
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
