import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from pathlib import Path

# ------------------------
# 配置
# ------------------------
HF_MODEL_ID = "nvidia/Alpamayo-R1-10B"
KEEP_RATIO = 0.7
DEVICE = "cuda"
OUTPUT_PATH = "alpamayo_r1_ffn70_pruned.pth"

# ------------------------
# 1️⃣ 加载模型
# ------------------------
print(f"⏳ Loading Alpamayo-R1 model from HF: {HF_MODEL_ID} ...")
model = AlpamayoR1.from_pretrained(HF_MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
model.eval()
print(f"✅ Model loaded on {DEVICE}")

# ------------------------
# 2️⃣ 遍历 FFN 层并剪枝
# ------------------------
total_pruned = 0

def prune_linear_layer(layer: torch.nn.Linear, keep_ratio: float):
    in_features, out_features = layer.weight.shape
    new_out = max(1, int(out_features * keep_ratio))
    if new_out == out_features:
        return layer, 0  # 没有剪枝
    # 随机选择要保留的输出通道
    idx = torch.randperm(out_features)[:new_out]
    new_weight = layer.weight.data[idx, :].clone()
    if layer.bias is not None:
        new_bias = layer.bias.data[idx].clone()
    else:
        new_bias = None
    new_layer = torch.nn.Linear(in_features, new_out, bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.data = new_weight
    if new_bias is not None:
        new_layer.bias.data = new_bias
    return new_layer, out_features - new_out

# 遍历模型
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        pruned_layer, pruned_count = prune_linear_layer(module, KEEP_RATIO)
        if pruned_count > 0:
            # 替换原模型层
            parent = model
            path = name.split(".")
            for p in path[:-1]:
                parent = getattr(parent, p)
            setattr(parent, path[-1], pruned_layer)
            total_pruned += 1
            print(f"Pruned {name}: removed {pruned_count} out of {module.out_features} channels")

print(f"✅ Total FFN layers pruned: {total_pruned}")

# ------------------------
# 3️⃣ 保存剪枝权重
# ------------------------
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"🎯 Pruned model saved to {OUTPUT_PATH}")
