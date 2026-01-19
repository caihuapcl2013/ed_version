import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from pathlib import Path

HF_MODEL_ID = "nvidia/Alpamayo-R1-10B"
DEVICE = "cuda"
OUTPUT_FILE = "alpamayo_r1_layers.txt"

# ------------------------
# 加载模型
# ------------------------
print(f"⏳ Loading Alpamayo-R1 model from HF: {HF_MODEL_ID} ...")
model = AlpamayoR1.from_pretrained(HF_MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
model.eval()
print(f"✅ Model loaded on {DEVICE}")

# ------------------------
# 遍历层级
# ------------------------
layer_infos = []
total_params = 0
ffn_layers = 0

for name, module in model.named_modules():
    if len(list(module.children())) > 0:
        continue  # 忽略非叶子节点
    params = sum(p.numel() for p in module.parameters())
    total_params += params
    layer_type = type(module).__name__
    shape = tuple(module.weight.shape) if hasattr(module, "weight") else "-"
    layer_infos.append(f"{name:50s} | {layer_type:20s} | {shape}")
    if isinstance(module, torch.nn.Linear):
        ffn_layers += 1

# ------------------------
# 保存到文件
# ------------------------
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(layer_infos))
    f.write(f"\n\nTotal FFN layers: {ffn_layers}\n")
    f.write(f"Total parameters: {total_params:,}\n")

print(f"🎯 Layer info saved to {OUTPUT_FILE}")
print(f"Total FFN layers: {ffn_layers}, Total params: {total_params:,}")
