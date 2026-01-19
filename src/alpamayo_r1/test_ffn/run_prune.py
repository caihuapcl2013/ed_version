import torch
import sys
from pathlib import Path

# ------------------------
# 确保能 import 本地 repo
# ------------------------
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from prune_ffn_alpamayo import prune_alpamayo_ffn

# ------------------------
# 配置
# ------------------------
HF_MODEL_ID = "nvidia/Alpamayo-R1-10B"  # HuggingFace 权重
KEEP_RATIO = 0.7                        # FFN 剪枝比例
OUTPUT_PATH = "alpamayo_r1_ffn70_pruned.pth"  # 剪枝后权重保存路径
DEVICE = "cuda"

# ------------------------
# 1️⃣ 下载 HF 权重并初始化模型
# ------------------------
print(f"⏳ Loading Alpamayo-R1 model from HF: {HF_MODEL_ID} ...")
model = AlpamayoR1.from_pretrained(HF_MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
model.eval()
print(f"✅ Model loaded on {DEVICE}")

# ------------------------
# 2️⃣ FFN 剪枝
# ------------------------
print(f"⏳ Pruning FFN with keep_ratio={KEEP_RATIO} ...")
model = prune_alpamayo_ffn(model, keep_ratio=KEEP_RATIO, verbose=True)
print("✅ FFN pruning completed")

# ------------------------
# 3️⃣ 保存剪枝后的权重
# ------------------------
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"🎯 Pruned model saved to {OUTPUT_PATH}")

# ------------------------
# 4️⃣ 测试加载剪枝权重（可选）
# ------------------------
# 加载验证
# model2 = AlpamayoR1.from_pretrained(HF_MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
# state = torch.load(OUTPUT_PATH, map_location=DEVICE)
# model2.load_state_dict(state, strict=True)
# print("✅ Pruned weights load test successful")
