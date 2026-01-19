
import sys
import torch

# 确保能 import alpamayo_r1
sys.path.append("/workspace/ed_version/src")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from prune_ffn_alpamayo import prune_alpamayo_ffn


def main():
    # 1️⃣ 构建模型（照 test_inference.py）
    model = AlpamayoR1().cuda().eval()  # 或者加上 test_inference 里需要的参数

    # 2️⃣ 加载 checkpoint
    ckpt_path = "alpamayo_r1_10b.pth"   # ← 改成你的实际路径
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    print("✅ Original checkpoint loaded")

    # 3️⃣ FFN 剪枝
    model = prune_alpamayo_ffn(
        model,
        keep_ratio=0.7,
        verbose=True
    )

    # 4️⃣ 保存剪枝结果
    torch.save(
        model.state_dict(),
        "alpamayo_r1_ffn70_pruned.pth"
    )

    print("🎯 FFN pruned model saved")


if __name__ == "__main__":
    main()
