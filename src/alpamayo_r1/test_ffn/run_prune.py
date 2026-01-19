import torch

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.configs.default import get_config
from prune_ffn_alpamayo import prune_alpamayo_ffn


def main():
    # 1️⃣ 构建 config（Alpamayo 原生）
    cfg = get_config()

    # 2️⃣ 构建模型结构
    model = AlpamayoR1(cfg).cuda().eval()

    # 3️⃣ 加载 checkpoint（不是 HF）
    ckpt_path = "alpamayo_r1_10b.pth"   # ← 你真实存在的文件
    state = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(state, strict=True)

    print("✅ Original checkpoint loaded")

    # 4️⃣ FFN 剪枝（结构发生变化）
    model = prune_alpamayo_ffn(
        model,
        keep_ratio=0.7,
        verbose=True
    )

    # 5️⃣ 保存剪枝后的权重
    torch.save(
        model.state_dict(),
        "alpamayo_r1_ffn70_pruned.pth"
    )

    print("🎯 FFN pruned model saved")


if __name__ == "__main__":
    main()
