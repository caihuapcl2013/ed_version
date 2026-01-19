import torch
from prune_ffn_alpamayo import prune_alpamayo_ffn

# ⚠️ 按你自己的工程路径修改
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


def main():
    # -------- 1. 加载模型 --------
    model = AlpamayoR1.from_pretrained(
        "checkpoints/alpamayo_r1_10b"
    ).cuda().eval()

    # -------- 2. FFN 剪枝 --------
    pruned_model = prune_alpamayo_ffn(
        model,
        keep_ratio=0.7,   # ⭐ 推荐 0.7 / 0.75 / 0.8
        verbose=True
    )

    # -------- 3. 保存权重 --------
    torch.save(
        pruned_model.state_dict(),
        "alpamayo_r1_ffn70_pruned.pth"
    )

    print("\n🎯 Pruned model saved.")


if __name__ == "__main__":
    main()
