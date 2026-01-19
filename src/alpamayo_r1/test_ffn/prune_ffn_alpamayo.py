import torch
import torch.nn as nn
from copy import deepcopy
from typing import Tuple


@torch.no_grad()
def prune_ffn_linear_pair(
    fc1: nn.Linear,
    fc2: nn.Linear,
    keep_ratio: float = 0.7
) -> Tuple[nn.Linear, nn.Linear]:
    """
    剪 FFN 中间维度 (d_ff)

    fc1: Linear(d_model -> d_ff)
    fc2: Linear(d_ff -> d_model)
    """
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    assert fc1.out_features == fc2.in_features

    d_ff = fc1.out_features
    keep_dim = max(1, int(d_ff * keep_ratio))

    # ---------- 1. 重要性评估 ----------
    # [d_ff, d_model] -> [d_ff]
    importance = fc1.weight.abs().mean(dim=1)

    # ---------- 2. Top-K 选择 ----------
    keep_idx = torch.topk(importance, keep_dim, largest=True).indices
    keep_idx, _ = torch.sort(keep_idx)

    # ---------- 3. 构建新 Linear ----------
    new_fc1 = nn.Linear(
        fc1.in_features,
        keep_dim,
        bias=fc1.bias is not None
    )
    new_fc2 = nn.Linear(
        keep_dim,
        fc2.out_features,
        bias=fc2.bias is not None
    )

    # ---------- 4. 权重拷贝 ----------
    new_fc1.weight.copy_(fc1.weight[keep_idx])
    if fc1.bias is not None:
        new_fc1.bias.copy_(fc1.bias[keep_idx])

    new_fc2.weight.copy_(fc2.weight[:, keep_idx])
    if fc2.bias is not None:
        new_fc2.bias.copy_(fc2.bias)

    return new_fc1, new_fc2


def is_ffn_block(module: nn.Module) -> bool:
    """
    判断是否为 FFN / MLP block
    兼容 Alpamayo / LLaMA / ViT 风格
    """
    return (
        hasattr(module, "fc1")
        and hasattr(module, "fc2")
        and isinstance(module.fc1, nn.Linear)
        and isinstance(module.fc2, nn.Linear)
    )


def prune_alpamayo_ffn(
    model: nn.Module,
    keep_ratio: float = 0.7,
    verbose: bool = True
) -> nn.Module:
    """
    遍历模型，剪所有 FFN block
    """
    assert 0.0 < keep_ratio <= 1.0

    pruned_model = deepcopy(model)
    total_blocks = 0

    for name, module in pruned_model.named_modules():
        if is_ffn_block(module):
            old_dim = module.fc1.out_features
            module.fc1, module.fc2 = prune_ffn_linear_pair(
                module.fc1,
                module.fc2,
                keep_ratio=keep_ratio
            )
            total_blocks += 1

            if verbose:
                print(
                    f"[FFN Pruned] {name}: "
                    f"d_ff {old_dim} -> {module.fc1.out_features}"
                )

    if verbose:
        print(f"\n✅ Total FFN blocks pruned: {total_blocks}")

    return pruned_model
