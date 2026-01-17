import torch
import numpy as np
import cv2
import os

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# ============================================================
# 0. 全局配置
# ============================================================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 强制同步，方便定位报错

DEVICE = "cuda"
DTYPE = torch.bfloat16

NUM_HISTORY_FRAMES = 4
FRAME_H = FRAME_W = 224  # ✅ 官方分辨率
VIDEO_PATH = "../../test.MOV"

print("==== ENV INFO ====")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("==================\n")

# ============================================================
# 1. 读取视频帧
# ============================================================
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []

while len(frames) < NUM_HISTORY_FRAMES:
    ret, f = cap.read()
    if not ret:
        break
    # resize 到官方分辨率
    f = cv2.resize(f, (FRAME_W, FRAME_H))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    frames.append(f)

cap.release()
assert len(frames) == NUM_HISTORY_FRAMES, "视频帧数不足"

image_frames = torch.from_numpy(np.array(frames)).permute(0,3,1,2).float()

print("==== IMAGE FRAMES ====")
print("image_frames.shape:", image_frames.shape)
print("image_frames.dtype:", image_frames.dtype)
print("======================\n")

# ============================================================
# 2. Ego history
# ============================================================
ego_history_xyz = torch.zeros(1, 1, NUM_HISTORY_FRAMES, 3)
ego_history_rot = (
    torch.eye(3)
    .unsqueeze(0)
    .unsqueeze(0)
    .unsqueeze(0)
    .repeat(1, 1, NUM_HISTORY_FRAMES, 1, 1)
)

print("==== EGO HISTORY ====")
print("ego_history_xyz:", ego_history_xyz.shape)
print("ego_history_rot:", ego_history_rot.shape)
print("=====================\n")

# ============================================================
# 3. 构建 message
# ============================================================
messages = helper.create_message(image_frames)

print("==== MESSAGE STRUCTURE ====")
print(messages)
print("===========================\n")

# ============================================================
# 4. 加载模型
# ============================================================
print("Loading Alpamayo-R1-10B ...")
model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=DTYPE
).to(DEVICE)
model.eval()

tokenizer = model.tokenizer
processor = helper.get_processor(tokenizer)

print("==== TOKENIZER INFO ====")
print("pad_token:", tokenizer.pad_token, "pad_token_id:", tokenizer.pad_token_id)
print("bos_token:", tokenizer.bos_token, "bos_token_id:", tokenizer.bos_token_id)
print("eos_token:", tokenizer.eos_token, "eos_token_id:", tokenizer.eos_token_id)
print("=========================\n")

# ============================================================
# 5. 强制 pad_token 合法
# ============================================================
if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
    print("[FIX] pad_token_id 非法，强制修复")
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.bos_token is not None:
        tokenizer.pad_token = tokenizer.bos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

vlm = model.vlm
vlm.config.pad_token_id = tokenizer.pad_token_id
vlm.generation_config.pad_token_id = tokenizer.pad_token_id
vlm.generation_config.bos_token_id = None
vlm.generation_config.eos_token_id = None

print("==== AFTER PAD FIX ====")
print("pad_token_id:", tokenizer.pad_token_id)
print("vlm.config.pad_token_id:", vlm.config.pad_token_id)
print("vlm.gen.pad_token_id:", vlm.generation_config.pad_token_id)
print("=======================\n")

# ============================================================
# 6. Tokenize
# ============================================================
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

# ✅ 官方分辨率 token 数量应该接近训练期
# 删除 attention_mask 避免 HF generate 的 masked_scatter
if "attention_mask" in inputs:
    inputs.pop("attention_mask")

print("==== TOKENIZED INPUTS ====")
for k, v in inputs.items():
    print(k, type(v), v.shape if hasattr(v,"shape") else "")
print("===========================\n")

# ============================================================
# 7. 构建模型输入
# ============================================================
model_inputs = {
    "tokenized_data": helper.to_device(inputs, DEVICE),
    "ego_history_xyz": ego_history_xyz.to(DEVICE),
    "ego_history_rot": ego_history_rot.to(DEVICE),
}

print("==== MODEL INPUTS READY ====\n")

# ============================================================
# 8. 推理
# ============================================================
torch.cuda.manual_seed_all(42)

print("==== START INFERENCE ====")
with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        num_traj_samples=1,
        max_generation_length=256,
        temperature=0.6,
        top_p=0.98,
        return_extra=True,
    )

print("==== INFERENCE DONE ====\n")

# ============================================================
# 9. 输出预测结果
# ============================================================
print("pred_xyz.shape:", pred_xyz.shape)
print("pred_rot.shape:", pred_rot.shape)

if "cot" in extra:
    print("CoT:\n", extra["cot"][0])
else:
    print("No CoT returned")

traj_xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
print("Predicted XY trajectory:\n", traj_xy)
