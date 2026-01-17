import torch
import numpy as np
import cv2

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# ============================================================
# 1. 配置（⚠️ 必须与模型训练一致）
# ============================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

NUM_HISTORY_FRAMES = 4   # ⭐ Alpamayo-R1 固定是 4
FRAME_HEIGHT = 128
FRAME_WIDTH = 128

VIDEO_PATH = "../../test.MOV"   # 或 0 使用摄像头


# ============================================================
# 2. 读取视频帧
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

frames = []
while len(frames) < NUM_HISTORY_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

cap.release()

assert len(frames) == NUM_HISTORY_FRAMES, "视频帧数不足"

print(f"[OK] Captured {len(frames)} frames")


# ============================================================
# 3. 构建 image tensor
#    [T, C, H, W]  (⚠️ Alpamayo 只接受这个)
# ============================================================

image_frames = (
    torch.from_numpy(np.array(frames))
    .permute(0, 3, 1, 2)     # [T, C, H, W]
    .float()
)

# ============================================================
# 4. Ego history（必须和 T 对齐）
# ============================================================

batch_size = 1

ego_history_xyz = torch.zeros(
    batch_size, 1, NUM_HISTORY_FRAMES, 3
)

ego_history_rot = (
    torch.eye(3)
    .unsqueeze(0)
    .unsqueeze(0)
    .unsqueeze(0)
    .repeat(batch_size, 1, NUM_HISTORY_FRAMES, 1, 1)
)

# ============================================================
# 5. 构建 VLM message（⚠️ 必须用 helper）
# ============================================================

messages = helper.create_message(image_frames)

# ============================================================
# 6. 加载模型 & processor
# ============================================================

print("[INFO] Loading Alpamayo-R1-10B ...")

model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=DTYPE,
).to(DEVICE)

processor = helper.get_processor(model.tokenizer)

model.eval()

# ============================================================
# 7. Chat template（⭐ 关键：add_generation_prompt=True）
# ============================================================

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,     # ⭐ 必须
    return_dict=True,
    return_tensors="pt",
)

# ============================================================
# 8. 组装最终输入
# ============================================================

model_inputs = {
    "tokenized_data": helper.to_device(inputs, DEVICE),
    "ego_history_xyz": ego_history_xyz.to(DEVICE),
    "ego_history_rot": ego_history_rot.to(DEVICE),
}

# ============================================================
# 9. 推理
# ============================================================

torch.cuda.manual_seed_all(42)

with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        num_traj_samples=1,
        max_generation_length=256,
        temperature=0.6,
        top_p=0.98,
        return_extra=True,
    )

# ============================================================
# 10. 输出结果
# ============================================================

print("\n================== RESULT ==================")
print("CoT:\n", extra["cot"][0])

print("pred_xyz shape:", pred_xyz.shape)
# [batch, traj_set, traj_sample, time, 3]

traj_xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
print("\nPredicted XY trajectory:\n", traj_xy)
