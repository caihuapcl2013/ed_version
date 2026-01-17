import torch
import cv2
import numpy as np
import os

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# ============================================================
# 0. 全局配置
# ============================================================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 强制同步，方便调试
DEVICE = "cuda"
DTYPE = torch.bfloat16

NUM_HISTORY_FRAMES = 4
FRAME_H = FRAME_W = 224  # 官方分辨率
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
    f = cv2.resize(f, (FRAME_W, FRAME_H))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    frames.append(f)

cap.release()
assert len(frames) == NUM_HISTORY_FRAMES, "视频帧不足"

image_frames = np.array(frames)  # [T, H, W, C]
print("image_frames.shape:", image_frames.shape)

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

# ============================================================
# 3. 加载模型
# ============================================================
print("Loading Alpamayo-R1-10B ...")
model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=DTYPE
).to(DEVICE)
model.eval()

processor = helper.get_processor(model.tokenizer)

# ============================================================
# 4. 官方 pipeline 生成 tokenized_data
# ============================================================
# 官方做法：processor 直接处理 image_frames
# 内部会自动生成与视觉 token 对齐的 input_ids
messages = helper.create_message(image_frames)  # 官方 message pipeline

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

print("==== TOKENIZED INPUTS ====")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape, v.dtype)
print("===========================\n")

# ============================================================
# 5. 构建模型输入
# ============================================================
model_inputs = {
    "tokenized_data": helper.to_device(inputs, DEVICE),
    "ego_history_xyz": ego_history_xyz.to(DEVICE),
    "ego_history_rot": ego_history_rot.to(DEVICE),
}

print("==== MODEL INPUTS READY ====\n")

# ============================================================
# 6. 推理
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
# 7. 输出预测结果
# ============================================================
print("pred_xyz.shape:", pred_xyz.shape)
print("pred_rot.shape:", pred_rot.shape)

if "cot" in extra:
    print("CoT:\n", extra["cot"][0])
else:
    print("No CoT returned")

traj_xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
print("Predicted XY trajectory:\n", traj_xy)
