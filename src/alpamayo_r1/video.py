import torch
import numpy as np
import cv2

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# ------------------- 配置 -------------------
NUM_HISTORY_FRAMES = 3  # 历史帧数，根据模型训练设置
FRAME_HEIGHT = 128       # 模型期望输入分辨率 H
FRAME_WIDTH = 128        # 模型期望输入分辨率 W

# ------------------- 初始化摄像头 -------------------
# cap = cv2.VideoCapture(0)  # 默认摄像头
cap = cv2.VideoCapture("../../test.MOV")
frame_list = []

print("Capturing frames from camera...")
while len(frame_list) < NUM_HISTORY_FRAMES:
    ret, frame = cap.read()
    if not ret:
        continue
    # resize 到模型输入大小
    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    # 转 RGB (OpenCV 默认 BGR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_list.append(frame_rgb)
cap.release()
print(f"{len(frame_list)} frames captured.")

# ------------------- 构建输入张量 -------------------
# [batch=1, frames, H, W, C]
# ------------------- 构建输入张量 -------------------
# [1, T, H, W, C]
image_frames = np.array(frame_list)[None, ...]

# → [T, C, H, W]  （Alpamayo 只接受这个）
image_frames_tensor = (
    torch.from_numpy(image_frames)
    .permute(0, 1, 4, 2, 3)   # [1, T, C, H, W]
    .squeeze(0)              # [T, C, H, W]
    .float()
)

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
# ------------------- 创建消息 -------------------

messages = helper.create_message(image_frames_tensor)


# 不使用 processor
model_inputs = {
    "tokenized_data": messages,
    "ego_history_xyz": ego_history_xyz,
    "ego_history_rot": ego_history_rot,
}
# model_inputs = helper.to_device(model_inputs, "cuda")

# ------------------- 加载模型 -------------------
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)

# ------------------- 应用 processor -------------------
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": ego_history_xyz,
    "ego_history_rot": ego_history_rot,
}
model_inputs = helper.to_device(model_inputs, "cuda")

# ------------------- 固定随机种子 -------------------
torch.cuda.manual_seed_all(42)

# ------------------- 推理 -------------------
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,  # 部署时通常为1
        max_generation_length=256,
        return_extra=True,
    )

# ------------------- 输出预测结果 -------------------
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])
print("Predicted trajectory shape:", pred_xyz.shape)  # [batch, traj_set, traj_sample, time, xyz]

# 可选：提取 XY 平面轨迹
pred_xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]  # 第一条轨迹
print("Predicted XY trajectory:\n", pred_xy)
