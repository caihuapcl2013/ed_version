import os
import cv2
import torch
import numpy as np

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# ====================================================
# 配置部分
# ====================================================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # optional for easier debugging
DEVICE = "cuda"
DTYPE = torch.bfloat16

NUM_HISTORY_FRAMES = 4
FRAME_H = FRAME_W = 224  # 与官方训练一致
VIDEO_PATH = "../../video.MOV"

print("Device:", torch.cuda.get_device_name(0))

# ====================================================
# 读取 video frames
# official pipeline expects time-aligned multi-frame context
frames = []
cap = cv2.VideoCapture(VIDEO_PATH)
while len(frames) < NUM_HISTORY_FRAMES:
    success, frame = cap.read()
    if not success:
        break
    # resize to official resolution
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    # convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()

assert len(frames) == NUM_HISTORY_FRAMES, "Cannot read enough video frames"

image_array = np.array(frames)  # [T, H, W, C]
print("Loaded frames:", image_array.shape)

# ====================================================
# Ego history (dummy zeros)
ego_history_xyz = torch.zeros(1,1,NUM_HISTORY_FRAMES,3)
ego_history_rot = (
    torch.eye(3)
    .unsqueeze(0).unsqueeze(0).unsqueeze(0)
    .repeat(1,1,NUM_HISTORY_FRAMES,1,1)
)

# ====================================================
# Load model & processor
# ====================================================
print("Loading model...")
model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=DTYPE,
).to(DEVICE)
model.eval()

processor = helper.get_processor(model.tokenizer)

# ====================================================
# Official processor pipeline (align tokens)
# ====================================================
messages = helper.create_message(image_array)

tokenized = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

print("Token lengths:", tokenized["input_ids"].shape)

# ====================================================
# Build model input dict
# ====================================================
model_inputs = {
    "tokenized_data": helper.to_device(tokenized, DEVICE),
    "ego_history_xyz": ego_history_xyz.to(DEVICE),
    "ego_history_rot": ego_history_rot.to(DEVICE),
}

print("Running inference...")

# ====================================================
# Inference
# ====================================================
with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        num_traj_samples=1,
        max_generation_length=256,
        temperature=0.6,
        top_p=0.98,
        return_extra=True,
    )

print("Traj shape:", pred_xyz.shape)
if "cot" in extra:
    print("CoT:", extra["cot"][0])
