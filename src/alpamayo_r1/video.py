import torch
import numpy as np
import cv2

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

DEVICE = "cuda"
DTYPE = torch.bfloat16

NUM_HISTORY_FRAMES = 4
FRAME_H, FRAME_W = 128, 128
VIDEO_PATH = "../../test.MOV"

# ------------------ read frames ------------------
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
assert len(frames) == NUM_HISTORY_FRAMES

image_frames = torch.from_numpy(np.array(frames)).permute(0,3,1,2).float()

# ------------------ ego history ------------------
ego_history_xyz = torch.zeros(1, 1, NUM_HISTORY_FRAMES, 3)
ego_history_rot = (
    torch.eye(3)
    .unsqueeze(0).unsqueeze(0).unsqueeze(0)
    .repeat(1, 1, NUM_HISTORY_FRAMES, 1, 1)
)

# ------------------ message ------------------
messages = helper.create_message(image_frames)

# ------------------ model ------------------
model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=DTYPE
).to(DEVICE)
model.eval()

processor = helper.get_processor(model.tokenizer)

# ------------------ CRITICAL FIX ------------------
vlm = model.vlm
vlm.generation_config.bos_token_id = None
vlm.generation_config.eos_token_id = None
vlm.generation_config.pad_token_id = model.tokenizer.pad_token_id

# ------------------ tokenize ------------------
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

model_inputs = {
    "tokenized_data": helper.to_device(inputs, DEVICE),
    "ego_history_xyz": ego_history_xyz.to(DEVICE),
    "ego_history_rot": ego_history_rot.to(DEVICE),
}

# ------------------ inference ------------------
with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        num_traj_samples=1,
        max_generation_length=256,
        temperature=0.6,
        top_p=0.98,
        return_extra=True,
    )

print("Trajectory shape:", pred_xyz.shape)
print("CoT:", extra["cot"][0])
