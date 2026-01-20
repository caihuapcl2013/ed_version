import cv2
import torch
import numpy as np
from einops import rearrange
import io
from typing import Any

# ----------------------------
# 简单模拟 VideoReader 类，只保留 decode_images_from_timestamps 行为
# ----------------------------
class SimpleVideoReader:
    """模拟 PhysicalAIAV VideoReader 接口"""

    def __init__(self, video_path: str, fps: float = 30.0):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = fps
        self.timestamps = (np.arange(self.num_frames) / fps * 1_000_000).astype(np.int64)  # μs

    def decode_images_from_timestamps(self, requested_timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回最接近 requested_timestamps 的帧"""
        frame_indices = np.searchsorted(self.timestamps, requested_timestamps, side="right") - 1
        frame_indices = np.clip(frame_indices, 0, self.num_frames - 1)
        frames = []
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(f"无法读取帧 {idx}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return np.stack(frames), self.timestamps[frame_indices]

    def close(self):
        self.cap.release()


# ----------------------------
# load_physical_aiavdataset_video 函数
# ----------------------------
def load_physical_aiavdataset_video(
    video_path: str,
    num_frames: int = 4,
    t0_us: int = 5_100_000,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
) -> dict[str, Any]:
    """
    模拟 PhysicalAIAV load_physical_aiavdataset，仅生成 image_frames + 占位 ego motion
    """

    # 创建 VideoReader
    camera = SimpleVideoReader(video_path)
    
    # 图像时间戳（μs）：t0 前后连续 num_frames
    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )

    frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)
    frames_tensor = torch.from_numpy(frames)  # (num_frames, H, W, 3)
    frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")  # (num_frames, 3, H, W)
    image_frames = frames_tensor.unsqueeze(0)  # (N_cameras=1, num_frames, 3, H, W)

    # 占位 ego motion
    ego_history_xyz = torch.zeros((1, 1, num_history_steps, 3), dtype=torch.float32)
    ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 1, num_history_steps, 1, 1)
    ego_future_xyz = torch.zeros((1, 1, num_future_steps, 3), dtype=torch.float32)
    ego_future_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 1, num_future_steps, 1, 1)

    # camera_indices + timestamps
    camera_indices = torch.tensor([0], dtype=torch.int64)
    absolute_timestamps = torch.tensor([frame_timestamps], dtype=torch.int64)
    relative_timestamps = (absolute_timestamps - absolute_timestamps.min()).float() * 1e-6

    camera.close()

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "ego_future_xyz": ego_future_xyz,
        "ego_future_rot": ego_future_rot,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": absolute_timestamps,
        "t0_us": t0_us,
        "clip_id": "local_video",
    }

# ----------------------------
# 示例调用
# ----------------------------
# data = load_physical_aiavdataset_video("../../test.MOV")
# print("image_frames shape:", data["image_frames"].shape)
