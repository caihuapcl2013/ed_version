import torch
import numpy as np
import argparse
import cv2

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_local_video import load_physical_aiavdataset_video
from alpamayo_r1 import helper


def infer_one_clip(model, processor, video_path, t0_us, device="cuda"):
    print(f"\n🚀 Inference at t0_us = {t0_us} ({t0_us/1e6:.2f}s)")

    data = load_physical_aiavdataset_video(video_path, t0_us=t0_us)

    messages = helper.create_message(data["image_frames"].flatten(0, 1))

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
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }

    model_inputs = helper.to_device(model_inputs, device)

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,     # 显存安全
            max_generation_length=256,
            return_extra=True,
        )

    # ===== 评估 minADE =====
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()

    print("🧠 CoT:", extra["cot"][0])
    print("📏 minADE:", min_ade, "meters")

    return min_ade


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps   # seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="video path")
    parser.add_argument("--interval", type=float, default=1.0, help="interval seconds")
    parser.add_argument("--start", type=float, default=0.0, help="start time (s)")
    parser.add_argument("--end", type=float, default=-1.0, help="end time (s), -1 = full video")
    args = parser.parse_args()

    device = "cuda"

    print("📦 Loading Alpamayo-R1 model ...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16
    ).to(device)

    processor = helper.get_processor(model.tokenizer)

    # 获取视频时长
    duration = get_video_duration(args.video_path)
    print("🎬 Video duration:", duration, "seconds")

    start_t = args.start
    end_t = duration if args.end < 0 else args.end

    t = start_t
    results = []

    while t < end_t:
        t0_us = int(t * 1e6)   # 秒 → 微秒
        min_ade = infer_one_clip(
            model, processor,
            args.video_path,
            t0_us,
            device
        )
        results.append((t, min_ade))
        t += args.interval

    # ===== 汇总结果 =====
    print("\n================ SUMMARY ================")
    for t, ade in results:
        print(f"time {t:.2f}s -> minADE {ade:.3f} m")

    avg_ade = np.mean([x[1] for x in results])
    print("\n📊 AVG minADE:", avg_ade, "meters")


if __name__ == "__main__":
    main()
