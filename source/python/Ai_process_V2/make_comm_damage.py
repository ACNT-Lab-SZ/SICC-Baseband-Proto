import cv2
import numpy as np
import os
import shutil
import subprocess
from pathlib import Path


def find_project_root(start_dir: Path) -> Path:
    for path in [start_dir, *start_dir.parents]:
        if (path / "requirements.txt").exists() and (path / "source").exists():
            return path
        if path.name == "organized_workspace_20260601":
            return path
    return start_dir


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
input_path = PROJECT_ROOT / "data" / "media" / "Ai_process_V2" / "sample_data" / "viso_002" / "002.avi"
output_dir = PROJECT_ROOT / "generated" / "media"
output_dir.mkdir(parents=True, exist_ok=True)
temp_video_path = output_dir / "002_comm_damage_noaudio.mp4"
output_path = output_dir / "002_comm_damage_like_demo.mp4"

np.random.seed(7)

cap = cv2.VideoCapture(str(input_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (w, h))

frame_idx = 0

def pixelate_region(region, scale=0.08):
    rh, rw = region.shape[:2]
    sw = max(2, int(rw * scale))
    sh = max(2, int(rh * scale))
    small = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

def add_noise(region, strength=35):
    noise = np.random.normal(0, strength, region.shape).astype(np.int16)
    out = np.clip(region.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out = frame.copy()

    # 1) 下半部分大面积宏块损坏，类似截图底部效果
    y0 = int(h * 0.52)
    lower = out[y0:h, 0:w].copy()

    # 让下半部分持续出现块状失真
    lower_pix = pixelate_region(lower, scale=0.06)

    # 叠加轻微噪声，让它不像单纯马赛克
    lower_pix = add_noise(lower_pix, strength=18)

    # 与原图混合，避免完全看不清
    alpha = 0.72
    lower_mixed = cv2.addWeighted(lower_pix, alpha, lower, 1 - alpha, 0)
    out[y0:h, 0:w] = lower_mixed

    # 2) 随机水平噪声条带，模拟局部码流损坏
    if frame_idx % 3 == 0:
        for _ in range(np.random.randint(2, 5)):
            band_h = np.random.randint(6, 22)
            by = np.random.randint(int(h * 0.55), h - band_h)
            band = out[by:by + band_h, :, :].copy()
            band = add_noise(band, strength=np.random.randint(45, 90))
            out[by:by + band_h, :, :] = band

    # 3) 随机黑块/灰块/色块，位置每帧变化
    if frame_idx % 4 == 0:
        num_blocks = np.random.randint(4, 10)
        for _ in range(num_blocks):
            bw = np.random.randint(max(20, w // 35), max(40, w // 10))
            bh = np.random.randint(max(16, h // 40), max(35, h // 8))
            bx = np.random.randint(0, max(1, w - bw))
            by = np.random.randint(int(h * 0.25), max(int(h * 0.95), h - bh))

            mode = np.random.choice(["black", "gray", "color", "pixel"])
            if mode == "black":
                color = np.random.randint(0, 35)
                out[by:by + bh, bx:bx + bw] = (color, color, color)
            elif mode == "gray":
                color = np.random.randint(70, 170)
                out[by:by + bh, bx:bx + bw] = (color, color, color)
            elif mode == "color":
                color = np.random.randint(0, 255, size=(1, 1, 3), dtype=np.uint8)
                out[by:by + bh, bx:bx + bw] = color
            else:
                patch = out[by:by + bh, bx:bx + bw].copy()
                patch = pixelate_region(patch, scale=0.10)
                patch = add_noise(patch, strength=40)
                out[by:by + bh, bx:bx + bw] = patch

    # 4) 每隔一段时间短暂花屏，持续若干帧
    burst_period = int(fps * 3.0)
    burst_len = int(fps * 0.25)

    if burst_period > 0 and frame_idx % burst_period < burst_len:
        gy0 = int(h * 0.42)
        glitch = out[gy0:h, :, :].copy()

        glitch = pixelate_region(glitch, scale=0.045)
        glitch = add_noise(glitch, strength=70)

        # 降低饱和度，提升对比度，制造通信花屏感
        hsv = cv2.cvtColor(glitch, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.35
        hsv[:, :, 2] *= 1.25
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        glitch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        out[gy0:h, :, :] = glitch

    writer.write(out)
    frame_idx += 1

cap.release()
writer.release()

print(f"Visual damaged video saved to: {temp_video_path}")

# 5) 使用 ffmpeg 把原视频音频拷贝回来
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is not None:
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(temp_video_path),
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"Final video saved to: {output_path}")
else:
    print("ffmpeg not found. The no-audio video has already been generated.")
