"""
批量从 long_videos 下的 .mov 提取音频为 .wav。

规范：
- 输出为 16kHz、单声道，符合 Lab 1–4 / feature.py 的假设
- 名字保持一致：video2_000.mov -> wav/video2_000.wav
- 依赖系统已安装 ffmpeg，并且 ffmpeg 在 PATH 中
"""

import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = ROOT_DIR / "long_videos"  # 输入 .mov 根目录
WAV_ROOT = ROOT_DIR / "wav"  # 输出 wav 根目录

FFMPEG = "ffmpeg"


def mov_to_wav_single(mov_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG,
        "-y",  # 自动覆盖
        "-i",
        str(mov_path),
        "-ac",
        "1",  # 单声道
        "-ar",
        "16000",  # 16kHz
        "-vn",  # 去掉视频
        str(wav_path),
    ]
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def batch_convert(video_root: Path, wav_root: Path) -> None:
    video_root = video_root.resolve()
    wav_root = wav_root.resolve()

    mov_files = sorted(video_root.rglob("*.mov"))
    if not mov_files:
        print(f"[WARN] No .mov files found under {video_root}")
        return

    print(f"[INFO] Found {len(mov_files)} .mov files under {video_root}")
    for mov in mov_files:
        base_name = mov.stem  # video2_000.mov -> video2_000
        wav_path = wav_root / f"{base_name}.wav"
        try:
            mov_to_wav_single(mov, wav_path)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ffmpeg failed on {mov}: {e}")


def main():
    print(f"[INFO] VIDEO_ROOT = {VIDEO_ROOT}")
    print(f"[INFO] WAV_ROOT   = {WAV_ROOT}")
    batch_convert(VIDEO_ROOT, WAV_ROOT)


if __name__ == "__main__":
    main()
