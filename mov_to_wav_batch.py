"""
Batch-extract audio (.wav) from .mov under long_videos / short_videos.

Specification:
- Output 16kHz, mono, aligned with Lab 1–4 / feature.py assumptions
- Keep base file name: video2_000.mov -> wav/video2_000.wav
- Requires ffmpeg installed on the system and available in PATH
"""

import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = ROOT_DIR / "short_videos"  # Input .mov root directory
WAV_ROOT = ROOT_DIR / "wav"  # Output wav root directory

FFMPEG = "ffmpeg"


def mov_to_wav_single(mov_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG,
        "-y",  # Overwrite without prompt
        "-i",
        str(mov_path),
        "-ac",
        "1",  # Mono
        "-ar",
        "16000",  # 16kHz
        "-vn",  # Drop video
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
