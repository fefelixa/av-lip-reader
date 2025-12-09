import numpy as np
import matplotlib
matplotlib.use("Agg")      # 关键：使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ROI_DIR = ROOT_DIR / "roi_npy"

def main():
    files = sorted(ROI_DIR.glob("*.npy"))
    if not files:
        print("[ERROR] no roi .npy found")
        return

    roi_path = files[0]  # 随便选一个，看第一帧
    print(f"[INFO] loading {roi_path}")

    roi_seq = np.load(roi_path)   # shape ~ (T, 64, 64)
    print("roi shape:", roi_seq.shape)

    frame = roi_seq[0]

    plt.imshow(frame, cmap="gray")
    plt.title(roi_path.name)

    out_path = ROOT_DIR / "debug_roi.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[INFO] saved roi image to {out_path}")

if __name__ == "__main__":
    main()
