"""
Video mouth ROI extraction for visual speech.

对应文档：
- Images-and-video-lab.pdf / Images-and-video-notes.pdf：
  视频逐帧读取、灰度化、ROI 操作
- visual-speech-features-notes.pdf：
  基于嘴部区域的视觉特征（mouth ROI）
- face_detection_script.py + haarcascades/*.xml：
  人脸 + 嘴部 Haar 级联检测 (Viola-Jones)

    ROOT          = 本文件所在目录 (例如 D:/av-lip-reader)
    VIDEO_ROOT    = ROOT / "long_videos"     # .mov 视频根目录
    ROI_ROOT      = ROOT / "roi_npy"         # ROI 序列输出目录
    CASCADE_DIR   = ROOT / "haarcascades"    # Haar 模型目录


- 递归扫描 VIDEO_ROOT 下所有 *.mov
- 对每个视频抽取嘴部 ROI 序列，尺寸统一为 64x64，归一化到 [0,1]
- 保存到 ROI_ROOT / "<原文件名>_roi.npy"，shape = (T, 64, 64)
"""

from pathlib import Path
import cv2 as cv
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = ROOT_DIR / "long_videos"
ROI_ROOT = ROOT_DIR / "roi_npy"
FACE_CASCADE_PATH = ROOT_DIR / "haarcascades" / "haarcascade_frontalface_default.xml"
MOUTH_CASCADE_PATH = ROOT_DIR / "haarcascades" / "haarcascade_smile.xml"


# ----------------- 检测器初始化（参考 face_detection_script.py） -----------------

def init_detectors():
    """
    加载 OpenCV Haar 级联分类器。

    对应文档：
    - haarcascades/haarcascade_frontalface_default.xml
    - haarcascades/haarcascade_smile.xml
    - face_detection_script.py 中的人脸与 smile 检测配置
    """
    face_cascade = cv.CascadeClassifier(str(FACE_CASCADE_PATH))
    mouth_cascade = cv.CascadeClassifier(str(MOUTH_CASCADE_PATH))

    if face_cascade.empty():
        raise RuntimeError(f"Failed to load face cascade: {FACE_CASCADE_PATH}")
    if mouth_cascade.empty():
        raise RuntimeError(f"Failed to load mouth cascade: {MOUTH_CASCADE_PATH}")

    return face_cascade, mouth_cascade


def select_main_face(faces):
    """
    选择面积最大的脸作为主脸。

    对应 visual-speech-features-notes.pdf 中“主目标选择”思路。
    """
    if len(faces) == 0:
        return None
    areas = [(w * h, (x, y, w, h)) for (x, y, w, h) in faces]
    areas.sort(key=lambda t: t[0], reverse=True)
    return areas[0][1]


# ----------------- 单帧嘴部 ROI 提取 -----------------

def extract_mouth_roi_from_frame(
    frame_bgr,
    face_cascade,
    mouth_cascade,
    target_size=(64, 64),
):
    """
    从单帧图像中提取嘴部 ROI。

    流程完全对齐文档：
    1. BGR 转灰度（Images-and-video-lab.pdf）
    2. Haar 人脸检测（face_detection_script.py / Viola-Jones）
    3. 取人脸下半部分作为嘴部搜索区域（face_detection_script.py）
    4. 在该区域上做 smile 检测，得到 mouth ROI
       （visual-speech-features-notes.pdf 中 mouth ROI 概念）
    5. ROI 统一 resize 到 target_size（size normalisation）
    """
    # 转灰度
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv.CASCADE_SCALE_IMAGE,
        minSize=(60, 60),
    )

    main_face = select_main_face(faces)
    if main_face is None:
        return None

    x, y, w, h = main_face

    # 人脸下半部作为嘴部候选区域（与 face_detection_script.py 一致）
    mouth_region_y = int(y + h * 0.55)
    mouth_region_h = int(h * 0.45)
    mouth_region = gray[mouth_region_y: mouth_region_y + mouth_region_h, x: x + w]
    if mouth_region.size == 0:
        return None

    # 在候选区域内检测“笑容”（mouth proxy）
    mouths = mouth_cascade.detectMultiScale(
        mouth_region,
        scaleFactor=1.1,
        minNeighbors=15,
        flags=cv.CASCADE_SCALE_IMAGE,
        minSize=(30, 15),
    )

    if len(mouths) > 0:
        # 选面积最大的 mouth 框
        areas = [(mw * mh, (mx, my, mw, mh)) for (mx, my, mw, mh) in mouths]
        areas.sort(key=lambda t: t[0], reverse=True)
        _, (mx, my, mw, mh) = areas[0]
        mouth_roi = mouth_region[my: my + mh, mx: mx + mw]
    else:
        # 检测不到时退化为整个 mouth_region 保证时序连续
        mouth_roi = mouth_region

    if mouth_roi.size == 0:
        return None

    # 统一尺寸（size normalisation）
    mouth_roi = cv.resize(mouth_roi, target_size, interpolation=cv.INTER_AREA)

    # 归一化到 [0,1]，方便后续 DCT / PCA（visual-speech-features-lab.pdf）
    mouth_roi = mouth_roi.astype(np.float32) / 255.0
    return mouth_roi


# ----------------- 单个视频处理 -----------------

def process_single_video(
    video_path: Path,
    out_dir: Path,
    face_cascade,
    mouth_cascade,
    every_nth: int = 1,
    max_frames=None,
    target_size=(64, 64),
):
    """
    将单个视频转成 mouth ROI 序列，保存为 .npy，shape=(T, H, W)。

    对应 Images-and-video-lab.pdf 中“逐帧处理视频”、
    visual-speech-features-lab.pdf 中对 mouth ROI 的时序建模。
    """
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    rois = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 every_nth 帧取一帧
        if frame_idx % every_nth != 0:
            frame_idx += 1
            continue

        roi = extract_mouth_roi_from_frame(
            frame, face_cascade, mouth_cascade, target_size=target_size
        )
        if roi is not None:
            rois.append(roi)

        frame_idx += 1
        if max_frames is not None and len(rois) >= max_frames:
            break

    cap.release()

    if len(rois) == 0:
        print(f"[WARN] No mouth ROI extracted for {video_path}")
        return

    rois = np.stack(rois, axis=0)  # (T, H, W)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_roi.npy"
    np.save(out_path, rois)

    print(f"[INFO] Saved ROI sequence: {out_path}  shape={rois.shape}")


# ----------------- 批量处理 long_videos 下所有 .mov -----------------

def process_all_videos():
    """
    批量处理 VIDEO_ROOT 下所有 *.mov。
    使用 Path.rglob("*.mov") 递归所有子目录。
    """
    face_cascade, mouth_cascade = init_detectors()

    video_root = VIDEO_ROOT
    out_root = ROI_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] VIDEO_ROOT = {video_root}")
    print(f"[INFO] ROI_ROOT   = {out_root}")

    video_files = sorted(video_root.rglob("*.mov"))
    if not video_files:
        print(f"[WARN] No .mov files found under {video_root}")
        return

    print(f"[INFO] Found {len(video_files)} video files.")

    for vp in video_files:
        process_single_video(
            video_path=vp,
            out_dir=out_root,
            face_cascade=face_cascade,
            mouth_cascade=mouth_cascade,
            every_nth=1,   # 可以按需要调整
            max_frames=80, # 每个视频最多 80 帧
            target_size=(64, 64),
        )

if __name__ == "__main__":
    process_all_videos()
