"""
Video mouth ROI extraction for visual speech.

Relevant documents:
- Images-and-video-lab.pdf / Images-and-video-notes.pdf:
  frame-wise video reading, grayscale conversion, ROI operations
- visual-speech-features-notes.pdf:
  visual features based on mouth region (mouth ROI)
- face_detection_script.py + haarcascades/*.xml:
  face + mouth Haar cascade detection (Viola-Jones)

    ROOT          = directory of this file (e.g. D:/av-lip-reader)
    VIDEO_ROOT    = ROOT / "short_videos"     # .mov video root directory
    ROI_ROOT      = ROOT / "roi_npy"          # ROI sequence output directory
    CASCADE_DIR   = ROOT / "haarcascades"     # Haar model directory

Pipeline:
- Recursively scan all *.mov under VIDEO_ROOT
- For each video, extract mouth ROI sequence, normalize size to 64x64, values in [0,1]
- Save to ROI_ROOT / "<original file name>_roi.npy", shape = (T, 64, 64)
"""

from pathlib import Path
import cv2 as cv
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = ROOT_DIR / "short_videos"
ROI_ROOT = ROOT_DIR / "roi_npy"
FACE_CASCADE_PATH = ROOT_DIR / "haarcascades" / "haarcascade_frontalface_default.xml"
MOUTH_CASCADE_PATH = ROOT_DIR / "haarcascades" / "haarcascade_mcs_mouth.xml"


# ----------------- Detector initialization (refer to face_detection_script.py) -----------------


def init_detectors():
    """
    Load OpenCV Haar cascade classifiers.

    Relevant documents:
    - haarcascades/haarcascade_frontalface_default.xml
    - haarcascades/haarcascade_smile.xml
    - Face and smile detection configuration in face_detection_script.py
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
    Select the largest face as the primary face.

    Consistent with the “main target selection” logic in visual-speech-features-notes.pdf.
    """
    if len(faces) == 0:
        return None
    areas = [(w * h, (x, y, w, h)) for (x, y, w, h) in faces]
    areas.sort(key=lambda t: t[0], reverse=True)
    return areas[0][1]


# ----------------- Single-frame mouth ROI extraction -----------------


def extract_mouth_roi_from_frame(
    frame_bgr,
    face_cascade,
    mouth_cascade,
    target_size=(64, 64),
):
    """
    Extract mouth ROI from a single frame.

    Process aligned with the docs:
    1. Convert BGR to grayscale (Images-and-video-lab.pdf)
    2. Haar face detection (face_detection_script.py / Viola-Jones)
    3. Take the lower half of the face as the mouth search region (face_detection_script.py)
    4. Run smile detection on this region to approximate the mouth ROI
       (mouth ROI concept in visual-speech-features-notes.pdf)
    5. Resize ROI to target_size for size normalization
    """
    # Convert to grayscale
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)

    # Face detection
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

    # Lower half of the face as mouth candidate region (same as face_detection_script.py)
    mouth_region_y = int(y + h * 0.55)
    mouth_region_h = int(h * 0.45)
    mouth_region = gray[mouth_region_y: mouth_region_y + mouth_region_h, x: x + w]
    if mouth_region.size == 0:
        return None

    # Detect “smile” in candidate region as a proxy for mouth
    mouths = mouth_cascade.detectMultiScale(
        mouth_region,
        scaleFactor=1.1,
        minNeighbors=15,
        flags=cv.CASCADE_SCALE_IMAGE,
        minSize=(30, 15),
    )

    if len(mouths) > 0:
        # Select the largest mouth box
        areas = [(mw * mh, (mx, my, mw, mh)) for (mx, my, mw, mh) in mouths]
        areas.sort(key=lambda t: t[0], reverse=True)
        _, (mx, my, mw, mh) = areas[0]
        mouth_roi = mouth_region[my: my + mh, mx: mx + mw]
    else:
        # Fallback: use entire mouth_region to keep temporal continuity
        mouth_roi = mouth_region

    if mouth_roi.size == 0:
        return None

    # Size normalization
    mouth_roi = cv.resize(mouth_roi, target_size, interpolation=cv.INTER_AREA)

    # Normalize to [0,1] for downstream DCT / PCA (visual-speech-features-lab.pdf)
    mouth_roi = mouth_roi.astype(np.float32) / 255.0
    return mouth_roi


# ----------------- Single video processing -----------------


def process_single_video(
    video_path: Path,
    out_dir: Path,
    face_cascade,
    mouth_cascade,
    every_nth: int = 1,
    max_frames=None,
    target_size=(64, 64),
    debug: bool = False,
):
    """
    Convert a single video into a mouth ROI sequence and save as .npy, shape=(T, H, W).

    Consistent with:
    - Frame-wise video processing in Images-and-video-lab.pdf
    - Temporal modeling of mouth ROI in visual-speech-features-lab.pdf

    When debug=True:
    - Display the original frame and extracted mouth ROI in real time
    - Press 'q' to interrupt processing of the current video
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

        # Take every_nth frame
        if frame_idx % every_nth != 0:
            frame_idx += 1
            continue

        roi = extract_mouth_roi_from_frame(
            frame, face_cascade, mouth_cascade, target_size=target_size
        )
        if roi is not None:
            rois.append(roi)

            # --------- Debug visualization logic (option A) ---------
            if debug:
                # ROI is [0,1] float grayscale; convert to 0–255 for display
                roi_vis = (roi * 255).astype("uint8")

                cv.imshow("original frame", frame)
                cv.imshow("mouth roi", roi_vis)

                # 30ms per frame; press q to break current video
                key = cv.waitKey(30) & 0xFF
                if key == ord("q"):
                    print(f"[INFO] Debug interrupted by user on video: {video_path}")
                    break

        frame_idx += 1
        if max_frames is not None and len(rois) >= max_frames:
            break

    cap.release()
    if debug:
        cv.destroyAllWindows()

    if len(rois) == 0:
        print(f"[WARN] No mouth ROI extracted for {video_path}")
        return

    rois = np.stack(rois, axis=0)  # (T, H, W)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_roi.npy"
    np.save(out_path, rois)

    print(f"[INFO] Saved ROI sequence: {out_path}  shape={rois.shape}")


# ----------------- Batch process all .mov under short_videos -----------------


def process_all_videos(debug: bool = False):
    """
    Batch process all *.mov under VIDEO_ROOT.

    Uses Path.rglob("*.mov") to traverse all subdirectories.

    When debug=True, every video will pop up a window showing original frame + mouth ROI.
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
        print(f"[INFO] Processing: {vp}")
        process_single_video(
            video_path=vp,
            out_dir=out_root,
            face_cascade=face_cascade,
            mouth_cascade=mouth_cascade,
            every_nth=1,   # Adjust if needed
            max_frames=80,  # Max 80 frames per video
            target_size=(64, 64),
            debug=debug,
        )


if __name__ == "__main__":
    # Debug flag to inspect whether mouth ROI is correct
    process_all_videos(debug=False)
