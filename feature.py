"""
Audio-Visual feature extraction (no CLI args by default).

- Extract audio MFCC(+Δ+ΔΔ) from wav
- Extract from mouth ROI sequences (*.npy):
    * Shape-based features (mouth height / width)
    * Image-based appearance features (low-frequency 2D DCT)
    * PCA appearance features (Eigenlips)
    * Hybrid features (shape + DCT)
- Audio and Visual are paired via file name prefix (<base_id>.wav ↔ <base_id>_roi.npy)
"""

from pathlib import Path

import cv2 as cv
import librosa
import numpy as np
from joblib import dump, load
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA

# ===================== Path configuration =====================

ROOT_DIR = Path(__file__).resolve().parent

# Inputs
ROI_DIR = ROOT_DIR / "roi_npy"  # Output from video_mouth_roi_extractor.py
WAV_DIR = ROOT_DIR / "wav"      # Output from mov_to_wav_batch.py

# Top-level output directories
AUDIO_OUT_DIR = ROOT_DIR / "audio_feats"
VISUAL_OUT_DIR = ROOT_DIR / "visual_feats"

# Audio subdirectories
AUDIO_MFCC_DIR = AUDIO_OUT_DIR / "mfcc"

# Visual subdirectories
VISUAL_SHAPE_DIR = VISUAL_OUT_DIR / "shape"
VISUAL_DCT_DIR = VISUAL_OUT_DIR / "dct"
VISUAL_PCA_DIR = VISUAL_OUT_DIR / "pca"
VISUAL_HYBRID_DIR = VISUAL_OUT_DIR / "hybrid"

# PCA model
MODELS_DIR = ROOT_DIR / "models"
PCA_MODEL_PATH = MODELS_DIR / "pca_eigenlips.joblib"

# DCT & PCA dimensions
N_DCT_COEFFS = 40
N_PCA_COMPONENTS = 30


# ===================== Audio features: MFCC (librosa) =====================


def extract_audio_mfcc_librosa(
    wav_path: Path,
    sr_target: int = 16000,
    n_mfcc: int = 13,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """
    Extract MFCC + Δ + ΔΔ with librosa, output shape: (D, T).
    """
    y, sr = librosa.load(str(wav_path), sr=sr_target)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    n_fft = int(sr * win_ms / 1000.0)
    hop_length = int(sr * hop_ms / 1000.0)

    # MFCC (including 0th cepstral)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )  # (n_mfcc, T)

    # First- and second-order deltas
    delta = librosa.feature.delta(mfcc, order=1)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    feats = np.vstack([mfcc, delta, delta_delta]).astype(np.float32)  # (3*n_mfcc, T)
    return feats


# ===================== Visual shape features: mouth height / width =====================


def mouth_shape_features(roi_frame: np.ndarray) -> np.ndarray:
    """
    Extract shape-based features from a single mouth ROI frame: height, width.

    Input:
        roi_frame: (H, W) float32, recommended normalized to [0,1]
    Output:
        np.array([height, width], dtype=float32)
    """
    # Convert to uint8 for Otsu threshold
    roi_uint8 = (roi_frame * 255).astype(np.uint8)

    try:
        thr = threshold_otsu(roi_uint8)
        binary = roi_uint8 > thr
    except Exception:
        # Fallback to fixed threshold if Otsu fails
        binary = roi_uint8 > 30

    lbl = label(binary)
    regions = regionprops(lbl)

    if not regions:
        return np.array([0.0, 0.0], dtype=np.float32)

    # Take the largest connected component as the lip region
    region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox
    height = float(maxr - minr)
    width = float(maxc - minc)

    return np.array([height, width], dtype=np.float32)


# ===================== Visual appearance features: low-frequency 2D DCT =====================


def dct_lowfreq_features(
    roi_frame: np.ndarray, num_coeff: int = N_DCT_COEFFS
) -> np.ndarray:
    """
    Apply 2D DCT on a single ROI frame and take the first num_coeff coefficients
    in diagonal (zig-zag) order.

    Returns:
        A 1D vector of shape (num_coeff,)
    """
    # cv.dct requires float32
    img = roi_frame.astype(np.float32)
    dct_mat = cv.dct(img)  # 2D DCT, shape=(H, W)

    h, w = dct_mat.shape
    coeffs: list[float] = []

    # Simple zig-zag: walk along diagonals starting at (0, 0)
    for s in range(h + w - 1):
        for i in range(s + 1):
            j = s - i
            if i < h and j < w:
                coeffs.append(float(dct_mat[i, j]))
                if len(coeffs) >= num_coeff:
                    return np.array(coeffs, dtype=np.float32)

    # If ROI is too small, we may have fewer than num_coeff coefficients
    return np.array(coeffs, dtype=np.float32)


# ===================== Visual appearance features: PCA (Eigenlips) =====================


def fit_pca_on_roi_dir(
    roi_dir: Path,
    n_components: int = N_PCA_COMPONENTS,
    pca_model_path: Path | None = None,
) -> PCA:
    """
    Fit PCA on the entire ROI directory for image-based appearance PCA features.

    Each .npy file in roi_dir is expected to be:
        <base_id>_roi.npy, shape=(T, H, W)
    """
    roi_files = sorted(roi_dir.glob("*_roi.npy"))
    if not roi_files:
        raise RuntimeError(f"No *_roi.npy files found in {roi_dir}")

    all_flat: list[np.ndarray] = []

    for roi_path in roi_files:
        roi_seq = np.load(roi_path)
        if roi_seq.ndim != 3:
            continue
        T, H, W = roi_seq.shape
        flat = roi_seq.reshape(T, H * W)  # (T, H*W)
        all_flat.append(flat)

    if not all_flat:
        raise RuntimeError(f"No valid ROI sequences found in {roi_dir}")

    X = np.vstack(all_flat).astype(np.float32)  # (N_frames_total, H*W)

    n_components_eff = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components_eff, whiten=False, svd_solver="randomized")
    pca.fit(X)

    if pca_model_path is not None:
        pca_model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(pca, str(pca_model_path))
        print(f"[INFO] PCA model saved to {pca_model_path}")

    return pca


def load_pca(pca_model_path: Path) -> PCA:
    """Load a previously fitted PCA model."""
    return load(str(pca_model_path))


# ===================== Main pipeline: align audio / visual file names and output features =====================


def process_av_features_default() -> None:
    """
    Inputs:
    - Under ROI_DIR: <base_id>_roi.npy
    - Under WAV_DIR: <base_id>.wav

    Outputs:
    - audio_feats/mfcc/:
        <base_id>_audio_mfcc.npy shape=(D_a, T_a)
    - visual_feats/shape/:
        <base_id>_shape.npy   shape=(T_v, 2)
    - visual_feats/dct/:
        <base_id>_dct.npy     shape=(T_v, N_DCT_COEFFS)
    - visual_feats/pca/:
        <base_id>_pca.npy     shape=(T_v, n_pca_eff)
    - visual_feats/hybrid/:
        <base_id>_hybrid.npy  shape=(T_v, 2+N_DCT_COEFFS)
    """
    # Directory mapping
    roi_dir = ROI_DIR
    wav_dir = WAV_DIR

    audio_mfcc_dir = AUDIO_MFCC_DIR

    visual_shape_dir = VISUAL_SHAPE_DIR
    visual_dct_dir = VISUAL_DCT_DIR
    visual_pca_dir = VISUAL_PCA_DIR
    visual_hybrid_dir = VISUAL_HYBRID_DIR

    pca_model_path = PCA_MODEL_PATH

    # Ensure input / output directories exist
    wav_dir.mkdir(exist_ok=True)
    roi_dir.mkdir(exist_ok=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_mfcc_dir.mkdir(parents=True, exist_ok=True)
    visual_shape_dir.mkdir(parents=True, exist_ok=True)
    visual_dct_dir.mkdir(parents=True, exist_ok=True)
    visual_pca_dir.mkdir(parents=True, exist_ok=True)
    visual_hybrid_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] ROI_DIR        = {roi_dir}")
    print(f"[INFO] WAV_DIR        = {wav_dir}")
    print(f"[INFO] AUDIO_MFCC_OUT = {audio_mfcc_dir}")
    print(f"[INFO] VISUAL_SHAPE   = {visual_shape_dir}")
    print(f"[INFO] VISUAL_DCT     = {visual_dct_dir}")
    print(f"[INFO] VISUAL_PCA     = {visual_pca_dir}")
    print(f"[INFO] VISUAL_HYBRID  = {visual_hybrid_dir}")
    print(f"[INFO] PCA_MODEL      = {pca_model_path}")

    # 1) Fit PCA (Eigenlips) on all ROIs
    pca = fit_pca_on_roi_dir(
        roi_dir=roi_dir,
        n_components=N_PCA_COMPONENTS,
        pca_model_path=pca_model_path,
    )

    # 2) Process each base_id
    roi_files = sorted(roi_dir.glob("*_roi.npy"))
    if not roi_files:
        print(f"[WARN] No *_roi.npy in {roi_dir}")
        return

    for roi_path in roi_files:
        base_id = roi_path.stem.replace("_roi", "")  # e.g. xxx_roi → xxx
        wav_path = wav_dir / f"{base_id}.wav"

        # ========== Audio features ==========
        if not wav_path.exists():
            print(f"[WARN] Missing audio file for {base_id}: {wav_path}")
        else:
            audio_feats = extract_audio_mfcc_librosa(wav_path)
            audio_out_path = audio_mfcc_dir / f"{base_id}_audio_mfcc.npy"
            np.save(audio_out_path, audio_feats.astype(np.float32))
            print(
                f"[INFO] Saved audio MFCC: {audio_out_path}, shape={audio_feats.shape}"
            )

        # ========== Visual features ==========
        roi_seq = np.load(roi_path)  # (T, H, W)
        if roi_seq.ndim != 3:
            print(f"[WARN] ROI has wrong shape: {roi_path}, shape={roi_seq.shape}")
            continue

        T, H, W = roi_seq.shape

        shape_list: list[np.ndarray] = []
        dct_list: list[np.ndarray] = []

        # Extract shape / DCT per frame
        for t in range(T):
            frame = roi_seq[t, :, :].astype(np.float32)  # (H, W)

            # Simple safeguard: if not in [0,1], normalize
            max_val = float(np.max(frame))
            if max_val > 0:
                frame_norm = frame / max_val
            else:
                frame_norm = frame

            shape_feat = mouth_shape_features(frame_norm)
            dct_feat = dct_lowfreq_features(frame_norm, num_coeff=N_DCT_COEFFS)
            shape_list.append(shape_feat)
            dct_list.append(dct_feat)

        shape_arr = np.stack(shape_list, axis=0).astype(np.float32)  # (T, 2)
        dct_arr = np.stack(dct_list, axis=0).astype(np.float32)      # (T, N_DCT_COEFFS)

        # PCA appearance: flatten each frame and transform
        flat = roi_seq.reshape(T, H * W).astype(np.float32)          # (T, H*W)
        pca_arr = pca.transform(flat).astype(np.float32)             # (T, n_pca_eff)

        # Hybrid: shape + DCT
        hybrid_arr = np.concatenate(
            [shape_arr, dct_arr], axis=1
        )  # (T, 2 + N_DCT_COEFFS)

        # Persist per subdirectory
        np.save(visual_shape_dir / f"{base_id}_shape.npy", shape_arr)
        np.save(visual_dct_dir / f"{base_id}_dct.npy", dct_arr)
        np.save(visual_pca_dir / f"{base_id}_pca.npy", pca_arr)
        np.save(visual_hybrid_dir / f"{base_id}_hybrid.npy", hybrid_arr)

        print(
            f"[INFO] {base_id}: shape={shape_arr.shape}, "
            f"dct={dct_arr.shape}, pca={pca_arr.shape}, hybrid={hybrid_arr.shape}"
        )


def main():
    process_av_features_default()


if __name__ == "__main__":
    main()
