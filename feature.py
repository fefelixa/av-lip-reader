"""
Audio-Visual feature extraction.

Inputs
------
- ROI sequences: ROOT/roi_npy/<base_id>_roi.npy, shape (T, H, W), float32 in [0,1]
- WAV audio:     ROOT/wav/<base_id>.wav

Outputs
-------
ROOT/
  audio_feats/
    mfcc/<base_id>_mfcc.npy            (Fa, Ta) == (D_a, TARGET_T)
  visual_feats/
    shape/<base_id>_shape.npy          (Tv, 2)          -> [mouth_h_norm, mouth_w_norm]
    dct/<base_id>_dct.npy              (Tv, Kdct)
    pca/<base_id>_pca.npy              (Tv_fix, Kpca)   -> (TARGET_T, Kpca)
    hybrid/<base_id>_hybrid.npy        (Tv, 2 + Kdct)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Optional deps (guarded imports so the file can be imported without them)
try:
    import cv2 as cv
except Exception as _:
    cv = None
try:
    import librosa
except Exception as _:
    librosa = None
try:
    from sklearn.decomposition import PCA
except Exception as _:
    PCA = None

ROOT_DIR = Path(__file__).resolve().parent

# Unified time length (must be aligned with av_model_training.py TARGET_T)
TARGET_T = 40

# Inputs
ROI_DIR = ROOT_DIR / "roi_npy"  # Output from video_mouth_roi_extractor.py
WAV_DIR = ROOT_DIR / "wav"      # Directory of wav files

# Outputs
AUDIO_OUT_DIR = ROOT_DIR / "audio_feats"
VISUAL_OUT_DIR = ROOT_DIR / "visual_feats"

AUDIO_MFCC_DIR = AUDIO_OUT_DIR / "mfcc"

VISUAL_SHAPE_DIR = VISUAL_OUT_DIR / "shape"
VISUAL_DCT_DIR   = VISUAL_OUT_DIR / "dct"
VISUAL_PCA_DIR   = VISUAL_OUT_DIR / "pca"
VISUAL_HYBRID_DIR= VISUAL_OUT_DIR / "hybrid"

# ------------------------- Utils -------------------------
def ensure_dirs():
    for d in [AUDIO_MFCC_DIR, VISUAL_SHAPE_DIR, VISUAL_DCT_DIR, VISUAL_PCA_DIR, VISUAL_HYBRID_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def list_base_ids() -> List[str]:
    """Return sorted list of base_ids derived from ROI files."""
    roi_files = sorted(ROI_DIR.glob("*_roi.npy"))
    ids = [p.name[:-8] for p in roi_files]  # strip "_roi.npy"
    return ids

def zscore_time(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-dimension z-score over time axis 0."""
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    return (x - m) / (s + eps)

def pad_or_crop_time_T_first(seq: np.ndarray, target_T: int) -> np.ndarray:
    """
    Align along the time dimension for (T, D) shaped sequences.

    Input:  shape=(T, D)
    Output: shape=(target_T, D)
    """
    if seq.ndim != 2:
        raise ValueError(f"expect (T, D), got {seq.shape}")

    T, D = seq.shape

    if T == target_T:
        return seq

    if T > target_T:
        start = (T - target_T) // 2
        end = start + target_T
        return seq[start:end, :]

    pad_len = target_T - T
    pad = np.zeros((pad_len, D), dtype=seq.dtype)
    return np.vstack([seq, pad])

def zigzag_indices(n: int):
    """Generate zigzag scan order for an n x n block."""
    idx = []
    for s in range(2*n - 1):
        if s % 2 == 0:
            r = min(s, n-1)
            c = s - r
            while r >= 0 and c < n:
                idx.append((r, c))
                r -= 1
                c += 1
        else:
            c = min(s, n-1)
            r = s - c
            while c >= 0 and r < n:
                idx.append((r, c))
                r += 1
                c -= 1
    return idx

_ZZ_CACHE = {}
def zigzag_flatten(block: np.ndarray, k: int) -> np.ndarray:
    """Return first k coefficients of the zigzag scan of block (n x n)."""
    n = block.shape[0]
    order = _ZZ_CACHE.get(n)
    if order is None:
        order = zigzag_indices(n)
        _ZZ_CACHE[n] = order
    flat = []
    for (r,c) in order[:k]:
        flat.append(block[r, c])
    return np.asarray(flat, dtype=np.float32)

# ------------------------- Audio -------------------------
def extract_mfcc(
    wav_path: Path,
    sr: int = 16000,
    n_mfcc: int = 13,
    target_T: int = TARGET_T,
) -> np.ndarray:
    """
    提取 MFCC + Δ + ΔΔ，并统一到固定时长，最终输出 (D_a, target_T)。

    - 原始 librosa: (T, n_mfcc)
    - 拼接 Δ/ΔΔ 后: (T, 3*n_mfcc)
    - 时间 z-score + pad/crop: (target_T, 3*n_mfcc)
    - 转置保存: (3*n_mfcc, target_T)
    """
    assert librosa is not None, "librosa is required for audio features"
    y, _sr = librosa.load(wav_path.as_posix(), sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T  # (T, n_mfcc)
    # Delta + Delta-Delta
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.concatenate([mfcc, d1, d2], axis=1).astype(np.float32)  # (T, 3*n_mfcc)

    # z-score over time for stability
    feat = zscore_time(feat)  # (T, 3*n_mfcc)

    # unify time length
    feat = pad_or_crop_time_T_first(feat, target_T=target_T)  # (target_T, 3*n_mfcc)

    # transpose to (D_a, T_a) expected by av_model_training.py
    feat = feat.T.astype(np.float32)  # (3*n_mfcc, target_T)

    return feat

# ------------------------- Visual -------------------------
def _mouth_hw_from_roi(roi: np.ndarray, thr_method: str = "otsu") -> Tuple[float, float]:
    """
    Estimate mouth opening height/width from a grayscale ROI in [0,1].
    Heuristic: threshold the dark interior and take the bbox of the largest blob.
    Fallback to zeros if nothing found.
    """
    if cv is None:
        # Fallback: no cv2, use simple threshold
        m = roi < 0.4
    else:
        roi8 = np.clip(roi * 255.0, 0, 255).astype(np.uint8)
        if thr_method == "otsu":
            _, m = cv.threshold(roi8, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        else:
            _, m = cv.threshold(roi8, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_TRIANGLE)
        m = m.astype(bool)
    if not m.any():
        return 0.0, 0.0
    # largest connected component
    if cv is not None:
        n, labels = cv.connectedComponents(m.astype(np.uint8))
        if n > 1:
            sizes = [(labels == i).sum() for i in range(1, n)]
            i_star = 1 + int(np.argmax(sizes))
            mask = labels == i_star
        else:
            mask = m
    else:
        mask = m
    ys, xs = np.where(mask)
    h = (ys.max() - ys.min() + 1) / roi.shape[0]
    w = (xs.max() - xs.min() + 1) / roi.shape[1]
    return float(h), float(w)

def extract_shape_features(roi_seq: np.ndarray) -> np.ndarray:
    """(T,H,W)->(T,2) normalized mouth height/width, values roughly in [0,1]."""
    T = roi_seq.shape[0]
    out = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        h, w = _mouth_hw_from_roi(roi_seq[t])
        out[t, 0] = h
        out[t, 1] = w
    return out

def extract_dct_features(roi_seq: np.ndarray, block_size: int = 32, k: int = 40) -> np.ndarray:
    """
    For each frame, compute 2D DCT of a resized square block and keep first k coeffs via zig-zag.
    Input ROI assumed to be in [0,1].
    """
    assert cv is not None, "opencv-python is required for visual DCT features"
    T = roi_seq.shape[0]
    feats = np.zeros((T, k), dtype=np.float32)
    for t in range(T):
        frame = roi_seq[t]
        if frame.shape[0] != block_size or frame.shape[1] != block_size:
            frame = cv.resize(frame, (block_size, block_size), interpolation=cv.INTER_AREA)
        block = np.float32(frame)
        dct = cv.dct(block)  # 2D DCT
        # take low-frequency coefficients from top-left via zig-zag
        feats[t] = zigzag_flatten(dct, k)
    # log-magnitude compress (except DC term)
    feats[:, 1:] = np.sign(feats[:, 1:]) * np.log1p(np.abs(feats[:, 1:]))
    # per-dimension z-score over time
    feats = zscore_time(feats)
    return feats

def extract_pca_features(
    roi_seq: np.ndarray,
    k: int = 50,
    target_T: int = TARGET_T,
) -> np.ndarray:
    """
    PCA over flattened frames within the clip (unsupervised, per-clip).

    目标:
    - 所有样本输出统一 shape = (TARGET_T, k)

    步骤:
    1) PCA 得到 (T, n_comp) 其中 n_comp <= min(k, H*W, T)
    2) 若 n_comp < k, 在特征维度补零; 若 n_comp > k, 截断到 k
    3) 时间维 z-score
    4) 时间轴 pad/crop 到 TARGET_T
    """
    assert PCA is not None, "scikit-learn is required for PCA features"
    T, H, W = roi_seq.shape
    X = roi_seq.reshape(T, H * W)

    n_comp = min(k, H * W, T)
    pca = PCA(n_components=n_comp)
    Xp = pca.fit_transform(X).astype(np.float32)  # (T, n_comp)

    # unify feature dimension to k
    cur_k = Xp.shape[1]
    if cur_k < k:
        pad = np.zeros((T, k - cur_k), dtype=Xp.dtype)
        Xp = np.concatenate([Xp, pad], axis=1)
    elif cur_k > k:
        Xp = Xp[:, :k]
    # now (T, k)

    Xp = zscore_time(Xp)  # stabilize scale

    # unify time dimension
    Xp = pad_or_crop_time_T_first(Xp, target_T=target_T)  # (TARGET_T, k)

    return Xp

def process_visual_single(base_id: str, save_pca: bool = True, k_dct: int = 40, k_pca: int = 50):
    roi_path = ROI_DIR / f"{base_id}_roi.npy"
    if not roi_path.exists():
        print(f"[WARN] ROI not found for base_id={base_id}")
        return
    roi_seq = np.load(roi_path).astype(np.float32)  # (T,H,W) in [0,1]

    # Features
    shape = extract_shape_features(roi_seq)                        # (T,2)
    dct   = extract_dct_features(roi_seq, block_size=32, k=k_dct) # (T,k_dct)
    if save_pca:
        pca = extract_pca_features(roi_seq, k=k_pca)              # (TARGET_T,k_pca)
    else:
        pca = None

    # Hybrid (shape + DCT). Shapes already normalized; dct z-scored.
    hybrid = np.concatenate([shape, dct], axis=1).astype(np.float32)

    # Persist
    np.save(VISUAL_SHAPE_DIR / f"{base_id}_shape.npy", shape.astype(np.float32))
    np.save(VISUAL_DCT_DIR   / f"{base_id}_dct.npy",   dct.astype(np.float32))
    if pca is not None:
        np.save(VISUAL_PCA_DIR   / f"{base_id}_pca.npy",   pca.astype(np.float32))
    np.save(VISUAL_HYBRID_DIR/ f"{base_id}_hybrid.npy", hybrid.astype(np.float32))

    print(
        f"[OK] {base_id}: shape={shape.shape}, dct={dct.shape}, "
        f"pca={(pca.shape if pca is not None else None)}, hybrid={hybrid.shape}"
    )

# ------------------------- Orchestrators -------------------------
def process_audio_default():
    wavs = sorted(WAV_DIR.glob("*.wav"))
    if not wavs:
        print(f"[WARN] No wav files under {WAV_DIR}")
        return
    for wp in wavs:
        base_id = wp.stem
        outp = AUDIO_MFCC_DIR / f"{base_id}_mfcc.npy"
        if outp.exists():
            continue
        try:
            feat = extract_mfcc(wp)  # (D_a, TARGET_T)
            np.save(outp, feat.astype(np.float32))
            print(f"[OK] {base_id}: mfcc={feat.shape}")
        except Exception as e:
            print(f"[ERR] MFCC failed for {wp}: {e}")

def process_visual_default(save_pca: bool = True):
    base_ids = list_base_ids()
    if not base_ids:
        print(f"[WARN] No ROI files under {ROI_DIR}")
        return
    for base_id in base_ids:
        try:
            process_visual_single(base_id, save_pca=save_pca)
        except Exception as e:
            print(f"[ERR] Visual feature failed for {base_id}: {e}")

def process_av_features_default():
    ensure_dirs()
    process_audio_default()
    process_visual_default(save_pca=True)

# ------------------------- CLI -------------------------
def main():
    process_av_features_default()

if __name__ == "__main__":
    main()
