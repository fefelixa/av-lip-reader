"""
Audio-Visual feature extraction (no CLI args by default).

对应文档：
- Lab 1 Sound analysis.pdf / Lab 2 Short time spectral processing.pdf /
  Lab 3 Filterbanks.pdf：短时分析、功率谱、Mel 滤波器组、DCT → MFCC
- visual-speech-features-lab.pdf：2D DCT 图像特征、视觉语音特征示例
- visual-speech-features-notes.pdf：appearance model / PCA (Eigenfaces / Eigenlips)
- Images-and-video-lab.pdf / Images-and-video-notes.pdf：ROI 处理、阈值分割、regionprops

- 从 wav 提取 audio MFCC(+Δ+ΔΔ)，使用 librosa 封装实现，遵循上述 lab 的流程设计
- 从 mouth ROI 序列 (*.npy) 提取：
    * Shape-based 特征（mouth height / width）
    * Image-based appearance 特征（2D DCT 低频）
    * PCA appearance 特征（Eigenlips）
    * Hybrid 特征（shape + DCT）
- Audio 与 Visual 通过文件名前缀配对（<base_id>.wav ↔ <base_id>_roi.npy）
"""

from pathlib import Path

import cv2 as cv
import librosa
import numpy as np
from joblib import dump, load
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA

ROOT_DIR = Path(__file__).resolve().parent

ROI_DIR = ROOT_DIR / "roi_npy"  # video_mouth_roi_extractor.py 输出
WAV_DIR = ROOT_DIR / "wav"  # 你从 .mov 抽出来的 wav
AUDIO_OUT_DIR = ROOT_DIR / "audio_feats"
VISUAL_OUT_DIR = ROOT_DIR / "visual_feats"
MODELS_DIR = ROOT_DIR / "models"
PCA_MODEL_PATH = MODELS_DIR / "pca_eigenlips.joblib"

# DCT & PCA 维度（来自 visual-speech-features-lab.pdf 中“保留若干低频系数”思路）
N_DCT_COEFFS = 40
N_PCA_COMPONENTS = 30


# ===================== Audio 特征：MFCC（librosa） =====================


def extract_audio_mfcc_librosa(
    wav_path: Path,
    sr_target: int = 16000,
    n_mfcc: int = 13,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """
    使用 librosa 提取 MFCC + Δ + ΔΔ，输出 shape: (D, T)。

    对应：
    - Lab 1：短时傅里叶变换（STFT）
    - Lab 2：帧长 ~25ms，帧移 ~10ms 的窗函数处理
    - Lab 3：Mel 滤波器组 + DCT → MFCC（这里用 librosa.feature.mfcc 封装）
    """
    y, sr = librosa.load(str(wav_path), sr=sr_target)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    n_fft = int(sr * win_ms / 1000.0)
    hop_length = int(sr * hop_ms / 1000.0)

    # MFCC (包含 0 阶 cepstral)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )  # (n_mfcc, T)

    # 一阶 / 二阶差分（dynamic features，lab 明确要求）
    delta = librosa.feature.delta(mfcc, order=1)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    feats = np.vstack([mfcc, delta, delta_delta]).astype(np.float32)  # (3*n_mfcc, T)
    return feats


# ===================== Visual 形状特征：mouth height / width =====================


def mouth_shape_features(roi_frame: np.ndarray) -> np.ndarray:
    """
    从单帧 mouth ROI 提取 shape-based 特征：height、width。

    - Images-and-video-lab.pdf：对灰度图做阈值 → 二值 mask
    - regionprops 计算连通域几何参数（面积 / bbox 等）
    - visual-speech-features-notes.pdf 提到的 “嘴巴高度/宽度作为 articulatory 特征”

    输入：
        roi_frame: (H, W) float32, 归一化 [0,1]
    输出：
        np.array([height, width], dtype=float32)
    """
    # 转为 uint8 方便 Otsu 阈值
    roi_uint8 = (roi_frame * 255).astype(np.uint8)

    try:
        thr = threshold_otsu(roi_uint8)
        binary = roi_uint8 > thr
    except Exception:
        # Otsu 失败时，fallback 固定阈值
        binary = roi_uint8 > 30

    lbl = label(binary)
    regions = regionprops(lbl)

    if not regions:
        return np.array([0.0, 0.0], dtype=np.float32)

    # 取面积最大的连通域作为嘴唇区域（coins 示例同样使用 area 最大）
    region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox
    height = float(maxr - minr)
    width = float(maxc - minc)

    return np.array([height, width], dtype=np.float32)


# ===================== Visual 外观特征：2D DCT 低频 =====================


def dct_lowfreq_features(
    roi_frame: np.ndarray, num_coeff: int = N_DCT_COEFFS
) -> np.ndarray:
    """
    对单帧 ROI 做 2D DCT，并按“从左上到右下”的对角线顺序取前 num_coeff 个系数。

    对应：
    - Images-and-video-lab.pdf 中的 2D DCT 实验（展示低频系数集中）
    - visual-speech-features-lab.pdf 中 “DCT-based appearance features” 描述

    返回：一维向量 (num_coeff,)
    """
    # cv.dct 要求 float32
    img = roi_frame.astype(np.float32)
    dct_mat = cv.dct(img)  # 2D DCT，shape=(H, W)

    h, w = dct_mat.shape
    coeffs = []

    # 简单 zig-zag：按对角线从 (0,0) 开始
    for s in range(h + w - 1):
        for i in range(s + 1):
            j = s - i
            if i < h and j < w:
                coeffs.append(dct_mat[i, j])
                if len(coeffs) >= num_coeff:
                    return np.array(coeffs, dtype=np.float32)

    # 如果 ROI 尺寸太小，系数可能不够 num_coeff
    return np.array(coeffs, dtype=np.float32)


# ===================== Visual 外观特征：PCA（Eigenlips） =====================


def fit_pca_on_roi_dir(
    roi_dir: Path,
    n_components: int = N_PCA_COMPONENTS,
    pca_model_path: Path | None = None,
) -> PCA:
    """
    在整个 ROI 目录上拟合 PCA，用于 Image-based Appearance PCA 特征。

    对齐 visual-speech-features-notes.pdf：
    - 将 ROI 展开为一维向量
    - 对所有样本做 PCA，得到 Eigenfaces/Eigenlips
    - 使用投影系数作为外观特征

    roi_dir 中每个 .npy 文件形如：
        <base_id>_roi.npy，shape=(T, H, W)
    """
    roi_files = sorted(roi_dir.glob("*_roi.npy"))
    if not roi_files:
        raise RuntimeError(f"No *_roi.npy found in {roi_dir}")

    all_flat: list[np.ndarray] = []

    for roi_path in roi_files:
        roi_seq = np.load(roi_path)  # (T, H, W)
        if roi_seq.ndim != 3:
            continue
        T, H, W = roi_seq.shape
        flat = roi_seq.reshape(T, H * W)  # (T, H*W)
        all_flat.append(flat)

    X = np.vstack(all_flat).astype(np.float32)  # (N_frames_total, H*W)

    n_components_eff = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components_eff, whiten=False, svd_solver="randomized")
    pca.fit(X)

    if pca_model_path is not None:
        pca_model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(pca, str(pca_model_path) + '.pca')
        print(f"[INFO] PCA model saved to {pca_model_path}")

    return pca


def load_pca(pca_model_path: Path) -> PCA:
    """加载之前拟合好的 PCA 模型。"""
    return load(pca_model_path)


# ===================== 主流程：对齐 audio / visual 文件名并输出多种特征 =====================


def process_av_features_default() -> None:
    """
    输入：
    - ROI_DIR  下：<base_id>_roi.npy
    - WAV_DIR  下：<base_id>.wav

    输出：
    - AUDIO_OUT_DIR 下：
        <base_id>_audio_mfcc.npy shape=(D_a, T_a)
    - VISUAL_OUT_DIR 下：
        <base_id>_shape.npy   shape=(T_v, 2)
        <base_id>_dct.npy     shape=(T_v, N_DCT_COEFFS)
        <base_id>_pca.npy     shape=(T_v, n_pca_eff)
        <base_id>_hybrid.npy  shape=(T_v, 2+N_DCT_COEFFS)

    这些输出与 visual-speech-features-lab.pdf 中“多种视觉特征 + Early Integration”要求一致。
    """
    roi_dir = ROI_DIR
    wav_dir = WAV_DIR
    audio_out_dir = AUDIO_OUT_DIR
    visual_out_dir = VISUAL_OUT_DIR
    pca_model_path = PCA_MODEL_PATH
    wav_dir.mkdir(exist_ok=True)
    roi_dir.mkdir(exist_ok=True)
    pca_model_path.mkdir(exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    visual_out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] ROI_DIR     = {roi_dir}")
    print(f"[INFO] WAV_DIR     = {wav_dir}")
    print(f"[INFO] AUDIO_OUT   = {audio_out_dir}")
    print(f"[INFO] VISUAL_OUT  = {visual_out_dir}")
    print(f"[INFO] PCA_MODEL   = {pca_model_path}")

    # 1）在全部 ROI 上拟合 PCA（Eigenlips）
    pca = fit_pca_on_roi_dir(
        roi_dir=roi_dir,
        n_components=N_PCA_COMPONENTS,
        pca_model_path=pca_model_path,
    )

    # 2）逐个 base_id 处理
    roi_files = sorted(roi_dir.glob("*_roi.npy"))
    if not roi_files:
        print(f"[WARN] No *_roi.npy in {roi_dir}")
        return

    for roi_path in roi_files:
        base_id = roi_path.stem.replace("_roi", "")  # 例如 video2_000_roi → video2_000
        wav_path = wav_dir / f"{base_id}.wav"

        # ========== Audio 特征 ==========
        if not wav_path.exists():
            print(f"[WARN] Missing audio file for {base_id}: {wav_path}")
        else:
            audio_feats = extract_audio_mfcc_librosa(wav_path)
            audio_out_path = audio_out_dir / f"{base_id}_audio_mfcc.npy"
            np.save(audio_out_path, audio_feats.astype(np.float32))
            print(
                f"[INFO] Saved audio MFCC: {audio_out_path}, shape={audio_feats.shape}"
            )

        # ========== Visual 特征 ==========
        roi_seq = np.load(roi_path)  # (T, H, W)
        if roi_seq.ndim != 3:
            print(f"[WARN] ROI has wrong shape: {roi_path}, shape={roi_seq.shape}")
            continue

        T, H, W = roi_seq.shape

        shape_list = []
        dct_list = []

        for t in range(T):
            frame = roi_seq[t, :, :]  # (H, W), 已在 [0,1]
            shape_feat = mouth_shape_features(frame)
            dct_feat = dct_lowfreq_features(frame, num_coeff=N_DCT_COEFFS)
            shape_list.append(shape_feat)
            dct_list.append(dct_feat)

        shape_arr = np.stack(shape_list, axis=0).astype(np.float32)  # (T, 2)
        dct_arr = np.stack(dct_list, axis=0).astype(np.float32)  # (T, N_DCT_COEFFS)

        # PCA appearance：对每帧 ROI flatten 后做 transform
        flat = roi_seq.reshape(T, H * W)  # (T, H*W)
        pca_arr = pca.transform(flat).astype(np.float32)  # (T, n_pca_eff)

        # Hybrid: shape + DCT（文档中的 Hybrid = shape + appearance）
        hybrid_arr = np.concatenate(
            [shape_arr, dct_arr], axis=1
        )  # (T, 2 + N_DCT_COEFFS)

        # 写盘
        np.save(visual_out_dir / f"{base_id}_shape.npy", shape_arr)
        np.save(visual_out_dir / f"{base_id}_dct.npy", dct_arr)
        np.save(visual_out_dir / f"{base_id}_pca.npy", pca_arr)
        np.save(visual_out_dir / f"{base_id}_hybrid.npy", hybrid_arr)

        print(
            f"[INFO] {base_id}: shape={shape_arr.shape}, "
            f"dct={dct_arr.shape}, pca={pca_arr.shape}, hybrid={hybrid_arr.shape}"
        )


def main():
    process_av_features_default()


if __name__ == "__main__":
    main()
