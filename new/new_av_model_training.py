"""
Audio-Visual lip-reading CNN training.

goals:
- Use audio MFCC features from audio_feats + hybrid features from visual_feats
- Build an audio-visual early integration model
- Align with the “Audio-visual speech recognition” part of the coursework

Prerequisites:
- feature.py has been executed and generated:
    audio_feats/
        <base_id>_audio_mfcc.npy       shape=(D_a, T_a)
    visual_feats/hybrid/
        <base_id>_hybrid.npy           shape=(T_v, D_v)
- NAMES.txt / name.txt contains all name labels (one name per line)

Notes:
- Audio and visual are paired via <base_id>: <base_id>_audio_mfcc ↔ <base_id>_hybrid
- Resample visual over time to match the number of audio frames, then concatenate along feature dimension
- Normalize the time dimension to a fixed length TARGET_T before feeding into Conv2D
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    InputLayer,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# ===================== Paths and configuration =====================

ROOT_DIR = Path(__file__).resolve().parent

# Audio / visual feature directories (update if your actual directories differ)
AUDIO_FEAT_DIR = ROOT_DIR / "audio_feats" / "mfcc"
VISUAL_HYBRID_DIR = ROOT_DIR / "visual_feats" / "hybrid"

NAMES_FILE = ROOT_DIR / "NAMES.txt"

AV_MODEL_OUT = ROOT_DIR / "models" / "visual_cnn_hybrid.keras"
AV_MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

TARGET_T = 40  # Unified number of time frames; aligned with visual-only for comparability
RANDOM_STATE = 0
TEST_VAL_RATIO = 0.2  # 80/10/10

# ===================== Utility functions =====================


def load_names(names_file: Path) -> List[str]:
    """Load class names from NAMES.txt."""
    names: List[str] = []
    with open(names_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def interp_visual_to_audio_rate(
    visual_seq: np.ndarray,
    audio_time_first: np.ndarray,
) -> np.ndarray:
    """
    Resample visual sequence to match the audio time axis.

    Args:
        visual_seq: shape=(T_v, D_v)
        audio_time_first: shape=(T_a, D_a), only T_a is used as the time reference

    Returns:
        visual_interp: shape=(T_a, D_v)
    """
    if visual_seq.ndim != 2:
        raise ValueError(f"visual_seq must be 2D, got {visual_seq.shape}")

    T_v, D_v = visual_seq.shape
    T_a = audio_time_first.shape[0]

    if T_v == 0 or T_a == 0:
        return np.zeros((T_a, D_v), dtype=np.float32)

    # Normalize both to [0, 1] and perform linear interpolation
    src_t = np.linspace(0.0, 1.0, T_v, dtype=np.float32)
    tgt_t = np.linspace(0.0, 1.0, T_a, dtype=np.float32)

    out = np.zeros((T_a, D_v), dtype=np.float32)
    for d in range(D_v):
        out[:, d] = np.interp(tgt_t, src_t, visual_seq[:, d])

    return out


def pad_or_crop_time(seq_time_first: np.ndarray, target_T: int) -> np.ndarray:
    """
    Align along the time dimension.

    Input:  shape=(T, D)
    Output: shape=(D, target_T)

    The logic is kept consistent with visual_only_model_training for fair comparison.
    """
    if seq_time_first.ndim != 2:
        raise ValueError(f"expect (T, D), got {seq_time_first.shape}")

    T, D = seq_time_first.shape

    if T == target_T:
        return seq_time_first.T

    if T > target_T:
        start = (T - target_T) // 2
        end = start + target_T
        cropped = seq_time_first[start:end, :]
        return cropped.T

    pad_len = target_T - T
    pad = np.zeros((pad_len, D), dtype=seq_time_first.dtype)
    padded = np.vstack([seq_time_first, pad])
    return padded.T


def load_av_dataset(
    audio_feat_dir: Path,
    visual_feat_dir: Path,
    names_file: Path,
    target_T: int,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Build an audio-visual fused dataset from audio and visual feature directories.

    - audio: <base_id>_audio_mfcc.npy, shape: (D_a, T_a)
    - visual hybrid: <base_id>_hybrid.npy, shape: (T_v, D_v)

    Label mapping strategy:
    - label_name = base_id.split("_")[0]
    - label_name must exist in the NAMES.txt list

    Returns:
        X: (N, freq, time, 1)
        y_cat: (N, num_classes)
        names: List[str]
    """
    audio_dir = Path(audio_feat_dir)
    visual_dir = Path(visual_feat_dir)

    names = load_names(names_file)
    label_to_idx = {name: idx for idx, name in enumerate(names)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    audio_files = sorted(audio_dir.glob("*_mfcc.npy"))
    if not audio_files:
        raise RuntimeError(f"No *_mfcc.npy found in {audio_dir}")

    for audio_path in audio_files:
        stem = audio_path.stem  # <base_id>_mfcc
        base_id = stem.replace("_mfcc", "")
        # 先把名字里的字母取出来，再首字母大写，对齐 NAMES.txt 里的写法
        raw_label = "".join(ch for ch in base_id if ch.isalpha())
        label_name = raw_label.capitalize()

        if label_name not in label_to_idx:
            print(f"[WARN] label {label_name} not in NAMES.txt, skip {audio_path}")
            continue

        visual_path = visual_dir / f"{base_id}_hybrid.npy"
        if not visual_path.exists():
            print(f"[WARN] Missing visual hybrid for {base_id}: {visual_path}")
            continue

        # Load audio / visual features
        audio_feat = np.load(audio_path).astype(np.float32)  # (D_a, T_a)
        visual_hybrid_seq = np.load(visual_path).astype(np.float32)  # (T_v, D_v)

        if audio_feat.ndim != 2 or visual_hybrid_seq.ndim != 2:
            print(
                f"[WARN] bad shapes audio={audio_feat.shape}, visual={visual_hybrid_seq.shape}, "
                f"skip {base_id}"
            )
            continue

        # Resample visual to match audio time axis (first move time to axis 0)
        T_a = audio_feat.shape[1]
        audio_time_first = audio_feat.T  # (T_a, D_a)
        visual_interp_time_first = interp_visual_to_audio_rate(
            visual_hybrid_seq, audio_time_first
        )  # (T_a, D_v)

        # Convert back to (D_v, T_a), to concatenate with audio_feat along frequency dimension
        visual_interp = visual_interp_time_first.T  # (D_v, T_a)

        # Early integration: concatenate along the "frequency/feature" dimension → (D_total, T_a)
        fused = np.concatenate([audio_feat, visual_interp], axis=0)

        # Normalize time dimension to target_T (convert to (T, D_total) then pad/crop)
        fused_fixed = pad_or_crop_time(fused.T, target_T=target_T)  # (D_total, target_T)

        X_list.append(fused_fixed)
        y_list.append(label_to_idx[label_name])

    if not X_list:
        raise RuntimeError("No valid audio-visual feature pairs found.")

    X = np.stack(X_list, axis=0)  # (N, D_total, target_T)
    y = np.array(y_list, dtype=np.int64)

    # Add channel dimension
    X = X[..., np.newaxis]  # (N, freq, time, 1)

    num_classes = len(names)
    y_cat = to_categorical(y, num_classes=num_classes)

    print(f"[INFO] AV dataset loaded: X={X.shape}, y={y_cat.shape}")
    return X, y_cat, names


# ===================== Model definition (CNN) =====================


def create_av_cnn(input_freq: int, input_time: int, num_classes: int) -> Sequential:
    """
    AV CNN architecture: aligned with the visual-only network for fair comparison.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(input_freq, input_time, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        optimizer="adam",
    )
    model.summary()
    return model


# ===================== Main pipeline: split data + train + test =====================


def main() -> None:
    print(f"[INFO] AUDIO_FEAT_DIR   = {AUDIO_FEAT_DIR}")
    print(f"[INFO] VISUAL_HYBRID_DIR= {VISUAL_HYBRID_DIR}")
    print(f"[INFO] NAMES_FILE       = {NAMES_FILE}")
    print(f"[INFO] TARGET_T         = {TARGET_T}")

    X, y, names = load_av_dataset(
        audio_feat_dir=AUDIO_FEAT_DIR,
        visual_feat_dir=VISUAL_HYBRID_DIR,
        names_file=NAMES_FILE,
        target_T=TARGET_T,
    )

    num_classes = y.shape[1]
    _, freq, time, _ = X.shape

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=TEST_VAL_RATIO, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_tmp
    )

    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = create_av_cnn(input_freq=freq, input_time=time, num_classes=num_classes)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(AV_MODEL_OUT),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=12,
        callbacks=callbacks,
        verbose=1,
    )

    print("[INFO] Evaluate on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] AV test accuracy = {test_acc:.4f}, loss = {test_loss:.4f}")


if __name__ == "__main__":
    main()
