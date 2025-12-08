"""
Visual-only lip-reading CNN training.

Prerequisites:
- feature.py has been executed and generated:
    visual_feats/
        shape/   <base_id>_shape.npy   shape=(T_v, 2)
        dct/     <base_id>_dct.npy     shape=(T_v, N_DCT_COEFFS)
        pca/     <base_id>_pca.npy     shape=(T_v, n_pca_eff)
        hybrid/  <base_id>_hybrid.npy  shape=(T_v, 2+N_DCT_COEFFS)
- NAMES.txt contains all name labels (one name per line)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

try:
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
except:
    import tensorflow as tf

    EarlyStopping = tf.keras.callbacks.EarlyStopping
    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
    Conv2D = tf.keras.layers.Conv2D
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    Flatten = tf.keras.layers.Flatten
    InputLayer = tf.keras.layers.InputLayer
    MaxPooling2D = tf.keras.layers.MaxPooling2D
    Sequential = tf.keras.models.Sequential
    to_categorical = tf.keras.utils.to_categorical
# ===================== Paths and configuration =====================

ROOT_DIR = Path(__file__).resolve().parent

# Root directory of visual features from feature.py
VISUAL_FEAT_ROOT = ROOT_DIR / "visual_feats"

NAMES_FILE = ROOT_DIR / "NAMES.txt"

MODEL_OUT = ROOT_DIR / "models" / "visual_cnn_model.h5"
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

# Which visual feature to use: "shape" / "dct" / "pca" / "hybrid"
FEATURE_KIND = "hybrid"

# Unified time length (number of frames), similar to using a fixed window on audio
TARGET_T = 40  # Tunable hyperparameter

RANDOM_STATE = 0
TEST_VAL_RATIO = 0.2  # Take 20% for val+test, then split into 10%/10%

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


def pad_or_crop_time(seq_time_first: np.ndarray, target_T: int) -> np.ndarray:
    """
    Align along the time dimension.

    Input:  shape=(T, D)
    Output: shape=(D, target_T)

    - If T > target_T: center crop
    - If T < target_T: zero pad at the end
    """
    if seq_time_first.ndim != 2:
        raise ValueError(f"expect (T, D), got {seq_time_first.shape}")

    T, D = seq_time_first.shape

    if T == target_T:
        return seq_time_first.T  # (D, T)

    if T > target_T:
        start = (T - target_T) // 2
        end = start + target_T
        cropped = seq_time_first[start:end, :]  # (target_T, D)
        return cropped.T

    # T < target_T → pad zeros at end
    pad_len = target_T - T
    pad = np.zeros((pad_len, D), dtype=seq_time_first.dtype)
    padded = np.vstack([seq_time_first, pad])  # (target_T, D)
    return padded.T  # (D, target_T)


def get_visual_subdir_and_suffix(feature_kind: str) -> Tuple[str, str]:
    """
    Map FEATURE_KIND to (subdirectory name, file suffix).
    """
    mapping = {
        "shape": ("shape", "_shape.npy"),
        "dct": ("dct", "_dct.npy"),
        "pca": ("pca", "_pca.npy"),
        "hybrid": ("hybrid", "_hybrid.npy"),
    }
    if feature_kind not in mapping:
        raise ValueError(f"Unsupported FEATURE_KIND={feature_kind}")
    return mapping[feature_kind]


def load_visual_dataset(
    visual_feat_root: Path,
    names_file: Path,
    feature_kind: str,
    target_T: int,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Build visual dataset from visual_feats/* subdirectories.

    Returns:
        X: (N, freq, time, 1)
        y_cat: (N, num_classes) one-hot
        names: List[str]
    """
    names = load_names(names_file)
    label_to_idx = {name: idx for idx, name in enumerate(names)}

    subdir, suffix = get_visual_subdir_and_suffix(feature_kind)
    feat_dir = visual_feat_root / subdir

    if not feat_dir.exists():
        raise RuntimeError(f"Visual feature dir not found: {feat_dir}")

    feat_files = sorted(feat_dir.glob(f"*{suffix}"))
    if not feat_files:
        raise RuntimeError(f"No visual feature files found in {feat_dir} with suffix {suffix}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for feat_path in feat_files:
        base_id = feat_path.stem.replace(suffix.replace(".npy", ""), "")
        # More robust approach: derive label directly from file name
        label_name = base_id.split("_")[0]

        if label_name not in label_to_idx:
            print(f"[WARN] label {label_name} not in NAMES.txt, skip {feat_path}")
            continue

        seq = np.load(feat_path).astype(np.float32)  # (T, D)
        if seq.ndim != 2:
            print(f"[WARN] bad visual feature shape {seq.shape}, skip {feat_path}")
            continue

        feat_fixed = pad_or_crop_time(seq_time_first=seq, target_T=target_T)  # (D, T)
        X_list.append(feat_fixed)
        y_list.append(label_to_idx[label_name])

    if not X_list:
        raise RuntimeError("No valid visual feature files found.")

    X = np.stack(X_list, axis=0)  # (N, D, T)
    y = np.array(y_list, dtype=np.int64)

    # Add channel dimension for Conv2D: (N, freq, time, 1)
    X = X[..., np.newaxis]

    num_classes = len(names)
    y_cat = to_categorical(y, num_classes=num_classes)

    print(f"[INFO] Visual dataset loaded: X={X.shape}, y={y_cat.shape}")
    return X, y_cat, names


# ===================== Model definition (CNN) =====================


def create_visual_cnn(input_freq: int, input_time: int, num_classes: int) -> Sequential:
    """
    Simple CNN aligned with Lab 4, with configurable input dimensions and number of classes.
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
    print(f"[INFO] VISUAL_FEAT_ROOT = {VISUAL_FEAT_ROOT}")
    print(f"[INFO] NAMES_FILE       = {NAMES_FILE}")
    print(f"[INFO] FEATURE_KIND     = {FEATURE_KIND}")
    print(f"[INFO] TARGET_T         = {TARGET_T}")

    X, y, names = load_visual_dataset(
        visual_feat_root=VISUAL_FEAT_ROOT,
        names_file=NAMES_FILE,
        feature_kind=FEATURE_KIND,
        target_T=TARGET_T,
    )

    num_classes = y.shape[1]
    _, freq, time, _ = X.shape

    # 80% train, 10% val, 10% test (two-stage split)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=TEST_VAL_RATIO, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_tmp
    )

    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = create_visual_cnn(input_freq=freq, input_time=time, num_classes=num_classes)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_OUT),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    print("[INFO] Evaluate on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(
        f"[RESULT] Visual-only test accuracy = {100.0*test_acc:.2f}%, loss = {test_loss:.2f}"
    )


if __name__ == "__main__":
    main()
print("done")
