"""
Audio-Visual model training.

对应“模型训练”阶段，基于：
- Lab 4 Speech recognition.pdf / final_modelling.py 中的 CNN 语音识别网络结构
- visual-speech-features-lab.pdf / visual-speech-features-notes.pdf 中的
  * visual feature 插值（将视觉特征对齐到音频帧率，用作 early integration）
  * audio-visual 早期融合（feature concatenation）

输入特征由 av_feature_extraction.py 生成：
- audio: <base_id>_audio_mfcc.npy，shape: (D_a, T_a)
- visual: <base_id>_hybrid.npy，shape: (T_v, D_v)
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPooling2D
from tensorflow.keras.models import Sequential


# ===================== 路径默认值：按 av-lip-reader 结构 =====================

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO_FEAT_DIR = ROOT_DIR / "audio_feats"
DEFAULT_VISUAL_FEAT_DIR = ROOT_DIR / "visual_feats"
DEFAULT_NAMES_FILE = ROOT_DIR / "NAMES.txt"
DEFAULT_MODEL_OUT = ROOT_DIR / "models" / "av_cnn_model.h5"


# ===================== 时间对齐：视觉特征插值到音频帧率 =====================


def interp_visual_to_audio_rate(visual_seq, audio_seq_time_first):
    """
    将视觉特征（低帧率）插值到音频特征的时间轴。

    visual_seq: shape (T_v, D_v)  —— T_v 为视觉帧数
    audio_seq_time_first: shape (T_a, D_a) —— T_a 为音频帧数（时间为第一维）

    返回：
        visual_interp: shape (T_a, D_v)

    逻辑对应 visual-speech-features-lab.pdf / visual-speech-features-notes.pdf 中的：
    - 将视觉帧索引映射到 [0,1]
    - 将音频帧索引映射到 [0,1]
    - 对每个维度做 1D 线性插值
    """
    visual_seq = np.asarray(visual_seq, dtype=np.float32)
    audio_seq_time_first = np.asarray(audio_seq_time_first, dtype=np.float32)

    T_v, D_v = visual_seq.shape
    T_a = audio_seq_time_first.shape[0]

    if T_v == 1:
        # 只有一个视觉帧时，简单 repeat
        return np.repeat(visual_seq, T_a, axis=0)

    t_v = np.linspace(0.0, 1.0, T_v)
    t_a = np.linspace(0.0, 1.0, T_a)

    visual_interp = np.empty((T_a, D_v), dtype=np.float32)
    for d in range(D_v):
        visual_interp[:, d] = np.interp(t_a, t_v, visual_seq[:, d])

    return visual_interp


# ===================== 数据加载：从 .npy 构建 AV 特征和标签 =====================


def load_names(names_file: Path):
    """
    从 NAMES.txt 读取类别名称列表。
    """
    names: list[str] = []
    with open(names_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def load_av_dataset(audio_feat_dir: Path, visual_feat_dir: Path, names_file: Path):
    """
    从音频和视觉特征目录构建音视频融合数据集。

    - audio: <base_id>_audio_mfcc.npy，shape: (D_a, T_a)
    - visual hybrid: <base_id>_hybrid.npy，shape: (T_v, D_v)

    label 映射策略：
    - label_name = base_id.split("_")[0]
    - label_name 必须出现在 NAMES.txt 列表中
    """
    audio_dir = Path(audio_feat_dir)
    visual_dir = Path(visual_feat_dir)

    names = load_names(names_file)
    label_to_idx = {name: idx for idx, name in enumerate(names)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    audio_files = sorted(audio_dir.glob("*_audio_mfcc.npy"))
    if not audio_files:
        raise RuntimeError(f"No *_audio_mfcc.npy found in {audio_dir}")

    for audio_path in audio_files:
        stem = audio_path.stem  # <base_id>_audio_mfcc
        base_id = stem.replace("_audio_mfcc", "")
        label_name = base_id.split("_")[0]

        if label_name not in label_to_idx:
            print(f"[WARN] label {label_name} not in NAMES.txt, skip {audio_path}")
            continue

        visual_path = visual_dir / f"{base_id}_hybrid.npy"
        if not visual_path.exists():
            print(f"[WARN] Missing visual hybrid for {base_id}: {visual_path}")
            continue

        # 加载 audio / visual 特征
        audio_feat = np.load(audio_path).astype(np.float32)           # (D_a, T_a)
        visual_hybrid_seq = np.load(visual_path).astype(np.float32)   # (T_v, D_v)

        # 将 visual 重采样到 audio 时间轴（时间维度先转到第 0 维）
        T_a = audio_feat.shape[1]
        audio_time_first = audio_feat.T  # (T_a, D_a)
        visual_interp_time_first = interp_visual_to_audio_rate(
            visual_hybrid_seq, audio_time_first
        )  # (T_a, D_v)

        # 再转回 (D_v, T_a)，方便与 audio_feat 在频率维拼接
        visual_interp = visual_interp_time_first.T  # (D_v, T_a)

        # Early integration：在“频率/特征维”上拼接
        fused = np.concatenate([audio_feat, visual_interp], axis=0)  # (D_total, T_a)

        X_list.append(fused)
        y_list.append(label_to_idx[label_name])

    if not X_list:
        raise RuntimeError("No valid audio-visual feature pairs found.")

    # 统一时间长度（T 维），并加入通道维度，形状与 final_modelling.py 中 CNN 输入一致：(N, D, T, 1)
    max_T = max(arr.shape[1] for arr in X_list)
    D_total = X_list[0].shape[0]
    N = len(X_list)

    X = np.zeros((N, D_total, max_T), dtype=np.float32)
    for i, arr in enumerate(X_list):
        T = arr.shape[1]
        X[i, :, :T] = arr  # 后面是零填充

    # 全局幅度归一化（参考 final_modelling.py）
    max_abs = np.max(np.abs(X))
    if max_abs > 0:
        X = X / max_abs

    # 加通道维（CNN 期望 [freq, time, channel]）
    X = X[..., np.newaxis]  # (N, D_total, max_T, 1)

    y = np.array(y_list, dtype=np.int64)

    return X, y, names


# ===================== 模型结构：基于 Lab 4 CNN =====================


def create_av_cnn_model(freq_bins: int, time_steps: int, num_classes: int) -> Sequential:
    """
    构建与 Lab 4 Speech recognition / final_modelling.py 同风格的 2D CNN 模型，
    但输入通道改为 audio+visual 融合特征。

    input_shape: (freq_bins, time_steps, 1)
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(freq_bins, time_steps, 1)))

    # 与 final_modelling.py 类似的卷积 + 池化结构
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ===================== 训练入口 =====================


def train_av_model(
    audio_feat_dir: Path = DEFAULT_AUDIO_FEAT_DIR,
    visual_feat_dir: Path = DEFAULT_VISUAL_FEAT_DIR,
    names_file: Path = DEFAULT_NAMES_FILE,
    model_out_path: Path = DEFAULT_MODEL_OUT,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 32,
    epochs: int = 50,
):
    """
    从 audio / visual 特征目录中加载 AV 数据集，拆分 train/val，并训练 CNN 模型。
    """
    X, y, names = load_av_dataset(
        audio_feat_dir=audio_feat_dir,
        visual_feat_dir=visual_feat_dir,
        names_file=names_file,
    )

    freq_bins = X.shape[1]
    time_steps = X.shape[2]
    num_classes = len(names)

    print(f"[INFO] Dataset: N={X.shape[0]}, freq_bins={freq_bins}, "
          f"time_steps={time_steps}, classes={num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = create_av_cnn_model(freq_bins, time_steps, num_classes)

    # 确保模型输出目录存在
    model_out_path = Path(model_out_path)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(model_out_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
    )

    print("[INFO] Training finished. Best model saved to", model_out_path)
    return model, history, names


def main():
    print("[INFO] Using default paths for AV model training")
    print(f"[INFO] AUDIO_FEAT_DIR  = {DEFAULT_AUDIO_FEAT_DIR}")
    print(f"[INFO] VISUAL_FEAT_DIR = {DEFAULT_VISUAL_FEAT_DIR}")
    print(f"[INFO] NAMES_FILE      = {DEFAULT_NAMES_FILE}")
    print(f"[INFO] MODEL_OUT       = {DEFAULT_MODEL_OUT}")

    train_av_model(
        audio_feat_dir=DEFAULT_AUDIO_FEAT_DIR,
        visual_feat_dir=DEFAULT_VISUAL_FEAT_DIR,
        names_file=DEFAULT_NAMES_FILE,
        model_out_path=DEFAULT_MODEL_OUT,
        test_size=0.2,
        batch_size=32,
        epochs=50,
        random_state=42,
    )


if __name__ == "__main__":
    main()
