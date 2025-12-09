#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature QA & Smoke Test for visual hybrid features (支持 42 或 122 维).
- 逐文件体检：NaN/Inf、长度T、时序平滑度 r、热力图
- 全库指标：T 分布、r 分布
- 线性探针：LogReg on [mean,std,Δmean,Δstd]，自适应折数
输出到 qa_outputs/
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# sklearn 可选；缺失则跳过探针
try:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

ROOT = Path(".")
HYB_DIR = ROOT / "visual_feats" / "hybrid"
OUT_DIR = ROOT / "qa_outputs"

def norm_label(base_id: str) -> str:
    import re
    t = re.split(r"[_\\-]", base_id)[0]
    t = re.sub(r"\\d+$", "", t)
    return t.lower()

def zscore_per_dim(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + 1e-6
    return (x - m) / s

def temporal_smooth_r(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 1.0
    dx = np.linalg.norm(x[1:] - x[:-1], axis=1)
    xn = np.linalg.norm(x[1:], axis=1) + 1e-6
    return float(np.median(dx / xn))

def save_heatmap(base_id: str, X: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    Z = zscore_per_dim(X)  # 仅可视化
    plt.figure(figsize=(6, 3))
    plt.imshow(Z.T, aspect="auto")
    plt.title(base_id, fontsize=10)
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / f"{base_id}.png", dpi=160)
    plt.close()

def build_probe_feature(X: np.ndarray) -> np.ndarray:
    """mean + std + Δmean + Δstd"""
    if X.shape[0] < 3:
        return None
    dX = np.diff(X, axis=0)
    feat = np.concatenate([X.mean(0), X.std(0), dX.mean(0), dX.std(0)], axis=0).astype(np.float32)
    return feat

def linear_probe_mean_std_delta(df: pd.DataFrame) -> str:
    if not SKLEARN_OK:
        return "sklearn 不可用：跳过线性探针。"

    # 读 NAMES.txt（若存在则仅评估有效类）
    names_path = ROOT / "NAMES.txt"
    valid = {n.strip().lower() for n in names_path.read_text(encoding="utf-8").splitlines() if n.strip()} \
            if names_path.exists() else None

    Xs, ys = [], []
    for _, row in df.iterrows():
        lab = row["label"]
        if valid is not None and lab not in valid:
            continue
        X = np.load(row["path"])
        feat = build_probe_feature(X)
        if feat is None:
            continue
        Xs.append(feat); ys.append(lab)

    if len(Xs) < 50 or len(set(ys)) < 2:
        return "数据不足：样本<50或类别<2，跳过线性探针。"

    X = np.vstack(Xs)
    labels = sorted(set(ys))
    y = np.array([labels.index(t) for t in ys], dtype=np.int64)

    counts = Counter(ys)
    print("Class counts (after filtering):", dict(counts))

    min_per_class = min(counts.values())
    # 自适应折数；不足则采用 ShuffleSplit
    if min_per_class >= 5:
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        accs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=400, n_jobs=1)
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[te])
            accs.append(accuracy_score(y[te], p))
        return f"LINEAR_PROBE k={k}: acc = {np.mean(accs):.4f} ± {np.std(accs):.4f} (C={len(labels)}, N={len(X)})"
    elif min_per_class >= 2:
        # 用 5 次分层随机划分，避免 n_splits 受限
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        accs = []
        for tr, te in sss.split(X, y):
            clf = LogisticRegression(max_iter=400, n_jobs=1)
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[te])
            accs.append(accuracy_score(y[te], p))
        return f"LINEAR_PROBE StratifiedShuffleSplit: acc = {np.mean(accs):.4f} ± {np.std(accs):.4f} (min_class={min_per_class}, C={len(labels)}, N={len(X)})"
    else:
        return f"数据不足：最小类 {min_per_class} 条，跳过线性探针。"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_heatmaps", type=int, default=24, help="随机导出热力图数量")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "heatmaps").mkdir(exist_ok=True)

    files = sorted(HYB_DIR.glob("*_hybrid.npy"))
    if not files:
        print(f"[FAIL] 未找到特征文件：{HYB_DIR}")
        return

    rows = []
    for fp in files:
        base = fp.stem.replace("_hybrid", "")
        X = np.load(fp)  # (T, D)
        rows.append({
            "base_id": base,
            "label": norm_label(base),
            "T": X.shape[0],
            "D": X.shape[1],
            "nan": int(np.isnan(X).any()),
            "inf": int(np.isinf(X).any()),
            "r": temporal_smooth_r(X),
            "path": str(fp),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "inventory.csv", index=False)

    # 汇总 KPI
    print("=== SUMMARY ===")
    print(f"files: {len(df)}")
    print(f"D set: {sorted(df['D'].unique().tolist())}")
    print(f"NaN rate: {(df['nan']==1).mean():.4f}, Inf rate: {(df['inf']==1).mean():.4f}")
    print(f"T: min={df['T'].min()}, p25={df['T'].quantile(0.25):.1f}, "
          f"p50={df['T'].median():.1f}, p75={df['T'].quantile(0.75):.1f}, max={df['T'].max()}")
    print(f"r: p25={df['r'].quantile(0.25):.3f}, p50={df['r'].median():.3f}, p75={df['r'].quantile(0.75):.3f}")

    # 直方图
    plt.figure(); df["T"].hist(bins=30); plt.title("T distribution"); plt.xlabel("T"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(OUT_DIR / "T_hist.png", dpi=160); plt.close()

    plt.figure(); df["r"].hist(bins=30); plt.title("temporal smoothness r"); plt.xlabel("r")
    plt.tight_layout(); plt.savefig(OUT_DIR / "r_hist.png", dpi=160); plt.close()

    # 随机热力图
    samp = df.sample(min(args.max_heatmaps, len(df)), random_state=0)
    for _, row in samp.iterrows():
        X = np.load(row["path"])
        save_heatmap(row["base_id"], X, OUT_DIR / "heatmaps")

    # 线性探针
    probe_msg = linear_probe_mean_std_delta(df)
    print("\n" + probe_msg)
    (OUT_DIR / "probe.txt").write_text(probe_msg, encoding="utf-8")

    print(f"\n[OK] 报告目录：{OUT_DIR.resolve()}")
    print("  - inventory.csv, T_hist.png, r_hist.png")
    print("  - heatmaps/*.png")
    print("  - probe.txt")

if __name__ == "__main__":
    main()
