"""PAMAP2 の読み込み・前処理・窓切り。

PAMAP2 Protocol/subject10X.dat の列構成(スペース区切り、54列):
    0      timestamp [s]
    1      activityID (0 = 遷移区間)
    2      heart rate [bpm](約9Hz。それ以外の行は NaN)
    3-19   IMU hand
    20-36  IMU chest
    37-53  IMU ankle
各 IMU の17列:
    +0      温度
    +1..3   加速度 ±16g
    +4..6   加速度 ±6g(高加速度で飽和するため使わない)
    +7..9   ジャイロ
    +10..12 磁気
    +13..16 姿勢(データセット付属の説明で invalid とされている)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# プロトコルで定義された12活動(0 と 任意活動は除外)
ACTIVITIES: dict[int, str] = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    24: "rope_jumping",
}

IMU_OFFSETS = {"hand": 3, "chest": 20, "ankle": 37}
ACC16 = (1, 2, 3)
GYRO = (7, 8, 9)
HR_COL = 2
SAMPLING_HZ = 100


def channel_names(use_hr: bool = True) -> list[str]:
    names = []
    for loc in IMU_OFFSETS:
        names += [f"{loc}_acc_{a}" for a in "xyz"]
        names += [f"{loc}_gyro_{a}" for a in "xyz"]
    if use_hr:
        names.append("heart_rate")
    return names


def _column_indices(use_hr: bool) -> list[int]:
    cols = []
    for base in IMU_OFFSETS.values():
        cols += [base + i for i in ACC16]
        cols += [base + i for i in GYRO]
    if use_hr:
        cols.append(HR_COL)
    return cols


def load_subject(
    path: Path, use_hr: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1被験者分を読み、(信号 [T, C], ラベル [T], 元の行のうち残した行のマスク) を返す。

    - 心拍は約9Hzなので、被験者内で線形補間して100Hzにそろえる
    - IMU の短い欠損(無線の取りこぼし)も線形補間で埋める
    - 補間後も残る NaN を含む行(先頭・末尾など)は捨てる
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 54:
        raise ValueError(f"{path}: 列数が54ではありません({df.shape[1]}列)")

    labels = df[1].to_numpy(dtype=np.int64)
    sig = df[_column_indices(use_hr)].astype(np.float32)

    # 活動0を除く前に補間する(境界の前後の値を使えるようにするため)
    sig = sig.interpolate(method="linear", limit_direction="both", axis=0)
    x = sig.to_numpy(dtype=np.float32)

    keep = np.isin(labels, list(ACTIVITIES)) & ~np.isnan(x).any(axis=1)
    return x[keep], labels[keep], keep


@dataclass
class Windows:
    x: np.ndarray        # [N, C, L]
    y: np.ndarray        # [N] クラス番号(0..K-1)
    subject: np.ndarray  # [N] 被験者ID


def make_windows(
    x: np.ndarray,
    labels: np.ndarray,
    keep: np.ndarray,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """同じ活動が連続する区間の中だけで窓を切る。

    窓の中にラベルが混ざらないように、また活動0を挟んで別の区間を
    つなげないように、元の時系列上で連続している区間ごとに処理する。
    """
    # keep で残った行が、元の時系列で連続していた位置を求める
    orig_idx = np.flatnonzero(keep)
    breaks = (np.diff(orig_idx) != 1) | (np.diff(labels) != 0)
    starts = np.r_[0, np.flatnonzero(breaks) + 1]
    ends = np.r_[starts[1:], len(labels)]

    xs, ys = [], []
    for s, e in zip(starts, ends):
        for w in range(s, e - window + 1, stride):
            xs.append(x[w : w + window].T)
            ys.append(labels[w])
    if not xs:
        return np.empty((0, x.shape[1], window), np.float32), np.empty(0, np.int64)
    return np.stack(xs).astype(np.float32), np.asarray(ys)


def load_dataset(
    data_dir: Path,
    subjects: list[int],
    window: int = 200,
    stride: int = 100,
    use_hr: bool = True,
) -> Windows:
    label_to_class = {a: i for i, a in enumerate(ACTIVITIES)}
    xs, ys, ss = [], [], []
    for sid in subjects:
        path = Path(data_dir) / f"subject{sid}.dat"
        if not path.exists():
            raise FileNotFoundError(path)
        x, labels, keep = load_subject(path, use_hr)
        wx, wy = make_windows(x, labels, keep, window, stride)
        xs.append(wx)
        ys.append(np.array([label_to_class[a] for a in wy], dtype=np.int64))
        ss.append(np.full(len(wy), sid, dtype=np.int64))
        print(f"  subject{sid}: {len(wy):5d} windows")
    return Windows(np.concatenate(xs), np.concatenate(ys), np.concatenate(ss))


class Standardizer:
    """チャネルごとの平均・標準偏差。学習側のデータだけで fit する。"""

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=(0, 2), keepdims=True)
        self.std = x.std(axis=(0, 2), keepdims=True) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)
