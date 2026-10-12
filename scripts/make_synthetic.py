"""PAMAP2 と同じ形式(54列、スペース区切り)の合成データを作る。

本物のデータを入れる前に、パイプラインが最後まで動くかを確かめるためのもの。
信号は活動ごとに周波数と振幅を変えた正弦波+ノイズで、物理的な意味はない。

    python scripts/make_synthetic.py --out data/synthetic --seconds 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
HZ = 100


def make_subject(sid: int, seconds: int, rng: np.random.Generator) -> np.ndarray:
    gain = rng.uniform(0.8, 1.2)       # 被験者ごとの個人差
    offset = rng.normal(0, 0.3)
    rows = []
    t0 = 0.0
    order = list(ACTIVITY_IDS)
    if sid % 3 == 0:
        order.remove(24)               # 縄跳びをしない被験者を混ぜる(実データと同じ状況)

    sequence = [0]
    for a in order:
        sequence += [a, 0]             # 活動の間に遷移区間(活動0)を挟む
    for act in sequence:
        n = (5 if act == 0 else seconds) * HZ
        t = t0 + np.arange(n) / HZ
        k = ACTIVITY_IDS.index(act) + 1 if act else 0
        freq = 0.3 + 0.25 * k
        amp = gain * (0.2 + 0.15 * k)

        block = np.full((n, 54), np.nan)
        block[:, 0] = t
        block[:, 1] = act
        hr = 70 + 4 * k + rng.normal(0, 2, n)
        block[::11, 2] = hr[::11]      # 心拍は約9Hz、他の行はNaN
        for base in (3, 20, 37):
            block[:, base] = 33 + rng.normal(0, 0.1, n)   # 温度
            for j in range(1, 13):
                phase = rng.uniform(0, 2 * np.pi)
                block[:, base + j] = (
                    amp * np.sin(2 * np.pi * freq * (1 + 0.1 * j) * t + phase)
                    + offset + rng.normal(0, 0.3, n)
                )
            block[:, base + 13 : base + 17] = 0.0          # 姿勢(未使用)
        drop = rng.random(n) < 0.01                        # 無線の取りこぼし
        block[drop, 3:] = np.nan
        rows.append(block)
        t0 = t[-1] + 1 / HZ
    return np.concatenate(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--seconds", type=int, default=20, help="1活動あたりの秒数")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for sid in range(101, 109):
        arr = make_subject(sid, args.seconds, rng)
        np.savetxt(args.out / f"subject{sid}.dat", arr, fmt="%.6g", delimiter=" ")
        print(f"subject{sid}.dat: {len(arr)} rows")


if __name__ == "__main__":
    main()
