"""学習と評価。

    python -m src.train --data-dir data/PAMAP2_Dataset/Protocol --model cnn --split loso
    python -m src.train --data-dir data/PAMAP2_Dataset/Protocol --model cnn --split random

--split loso   被験者を1人ずつテストに回す(leave-one-subject-out)。
               残りのうち1人を検証用(早期終了の判断)にし、残りで学習する。
--split random 窓をランダムに分割する。重なりのある隣の窓が学習側と評価側に
               分かれて入るので、精度は高めに出る。LOSOとの差を見るための比較用。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import ACTIVITIES, Standardizer, channel_names, load_dataset
from .models import build_model

CLASS_NAMES = list(ACTIVITIES.values())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    for (xb,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch):
        preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # テスト被験者に存在しない活動(縄跳びをしていない被験者など)は平均に入れない
    return f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true), zero_division=0)


def train_one(
    args: argparse.Namespace,
    xtr: np.ndarray, ytr: np.ndarray,
    xva: np.ndarray, yva: np.ndarray,
    device: str,
) -> nn.Module:
    model = build_model(args.model, xtr.shape[1], len(CLASS_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # クラス不均衡への対処: 出現頻度の逆数で損失に重みを付ける
    counts = np.bincount(ytr, minlength=len(CLASS_NAMES)).astype(np.float32)
    w = np.where(counts > 0, counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1)), 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    best_f1, best_state, wait = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader(xtr, ytr, args.batch, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)

        val_f1 = macro_f1(yva, predict(model, xva, device))
        if args.verbose:
            print(f"    epoch {epoch:3d}  loss {total / len(ytr):.4f}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    return model


def run_loso(args, data, device) -> dict:
    subjects = sorted(set(data.subject.tolist()))
    folds, all_true, all_pred = [], [], []
    for i, test_sid in enumerate(subjects):
        others = [s for s in subjects if s != test_sid]
        val_sid = others[i % len(others)]
        tr = np.isin(data.subject, [s for s in others if s != val_sid])
        va = data.subject == val_sid
        te = data.subject == test_sid

        scaler = Standardizer().fit(data.x[tr])  # 学習側だけで fit
        xtr, xva, xte = (scaler.transform(data.x[m]) for m in (tr, va, te))

        set_seed(args.seed)
        t0 = time.time()
        model = train_one(args, xtr, data.y[tr], xva, data.y[va], device)
        pred = predict(model, xte, device)
        yte = data.y[te]

        fold = {
            "test_subject": int(test_sid),
            "val_subject": int(val_sid),
            "n_test": int(te.sum()),
            "accuracy": float(accuracy_score(yte, pred)),
            "macro_f1": float(macro_f1(yte, pred)),
            "seconds": round(time.time() - t0, 1),
        }
        folds.append(fold)
        all_true.append(yte)
        all_pred.append(pred)
        print(f"  test subject{test_sid} (val {val_sid}): "
              f"acc {fold['accuracy']:.3f}  macro-F1 {fold['macro_f1']:.3f}")

    return summarize(folds, np.concatenate(all_true), np.concatenate(all_pred))


def run_random(args, data, device) -> dict:
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(data.y))
    n = len(idx)
    tr, va, te = idx[: int(n * 0.8)], idx[int(n * 0.8) : int(n * 0.9)], idx[int(n * 0.9) :]

    scaler = Standardizer().fit(data.x[tr])
    xtr, xva, xte = (scaler.transform(data.x[m]) for m in (tr, va, te))

    set_seed(args.seed)
    model = train_one(args, xtr, data.y[tr], xva, data.y[va], device)
    pred = predict(model, xte, device)
    yte = data.y[te]
    fold = {
        "test_subject": None,
        "n_test": int(len(te)),
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(macro_f1(yte, pred)),
    }
    print(f"  random split: acc {fold['accuracy']:.3f}  macro-F1 {fold['macro_f1']:.3f}")
    return summarize([fold], yte, pred)


def summarize(folds: list[dict], y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = np.array([f["accuracy"] for f in folds])
    f1 = np.array([f["macro_f1"] for f in folds])
    return {
        "folds": folds,
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std()),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(CLASS_NAMES))
        ).tolist(),
    }


def save_confusion(cm: list[list[int]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=60, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+",
                   default=[101, 102, 103, 104, 105, 106, 107, 108])
    p.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    p.add_argument("--split", choices=["loso", "random"], default="loso")
    p.add_argument("--window", type=int, default=200, help="窓の長さ(サンプル数、100Hz)")
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--no-hr", action="store_true", help="心拍を入力に使わない")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_hr = not args.no_hr
    print(f"device: {device}")
    print(f"channels ({len(channel_names(use_hr))}): {', '.join(channel_names(use_hr))}")

    print("loading ...")
    data = load_dataset(args.data_dir, args.subjects, args.window, args.stride, use_hr)
    print(f"total windows: {len(data.y)}, shape per window: {data.x.shape[1:]}")

    result = run_loso(args, data, device) if args.split == "loso" else run_random(args, data, device)
    result["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.split}" + ("_nohr" if args.no_hr else "")
    (args.out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    save_confusion(result["confusion_matrix"], args.out_dir / f"{tag}_confusion.png",
                   f"{args.model} / {args.split}")

    print(f"\n{tag}: accuracy {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}, "
          f"macro-F1 {result['macro_f1_mean']:.3f} ± {result['macro_f1_std']:.3f}")
    print(f"saved: {args.out_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
