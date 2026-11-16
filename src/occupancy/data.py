"""Data loading, leakage-safe chronological split, scaling and windowing.

Alur:
1. load_raw()            -> DataFrame + kolom 'timestamp'
2. add_sessions()        -> memotong deret menjadi 'session' bila jeda > max_gap_seconds
3. chronological_split() -> setiap session dibagi 60/20/20 berdasarkan waktu
                            (bagian awal = train, tengah = val, akhir = test)
4. StandardScaler1D      -> di-fit HANYA pada train
5. make_windows()        -> jendela [t-T+1 .. t] tidak pernah melintasi session/split
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# OUT_DIR bisa dialihkan (mis. untuk uji) lewat env OCC_OUT; default = root repo
OUT_DIR = Path(os.environ.get("OCC_OUT", ROOT))


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("OCC_CONFIG") or ROOT / "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    path = Path(path or os.environ.get("OCC_DATA") or ROOT / cfg["dataset_file"])
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                                     format="mixed", dayfirst=False)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df


def add_sessions(df: pd.DataFrame, max_gap_seconds: int = 120) -> pd.DataFrame:
    gap = df["timestamp"].diff().dt.total_seconds().fillna(0)
    df = df.copy()
    df["session"] = (gap > max_gap_seconds).cumsum().astype(int)
    return df


def chronological_split(df: pd.DataFrame, train=0.6, val=0.2, rest: str = "test") -> pd.DataFrame:
    """Split per session secara kronologis. Bagian awal setiap session menjadi train,
    bagian berikutnya validation, sisanya `rest` ("test" untuk split utama, "unused" untuk
    fold CV). Tidak ada pengacakan, sehingga data masa depan tidak pernah dipakai melatih."""
    df = df.copy()
    df["split"] = "train"
    for _, idx in df.groupby("session").groups.items():
        idx = np.asarray(sorted(idx))
        n = len(idx)
        n_tr = int(round(n * train))
        n_end = int(round(n * (train + val)))
        df.loc[idx[n_tr:n_end], "split"] = "val"
        df.loc[idx[n_end:], "split"] = rest
    return df


@dataclass
class StandardScaler1D:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler1D":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def to_dict(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler1D":
        return cls(np.asarray(d["mean"], dtype=float), np.asarray(d["std"], dtype=float))


def make_windows(X: np.ndarray, y: np.ndarray | None, groups: np.ndarray, T: int):
    """Membuat jendela sekuens panjang T. Label = kelas pada langkah terakhir.
    groups: id gabungan (session, split) sehingga jendela tidak melintasi batas."""
    Xw, yw, end_idx = [], [], []
    start = 0
    n = len(X)
    while start < n:
        end = start
        while end + 1 < n and groups[end + 1] == groups[start]:
            end += 1
        seg = np.arange(start, end + 1)
        if len(seg) >= T:
            for t in range(T - 1, len(seg)):
                Xw.append(X[seg[t - T + 1:t + 1]])
                if y is not None:
                    yw.append(y[seg[t]])
                end_idx.append(seg[t])
        start = end + 1
    Xw = np.asarray(Xw, dtype=np.float64).reshape(-1, T, X.shape[1])
    yw = np.asarray(yw, dtype=np.int64) if y is not None else None
    return Xw, yw, np.asarray(end_idx, dtype=np.int64)


class DataBundle:
    """Menyimpan data ter-split & ter-skala; menyediakan window untuk subset fitur/T."""

    def __init__(self, cfg: dict | None = None, path: str | Path | None = None, fold: tuple | None = None):
        """fold=None -> split utama 60/20/20 (train/val/test).
        fold=(a, b)  -> fold CV forward-chaining: train = [0, a), val = [a, b) di setiap sesi,
                        sisanya 'unused' (termasuk seluruh periode test)."""
        self.cfg = cfg or load_config()
        df = load_raw(path)
        df = add_sessions(df, self.cfg["max_gap_seconds"])
        if fold is None:
            df = chronological_split(df, self.cfg["split"]["train"], self.cfg["split"]["val"])
        else:
            a, b = fold
            assert b <= self.cfg["split"]["train"] + self.cfg["split"]["val"] + 1e-9, "fold menyentuh test"
            df = chronological_split(df, a, b - a, rest="unused")
        self.fold = fold
        self.df = df
        self.features = list(self.cfg["sensor_features"])
        self.target = self.cfg["target"]
        tr = df["split"] == "train"
        self.scaler = StandardScaler1D().fit(df.loc[tr, self.features].to_numpy(float))
        self.Xs = self.scaler.transform(df[self.features].to_numpy(float))
        self.y = df[self.target].to_numpy(int)
        # group id = session*10 + split code -> jendela tidak melintasi batas split
        code = df["split"].map({"train": 0, "val": 1, "test": 2, "unused": 3}).to_numpy()
        self.groups = df["session"].to_numpy() * 10 + code
        self.split_code = code
        self._cache: dict = {}

    def windows(self, split: str, T: int, feat_idx=None):
        key = (split, T)
        if key not in self._cache:
            c = {"train": 0, "val": 1, "test": 2}[split]
            m = self.split_code == c
            idx = np.where(m)[0]
            Xw, yw, end = make_windows(self.Xs[idx], self.y[idx], self.groups[idx], T)
            self._cache[key] = (Xw, yw, idx[end])
        Xw, yw, end = self._cache[key]
        if feat_idx is not None:
            Xw = Xw[:, :, feat_idx]
        return Xw, yw, end


def cv_folds(cfg: dict | None = None, path=None) -> list:
    """Bundle untuk setiap fold CV forward-chaining (dipakai GA & PSO)."""
    cfg = cfg or load_config()
    return [DataBundle(cfg, path, fold=tuple(f)) for f in cfg["cv_folds"]]


def class_weights(y: np.ndarray, n_classes: int = 4) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts)
    return w / w.mean()
