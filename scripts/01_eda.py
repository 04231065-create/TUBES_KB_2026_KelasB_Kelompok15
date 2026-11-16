"""W12 - EDA: statistik, missing values, distribusi target, korelasi, ringkasan split."""
from _common import ART, Logger, save_json  # noqa: F401

import numpy as np

from occupancy.data import DataBundle, load_config

log = Logger("01_eda")
cfg = load_config()
b = DataBundle(cfg)
df, feats, tgt = b.df, b.features, b.target

eda = {
    "n_rows": int(len(df)),
    "n_columns_raw": int(len(df.columns) - 3),  # tanpa timestamp/session/split
    "n_sensor_features": len(feats),
    "time_start": str(df["timestamp"].min()),
    "time_end": str(df["timestamp"].max()),
    "dates": sorted(df["Date"].astype(str).unique().tolist()),
    "sampling_median_s": float(df["timestamp"].diff().dt.total_seconds().median()),
    "missing_values": {c: int(v) for c, v in df[feats + [tgt]].isna().sum().items()},
    "duplicated_rows": int(df.duplicated(subset=["Date", "Time"]).sum()),
    "class_counts": {int(k): int(v) for k, v in df[tgt].value_counts().sort_index().items()},
    "describe": df[feats].describe().T.round(4).to_dict(orient="index"),
    "corr_with_target": {c: float(np.corrcoef(df[c], df[tgt])[0, 1]) for c in feats},
    "corr_matrix": df[feats + [tgt]].corr().round(3).values.tolist(),
    "corr_labels": feats + [tgt],
    "n_sessions": int(df["session"].nunique()),
    "sessions": [],
    "split_counts": {},
}
for s, g in df.groupby("session"):
    eda["sessions"].append({"session": int(s), "start": str(g["timestamp"].min()),
                            "end": str(g["timestamp"].max()), "n": int(len(g)),
                            "class_counts": {int(k): int(v) for k, v in g[tgt].value_counts().sort_index().items()}})
for sp in ["train", "val", "test"]:
    g = df[df["split"] == sp]
    eda["split_counts"][sp] = {"n": int(len(g)),
                               "class_counts": {int(k): int(v) for k, v in g[tgt].value_counts().sort_index().items()},
                               "start": str(g["timestamp"].min()), "end": str(g["timestamp"].max())}
imb = max(eda["class_counts"].values()) / min(eda["class_counts"].values())
eda["imbalance_ratio_max_min"] = float(imb)
save_json(eda, "eda.json")
log(f"rows={eda['n_rows']} sessions={eda['n_sessions']} classes={eda['class_counts']} imbalance={imb:.1f}")
for sp, v in eda["split_counts"].items():
    log(f"{sp}: {v}")
for s in eda["sessions"]:
    log(f"session {s}")
