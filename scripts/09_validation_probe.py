"""Uji protokol validasi (tanpa test): apakah skor bisa membedakan subset sensor?

Membandingkan skor (macro-F1 - log-loss) dari
  (1) satu blok validasi [60%, 80%)  -> protokol iterasi 1
  (2) 3 fold CV forward-chaining     -> protokol iterasi 2 (config.cv_folds)
untuk beberapa subset sensor, memakai RNN fitness GA. Test tidak dipakai.
"""
from _common import Logger, load_json, save_json

from occupancy.data import DataBundle, cv_folds, load_config
from occupancy.trainer import cv_score

log = Logger("09_validation_probe")
cfg = load_config()
fm = cfg["ga"]["fitness_model"]
F = cfg["sensor_features"]
main = DataBundle(cfg)
folds = cv_folds(cfg)
iter1 = load_json("iter1_single_val/ga_results.json")["final_selection"]["features"]
subsets = {
    "Semua 16 sensor": F,
    "Subset GA iterasi 1": iter1,
    "Tanpa sensor suhu (12)": [f for f in F if "Temp" not in f],
    "Light + CO2 + PIR (8)": [f for f in F if "Light" in f or "CO2" in f or "PIR" in f],
    "Light saja (4)": [f for f in F if "Light" in f],
}
rows = []
for name, feats in subsets.items():
    idx = [F.index(f) for f in feats]
    single = cv_score([main], idx, fm, (0,), 1.0, max_epochs=fm["max_epochs"], patience=fm["patience"])
    cv = cv_score(folds, idx, fm, (0,), 1.0, max_epochs=fm["max_epochs"], patience=fm["patience"])
    rows.append({"subset": name, "features": feats, "single_val_score": single["score"],
                 "single_val_f1": single["f1"], "cv_score": cv["score"], "cv_f1": cv["f1"],
                 "cv_per_fold": cv["per_fold"]})
    log(f"{name:26s} single-val score {single['score']:.4f} (F1 {single['f1']:.4f}) | "
        f"CV score {cv['score']:.4f} (F1 {cv['f1']:.4f}) folds "
        + " ".join(f"{p['f1']:.3f}" for p in cv["per_fold"]))
save_json({"folds": cfg["cv_folds"], "rows": rows}, "validation_probe.json")
