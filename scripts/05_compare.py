"""W14 - Eksperimen pembanding 4 konfigurasi (A-D), masing-masing 5 seed training.

A Baseline NN   : 16 sensor, hyperparameter default
B GA + NN       : sensor hasil GA, hyperparameter default
C PSO + NN      : 16 sensor, hyperparameter hasil PSO (all)
D GA + PSO + NN : sensor hasil GA, hyperparameter hasil PSO (ga)

Protokol sama untuk semua: split kronologis yang sama, max_epochs & patience sama,
early stopping pada validation, test dipakai SEKALI untuk evaluasi akhir.
Model yang disimpan per konfigurasi = seed dengan skor VALIDASI (macro-F1 - log-loss) tertinggi.
"""
import numpy as np
from _common import MODELS, Logger, load_json, save_json

from occupancy.data import DataBundle, load_config
from occupancy.trainer import train_eval

log = Logger("05_compare")
cfg = load_config()
b = DataBundle(cfg)
tr = cfg["train"]
ga_feats = load_json("ga_results.json")["final_selection"]["features"]
pso_all = load_json("pso_all.json")["final"]["hparams"]
pso_ga = load_json("pso_ga.json")["final"]["hparams"]
CONFIGS = {
    "baseline_rnn": ("A", "Baseline NN", b.features, cfg["baseline_hparams"]),
    "ga_rnn": ("B", "GA + NN", ga_feats, cfg["baseline_hparams"]),
    "pso_rnn": ("C", "PSO + NN", b.features, pso_all),
    "ga_pso_rnn": ("D", "GA + PSO + NN", ga_feats, pso_ga),
}
results = {}
for name, (code, label, feats, hp) in CONFIGS.items():
    idx = [b.features.index(f) for f in feats]
    runs, best = [], None
    for s in cfg["final_train_seeds"]:
        model, out = train_eval(b, idx, hp, seed=s, max_epochs=tr["max_epochs"], patience=tr["patience"],
                                l2=tr["l2"], clip=tr["clip"], eval_test=True)
        runs.append({"seed": s, "val": out["val"], "test": out["test"], "train_time_s": out["train_time_s"],
                     "epochs_run": out["epochs_run"]})
        log(f"{code} {label:14s} seed {s}: val F1 {out['val']['f1_macro']:.4f} test F1 "
            f"{out['test']['f1_macro']:.4f} acc {out['test']['accuracy']:.4f} t {out['train_time_s']:.1f}s")
        vs = out["val"]["f1_macro"] - out["val"]["log_loss"]
        if best is None or vs > best[1]["val"]["f1_macro"] - best[1]["val"]["log_loss"]:
            best = (model, out, s)
    model, out, s = best
    model.save(MODELS / f"{name}.npz", features=feats, hparams=hp, config=label, code=code,
               selected_seed=s)
    # prediksi test untuk dashboard (confusion matrix, time plot)
    Xte, yte, end = b.windows("test", int(hp["seq_len"]), idx)
    prob = model.predict_proba(Xte)

    def agg(key, split):
        v = np.array([r[split][key] for r in runs])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "best": float(v.max())}

    results[name] = {
        "code": code, "label": label, "features": feats, "n_features": len(feats), "hparams": hp,
        "n_params": model.n_params(), "runs": runs, "selected_seed": s,
        "selected_model": {"val": out["val"], "test": out["test"], "history": model.history},
        "summary": {k: {"val": agg(k, "val"), "test": agg(k, "test")}
                    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "log_loss"]},
        "train_time_mean_s": float(np.mean([r["train_time_s"] for r in runs])),
        "test_predictions": {"row_index": end.tolist(), "y_true": yte.tolist(),
                             "y_pred": prob.argmax(1).tolist(), "prob": np.round(prob, 4).tolist()},
    }
save_json(results, "comparison.json")
for name, r in results.items():
    s = r["summary"]
    log(f"{r['code']} {r['label']:14s} nfeat {r['n_features']:2d} | val F1 {s['f1_macro']['val']['mean']:.4f}"
        f"+-{s['f1_macro']['val']['std']:.4f} | test F1 {s['f1_macro']['test']['mean']:.4f}"
        f"+-{s['f1_macro']['test']['std']:.4f} | test acc {s['accuracy']['test']['mean']:.4f} | "
        f"time {r['train_time_mean_s']:.1f}s")
