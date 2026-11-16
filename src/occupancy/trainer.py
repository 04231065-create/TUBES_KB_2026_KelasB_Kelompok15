"""Helper untuk melatih/evaluasi RNN pada subset fitur + hyperparameter tertentu."""
from __future__ import annotations

import numpy as np

from .data import DataBundle, class_weights
from .metrics import classification_report
from .rnn import ElmanRNN


def train_eval(bundle: DataBundle, feat_idx, hp: dict, seed: int = 0, max_epochs: int = 40,
               patience: int = 6, l2: float = 1e-4, clip: float = 5.0, eval_test: bool = False,
               verbose: bool = False):
    """Latih pada train, early stopping pada val. Test hanya dievaluasi jika eval_test=True
    (dipakai untuk evaluasi akhir; GA & PSO TIDAK PERNAH melihat test)."""
    feat_idx = np.asarray(feat_idx, dtype=int)
    T = int(hp["seq_len"])
    Xtr, ytr, _ = bundle.windows("train", T, feat_idx)
    Xva, yva, _ = bundle.windows("val", T, feat_idx)
    cw = class_weights(ytr)
    model = ElmanRNN(len(feat_idx), int(hp["hidden"]), 4, float(hp.get("dropout", 0.0)), seed)
    model.fit(Xtr, ytr, Xva, yva, lr=float(hp["lr"]), batch_size=int(hp["batch_size"]),
              max_epochs=max_epochs, patience=patience, clip=clip, l2=l2, class_w=cw,
              verbose=verbose)
    pva = model.predict_proba(Xva)
    out = {"val": classification_report(yva, pva.argmax(1))}
    out["val"]["log_loss"] = float(-np.mean(np.log(pva[np.arange(len(yva)), yva] + 1e-12)))
    if eval_test:
        Xte, yte, _ = bundle.windows("test", T, feat_idx)
        pte = model.predict_proba(Xte)
        out["test"] = classification_report(yte, pte.argmax(1))
        out["test"]["log_loss"] = float(-np.mean(np.log(pte[np.arange(len(yte)), yte] + 1e-12)))
    out["train_time_s"] = model.history["train_time_s"]
    out["epochs_run"] = model.history["epochs_run"]
    return model, out


def cv_score(folds: list, feat_idx, hp: dict, seeds=(0,), ce_weight: float = 1.0, **kw):
    """Skor untuk GA/PSO: rata-rata atas fold CV forward-chaining (dan seed) dari
    (macroF1_val - ce_weight * logloss_val). Semua fold berada di periode train+val;
    test tidak pernah disentuh. Beberapa jendela validasi kronologis membuat skor lebih
    representatif daripada satu blok validasi yang macro-F1-nya sudah jenuh (~1.0)."""
    f1s, ces, accs, per_fold, t = [], [], [], [], 0.0
    for k, b in enumerate(folds):
        for s in seeds:
            _, out = train_eval(b, feat_idx, hp, seed=s, **kw)
            f1s.append(out["val"]["f1_macro"])
            ces.append(out["val"]["log_loss"])
            accs.append(out["val"]["accuracy"])
            per_fold.append({"fold": k, "seed": s, "f1": out["val"]["f1_macro"], "log_loss": out["val"]["log_loss"]})
            t += out["train_time_s"]
    f1, ce = float(np.mean(f1s)), float(np.mean(ces))
    return {"score": f1 - ce_weight * ce, "f1": f1, "log_loss": ce, "acc": float(np.mean(accs)),
            "per_fold": per_fold, "time_s": t}
