"""Elman RNN (many-to-one) untuk klasifikasi jumlah penghuni, ditulis dengan NumPy.

Arsitektur:
    x_t (F)  ->  h_t = tanh(x_t W_x + h_{t-1} W_h + b_h)   untuk t = 1..T
    h_T      ->  dropout(p)  ->  logits = h_T W_o + b_o  ->  softmax (4 kelas: 0,1,2,3 orang)

Training: weighted cross-entropy (bobot kelas untuk imbalance) + L2,
backpropagation through time (BPTT) penuh, Adam, gradient clipping (norm global),
early stopping berdasarkan validation loss (cross-entropy).

Alasan memilih RNN: data sensor adalah deret waktu (sampling 30 detik). Perubahan
CO2, cahaya dan suara saat orang masuk/keluar ruangan bersifat temporal, sehingga
model yang membaca jendela beberapa langkah sebelumnya lebih tepat dibanding
model yang hanya melihat satu titik waktu.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .metrics import classification_report


class ElmanRNN:
    def __init__(self, n_features: int, hidden: int = 32, n_classes: int = 4,
                 dropout: float = 0.0, seed: int = 0):
        self.F, self.H, self.C = int(n_features), int(hidden), int(n_classes)
        self.dropout = float(dropout)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        lim = np.sqrt(6.0 / (self.F + self.H))
        self.params = {
            "Wx": rng.uniform(-lim, lim, (self.F, self.H)),
            "Wh": self._orthogonal(rng, self.H) * 0.9,
            "bh": np.zeros(self.H),
            "Wo": rng.uniform(-np.sqrt(6.0 / (self.H + self.C)), np.sqrt(6.0 / (self.H + self.C)),
                              (self.H, self.C)),
            "bo": np.zeros(self.C),
        }
        self.history: dict = {}

    @staticmethod
    def _orthogonal(rng, n):
        q, r = np.linalg.qr(rng.normal(size=(n, n)))
        return q * np.sign(np.diag(r))

    def n_params(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    # ------------------------------------------------------------------ forward
    def _forward(self, X, train=False, rng=None):
        P = self.params
        B, T, _ = X.shape
        hs = np.zeros((T + 1, B, self.H))
        for t in range(T):
            hs[t + 1] = np.tanh(X[:, t] @ P["Wx"] + hs[t] @ P["Wh"] + P["bh"])
        hT = hs[T]
        mask = None
        if train and self.dropout > 0:
            mask = (rng.random(hT.shape) >= self.dropout) / (1.0 - self.dropout)
            hT = hT * mask
        logits = hT @ P["Wo"] + P["bo"]
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        prob = e / e.sum(axis=1, keepdims=True)
        return prob, hs, hT, mask

    def predict_proba(self, X, batch: int = 2048) -> np.ndarray:
        out = []
        for i in range(0, len(X), batch):
            out.append(self._forward(X[i:i + batch])[0])
        return np.concatenate(out) if out else np.zeros((0, self.C))

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ----------------------------------------------------------------- backward
    def _loss_grad(self, X, y, cw, l2, rng):
        P = self.params
        B, T, _ = X.shape
        prob, hs, hT, mask = self._forward(X, train=True, rng=rng)
        w = cw[y]
        wsum = w.sum()
        loss = -(w * np.log(prob[np.arange(B), y] + 1e-12)).sum() / wsum
        loss += 0.5 * l2 * (np.sum(P["Wx"] ** 2) + np.sum(P["Wh"] ** 2) + np.sum(P["Wo"] ** 2))
        d = prob.copy()
        d[np.arange(B), y] -= 1.0
        d *= (w / wsum)[:, None]
        g = {k: np.zeros_like(v) for k, v in P.items()}
        g["Wo"] = hT.T @ d + l2 * P["Wo"]
        g["bo"] = d.sum(axis=0)
        dh = d @ P["Wo"].T
        if mask is not None:
            dh *= mask
        for t in range(T, 0, -1):
            da = dh * (1.0 - hs[t] ** 2)
            g["Wx"] += X[:, t - 1].T @ da
            g["Wh"] += hs[t - 1].T @ da
            g["bh"] += da.sum(axis=0)
            dh = da @ P["Wh"].T
        g["Wx"] += l2 * P["Wx"]
        g["Wh"] += l2 * P["Wh"]
        return loss, g

    # -------------------------------------------------------------------- train
    def fit(self, Xtr, ytr, Xva, yva, lr=5e-3, batch_size=64, max_epochs=40,
            patience=6, clip=5.0, l2=1e-4, class_w=None, verbose=False,
            time_budget_s: float | None = None, select: str = "loss"):
        """select="loss": simpan bobot dengan validation loss terendah (default; macro-F1
        validasi cepat jenuh ~1.0 sehingga tidak bisa membedakan epoch).
        select="f1": simpan bobot dengan macro-F1 validasi tertinggi (tie -> loss terendah)."""
        rng = np.random.default_rng(self.seed + 1000)
        cw = np.ones(self.C) if class_w is None else np.asarray(class_w, float)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        best = (-1.0, np.inf)
        best_params = {k: p.copy() for k, p in self.params.items()}
        hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [],
                "val_acc": [], "val_f1": []}
        wait, t0 = 0, time.time()
        n = len(Xtr)
        batch_size = int(batch_size)
        for ep in range(1, max_epochs + 1):
            perm = rng.permutation(n)
            ep_loss, nb = 0.0, 0
            for i in range(0, n, batch_size):
                bi = perm[i:i + batch_size]
                loss, g = self._loss_grad(Xtr[bi], ytr[bi], cw, l2, rng)
                gnorm = np.sqrt(sum(np.sum(x ** 2) for x in g.values()))
                if gnorm > clip:
                    for k in g:
                        g[k] *= clip / (gnorm + 1e-12)
                step += 1
                for k in self.params:
                    m[k] = b1 * m[k] + (1 - b1) * g[k]
                    v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                    mh = m[k] / (1 - b1 ** step)
                    vh = v[k] / (1 - b2 ** step)
                    self.params[k] -= lr * mh / (np.sqrt(vh) + eps)
                ep_loss += loss
                nb += 1
            ptr = self.predict_proba(Xtr)
            pva = self.predict_proba(Xva)
            tr_loss = float(-np.mean(np.log(ptr[np.arange(len(ytr)), ytr] + 1e-12)))
            va_loss = float(-np.mean(np.log(pva[np.arange(len(yva)), yva] + 1e-12)))
            rep = classification_report(yva, pva.argmax(1), self.C)
            hist["epoch"].append(ep)
            hist["train_loss"].append(tr_loss)
            hist["val_loss"].append(va_loss)
            hist["train_acc"].append(float((ptr.argmax(1) == ytr).mean()))
            hist["val_acc"].append(rep["accuracy"])
            hist["val_f1"].append(rep["f1_macro"])
            if verbose:
                print(f"ep {ep:3d} train_loss {tr_loss:.4f} val_loss {va_loss:.4f} "
                      f"val_acc {rep['accuracy']:.4f} val_f1 {rep['f1_macro']:.4f}")
            score = (rep["f1_macro"], va_loss)
            if select == "loss":
                improved = va_loss < best[1] - 1e-5
            else:
                improved = score[0] > best[0] + 1e-4 or (abs(score[0] - best[0]) <= 1e-4 and score[1] < best[1])
            if improved:
                best = score
                best_params = {k: p.copy() for k, p in self.params.items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
            if time_budget_s and time.time() - t0 > time_budget_s:
                break
        self.params = best_params
        hist["best_val_f1"] = best[0]
        hist["best_val_loss"] = best[1]
        hist["select"] = select
        hist["train_time_s"] = time.time() - t0
        hist["epochs_run"] = len(hist["epoch"])
        self.history = hist
        return self

    # ------------------------------------------------------------------ save/load
    def save(self, path: str | Path, **meta):
        import json
        meta_json = json.dumps({"F": self.F, "H": self.H, "C": self.C,
                                "dropout": self.dropout, "seed": self.seed, **meta})
        np.savez(path, meta=np.array(meta_json), **self.params)

    @classmethod
    def load(cls, path: str | Path):
        import json
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        model = cls(meta["F"], meta["H"], meta["C"], meta["dropout"], meta["seed"])
        for k in model.params:
            model.params[k] = z[k]
        return model, meta


def gradient_check(seed: int = 0) -> float:
    """Uji numerik turunan BPTT (dipakai di tests/)."""
    rng = np.random.default_rng(seed)
    m = ElmanRNN(5, 6, 4, dropout=0.0, seed=seed)
    X = rng.normal(size=(7, 4, 5))
    y = rng.integers(0, 4, 7)
    cw = np.array([0.5, 1.0, 1.5, 2.0])
    _, g = m._loss_grad(X, y, cw, 1e-3, rng)
    max_rel = 0.0
    for k, p in m.params.items():
        it = np.nditer(p, flags=["multi_index"])
        for _ in range(min(p.size, 12)):
            idx = it.multi_index
            old = p[idx]
            p[idx] = old + 1e-5
            lp, _ = m._loss_grad(X, y, cw, 1e-3, rng)
            p[idx] = old - 1e-5
            lm, _ = m._loss_grad(X, y, cw, 1e-3, rng)
            p[idx] = old
            num = (lp - lm) / 2e-5
            rel = abs(num - g[k][idx]) / max(1e-8, abs(num) + abs(g[k][idx]))
            max_rel = max(max_rel, rel)
            it.iternext()
    return max_rel
