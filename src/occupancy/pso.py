"""Particle Swarm Optimization untuk hyperparameter RNN.

Partikel  : x = [hidden, log10_lr, seq_len, dropout, log2_batch]  (5 dimensi kontinu)
Decode    : hidden = round(x0), lr = 10**x1, seq_len = round(x2), dropout = x3,
            batch_size = 2**round(x4)
Fitness   : mean_seed[ macroF1_val - logloss_val ] (dimaksimalkan); test tidak disentuh.
Update    : v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x)
            x <- x + v      (v dibatasi |v| <= vmax_frac * (ub - lb); posisi di-clip ke batas,
                             komponen kecepatan yang menabrak batas dibalik & diredam)
            w menurun linear dari w_start ke w_end.
Stopping  : iterasi maksimum ATAU gbest tidak membaik selama `patience` iterasi.
"""
from __future__ import annotations

import time

import numpy as np

PARAM_NAMES = ["hidden", "log10_lr", "seq_len", "dropout", "log2_batch"]


def decode(x) -> dict:
    return {
        "hidden": int(round(x[0])),
        "lr": float(10 ** x[1]),
        "seq_len": int(round(x[2])),
        "dropout": float(np.clip(x[3], 0.0, 0.9)),
        "batch_size": int(2 ** int(round(x[4]))),
    }


class PSOOptimizer:
    def __init__(self, bounds: dict, fitness_fn, swarm_size=10, iterations=10, w_start=0.9,
                 w_end=0.4, c1=1.5, c2=1.5, vmax_frac=0.2, patience=5, seed=0, log=print):
        self.lb = np.array([bounds[k][0] for k in PARAM_NAMES], float)
        self.ub = np.array([bounds[k][1] for k in PARAM_NAMES], float)
        self.fitness_fn = fitness_fn
        self.S, self.iters = swarm_size, iterations
        self.w0, self.w1, self.c1, self.c2 = w_start, w_end, c1, c2
        self.vmax = vmax_frac * (self.ub - self.lb)
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.cache: dict = {}
        self.n_evals = 0
        self.log = log

    def _f(self, x):
        hp = decode(x)
        key = tuple(sorted(hp.items()))
        key = tuple((k, round(v, 6) if isinstance(v, float) else v) for k, v in key)
        if key not in self.cache:
            self.cache[key] = self.fitness_fn(hp)
            self.n_evals += 1
        return self.cache[key]

    def run(self):
        t0 = time.time()
        D = len(self.lb)
        X = self.lb + self.rng.random((self.S, D)) * (self.ub - self.lb)
        V = (self.rng.random((self.S, D)) - 0.5) * self.vmax
        fit = np.array([self._f(x)["fitness"] for x in X])
        pbest, pbest_f = X.copy(), fit.copy()
        g = int(np.argmax(pbest_f))
        gbest, gbest_f = pbest[g].copy(), pbest_f[g]
        history, wait = [], 0
        particles_log = []

        def record(it):
            history.append({"iteration": it, "gbest_fitness": float(gbest_f),
                            "mean_fitness": float(fit.mean()), "std_fitness": float(fit.std()),
                            "gbest_hparams": decode(gbest), "evals_cum": self.n_evals,
                            "elapsed_s": time.time() - t0})
            for i in range(self.S):
                particles_log.append({"iteration": it, "particle": i, "fitness": float(fit[i]),
                                      **{n: float(X[i, j]) for j, n in enumerate(PARAM_NAMES)}})
            self.log(f"[PSO seed {self.seed}] it {it:2d} gbest {gbest_f:.4f} mean {fit.mean():.4f} "
                     f"hp {decode(gbest)} evals {self.n_evals}")

        record(0)
        for it in range(1, self.iters + 1):
            w = self.w0 - (self.w0 - self.w1) * (it - 1) / max(self.iters - 1, 1)
            r1, r2 = self.rng.random((self.S, D)), self.rng.random((self.S, D))
            V = w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
            V = np.clip(V, -self.vmax, self.vmax)
            X = X + V
            hit = (X < self.lb) | (X > self.ub)
            V[hit] *= -0.5
            X = np.clip(X, self.lb, self.ub)
            fit = np.array([self._f(x)["fitness"] for x in X])
            imp = fit > pbest_f
            pbest[imp], pbest_f[imp] = X[imp], fit[imp]
            g = int(np.argmax(pbest_f))
            if pbest_f[g] > gbest_f + 1e-6:
                gbest, gbest_f, wait = pbest[g].copy(), pbest_f[g], 0
            else:
                wait += 1
            record(it)
            if wait >= self.patience:
                break
        return {"seed": self.seed, "gbest_position": gbest.tolist(), "gbest_hparams": decode(gbest),
                "gbest_fitness": float(gbest_f), "gbest_info": self._f(gbest), "history": history,
                "particles": particles_log, "n_evals": self.n_evals, "time_s": time.time() - t0}
