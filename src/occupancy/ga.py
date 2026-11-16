"""Genetic Algorithm untuk seleksi sensor/fitur.

Kromosom : vektor biner panjang 16 (1 = sensor dipakai, 0 = dibuang).
Fitness  : F(c) = mean_seed[ macroF1_val(RNN(c)) - logloss_val(RNN(c)) ] - lambda * |c| / 16
           RNN dilatih pada split TRAIN dan dinilai pada split VALIDATION (test tidak disentuh).
           Log-loss dipakai karena macro-F1 validasi cepat jenuh (~1.0). Seed training tetap
           -> fitness deterministik untuk kromosom yang sama (di-cache).
Operator : tournament selection (k), uniform crossover (pc), bit-flip mutation (pm),
           elitism (e individu terbaik dibawa ke generasi berikutnya).
Stopping : jumlah generasi maksimum ATAU tidak ada perbaikan best-fitness selama `patience` generasi.
"""
from __future__ import annotations

import time

import numpy as np


class GeneticFeatureSelector:
    def __init__(self, n_genes, fitness_fn, pop_size=16, generations=15, tournament_k=3,
                 p_crossover=0.9, p_mutation=None, elitism=2, patience=6, seed=0, log=print):
        self.n = n_genes
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.generations = generations
        self.k = tournament_k
        self.pc = p_crossover
        self.pm = p_mutation if p_mutation is not None else 1.0 / n_genes
        self.elitism = elitism
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.cache: dict = {}
        self.log = log
        self.n_evals = 0

    def _repair(self, c):
        if c.sum() == 0:
            c[self.rng.integers(self.n)] = 1
        return c

    def _fitness(self, c):
        key = "".join(map(str, c.astype(int)))
        if key not in self.cache:
            self.cache[key] = self.fitness_fn(c.astype(bool))
            self.n_evals += 1
        return self.cache[key]

    def _tournament(self, pop, fit):
        idx = self.rng.choice(len(pop), self.k, replace=False)
        return pop[idx[np.argmax(fit[idx])]].copy()

    def run(self):
        t0 = time.time()
        pop = (self.rng.random((self.pop_size, self.n)) < 0.5).astype(int)
        pop[0] = 1  # satu individu berisi semua fitur (titik acuan)
        pop = np.array([self._repair(c) for c in pop])
        fit = np.array([self._fitness(c)["fitness"] for c in pop])
        history = []
        best_f, wait = -np.inf, 0
        for g in range(self.generations + 1):
            order = np.argsort(-fit)
            best = pop[order[0]]
            info = self._fitness(best)
            history.append({
                "generation": g, "best_fitness": float(fit[order[0]]),
                "mean_fitness": float(fit.mean()), "std_fitness": float(fit.std()),
                "best_f1": float(info["f1"]), "best_n_features": int(best.sum()),
                "best_chromosome": "".join(map(str, best)), "evals_cum": self.n_evals,
                "elapsed_s": time.time() - t0,
            })
            self.log(f"[GA seed {self.seed}] gen {g:2d} best {fit[order[0]]:.4f} "
                     f"mean {fit.mean():.4f} nfeat {best.sum()} evals {self.n_evals}")
            if fit[order[0]] > best_f + 1e-6:
                best_f, wait = fit[order[0]], 0
            else:
                wait += 1
            if g == self.generations or wait >= self.patience:
                break
            new = [pop[i].copy() for i in order[:self.elitism]]
            while len(new) < self.pop_size:
                p1, p2 = self._tournament(pop, fit), self._tournament(pop, fit)
                if self.rng.random() < self.pc:
                    mask = self.rng.random(self.n) < 0.5
                    c1 = np.where(mask, p1, p2)
                    c2 = np.where(mask, p2, p1)
                else:
                    c1, c2 = p1, p2
                for c in (c1, c2):
                    flip = self.rng.random(self.n) < self.pm
                    c = np.where(flip, 1 - c, c)
                    new.append(self._repair(c))
            pop = np.array(new[:self.pop_size])
            fit = np.array([self._fitness(c)["fitness"] for c in pop])
        order = np.argsort(-fit)
        best = pop[order[0]]
        return {
            "seed": self.seed, "best_chromosome": "".join(map(str, best)),
            "best_fitness": float(fit[order[0]]), "best_info": self._fitness(best),
            "history": history, "n_evals": self.n_evals, "time_s": time.time() - t0,
            "final_population": ["".join(map(str, c)) for c in pop],
        }
