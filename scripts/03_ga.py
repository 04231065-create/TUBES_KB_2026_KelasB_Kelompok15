"""W13 - Genetic Algorithm untuk seleksi sensor (beberapa seed independen).

Fitness = mean_fold[ macroF1_val - logloss_val ](RNN kecil pada subset) - lambda * n_fitur/16,
dengan 3 fold CV forward-chaining di dalam periode train+val (config.cv_folds).
Test tidak pernah disentuh.
"""
import time

import numpy as np
from _common import Logger, save_json

from occupancy.data import DataBundle, cv_folds, load_config
from occupancy.ga import GeneticFeatureSelector
from occupancy.trainer import cv_score

log = Logger("03_ga")
cfg = load_config()
gcfg = cfg["ga"]
fm = gcfg["fitness_model"]
b = DataBundle(cfg)
folds = cv_folds(cfg)
feats = b.features
n = len(feats)


def fitness(mask):
    idx = np.where(mask)[0]
    t = time.time()
    v = cv_score(folds, idx, fm, gcfg["cv_seeds"], gcfg["ce_weight"], max_epochs=fm["max_epochs"],
                 patience=fm["patience"], l2=cfg["train"]["l2"])
    return {"fitness": v["score"] - gcfg["lambda_size"] * len(idx) / n, "score": v["score"], "f1": v["f1"],
            "log_loss": v["log_loss"], "acc": v["acc"], "per_fold": v["per_fold"], "n_features": int(len(idx)),
            "time_s": time.time() - t}


runs = []
for seed in gcfg["seeds"]:
    ga = GeneticFeatureSelector(n, fitness, gcfg["pop_size"], gcfg["generations"], gcfg["tournament_k"],
                                gcfg["p_crossover"], gcfg["p_mutation"], gcfg["elitism"], gcfg["patience"],
                                seed=seed, log=log)
    res = ga.run()
    res["selected_features"] = [f for f, bit in zip(feats, res["best_chromosome"]) if bit == "1"]
    log(f"seed {seed}: best fitness {res['best_fitness']:.4f} f1 {res['best_info']['f1']:.4f} "
        f"features {res['selected_features']} evals {res['n_evals']} time {res['time_s']:.1f}s")
    runs.append(res)

best_fit = np.array([r["best_fitness"] for r in runs])
best_f1 = np.array([r["best_info"]["f1"] for r in runs])
freq = np.mean([[int(c) for c in r["best_chromosome"]] for r in runs], axis=0)
winner = runs[int(np.argmax(best_fit))]
summary = {
    "config": gcfg, "features": feats, "runs": runs,
    "stats": {"best_fitness_best": float(best_fit.max()), "best_fitness_mean": float(best_fit.mean()),
              "best_fitness_std": float(best_fit.std(ddof=1)) if len(runs) > 1 else 0.0,
              "f1_best": float(best_f1.max()), "f1_mean": float(best_f1.mean()),
              "f1_std": float(best_f1.std(ddof=1)) if len(runs) > 1 else 0.0,
              "n_features_mean": float(np.mean([r["best_info"]["n_features"] for r in runs])),
              "total_evals": int(sum(r["n_evals"] for r in runs)),
              "total_time_s": float(sum(r["time_s"] for r in runs))},
    "selection_frequency": {f: float(v) for f, v in zip(feats, freq)},
    "final_selection": {"rule": "kromosom dengan fitness validasi tertinggi di antara semua seed",
                        "seed": winner["seed"], "chromosome": winner["best_chromosome"],
                        "features": winner["selected_features"]},
}
save_json(summary, "ga_results.json")
log(f"FINAL GA features ({len(winner['selected_features'])}): {winner['selected_features']}")
log(f"stats {summary['stats']}")
