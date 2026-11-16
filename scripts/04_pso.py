"""W14 - PSO untuk hyperparameter RNN.

Pemakaian:  python scripts/04_pso.py all   -> PSO+NN     (16 sensor)
            python scripts/04_pso.py ga    -> GA+PSO+NN  (sensor hasil GA)
Fitness = mean_fold[ macroF1_val - logloss_val ] atas 3 fold CV forward-chaining. Test tidak disentuh.
"""
import sys
import numpy as np
from _common import Logger, load_json, save_json

from occupancy.data import DataBundle, cv_folds, load_config
from occupancy.pso import PSOOptimizer
from occupancy.trainer import cv_score

tag = sys.argv[1] if len(sys.argv) > 1 else "all"
log = Logger(f"04_pso_{tag}")
cfg = load_config()
pcfg = cfg["pso"]
b = DataBundle(cfg)
folds = cv_folds(cfg)
if tag == "ga":
    sel = load_json("ga_results.json")["final_selection"]["features"]
else:
    sel = b.features
idx = [b.features.index(f) for f in sel]
log(f"features ({len(idx)}): {sel}")


def fitness(hp):
    v = cv_score(folds, idx, hp, pcfg["cv_seeds"], pcfg["ce_weight"], max_epochs=pcfg["max_epochs"],
                 patience=pcfg["train_patience"], l2=cfg["train"]["l2"])
    return {"fitness": v["score"], "f1": v["f1"], "log_loss": v["log_loss"], "acc": v["acc"],
            "per_fold": v["per_fold"], "time_s": v["time_s"]}


runs = []
for seed in pcfg["seeds"]:
    pso = PSOOptimizer(pcfg["bounds"], fitness, pcfg["swarm_size"], pcfg["iterations"], pcfg["w_start"],
                       pcfg["w_end"], pcfg["c1"], pcfg["c2"], pcfg["vmax_frac"], pcfg["patience"],
                       seed=seed, log=log)
    res = pso.run()
    log(f"seed {seed}: gbest {res['gbest_fitness']:.4f} hp {res['gbest_hparams']} "
        f"evals {res['n_evals']} time {res['time_s']:.1f}s")
    runs.append(res)

g = np.array([r["gbest_fitness"] for r in runs])
winner = runs[int(np.argmax(g))]
save_json({"tag": tag, "features": sel, "config": pcfg, "runs": runs,
           "stats": {"gbest_best": float(g.max()), "gbest_mean": float(g.mean()),
                     "gbest_std": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
                     "total_evals": int(sum(r["n_evals"] for r in runs)),
                     "total_time_s": float(sum(r["time_s"] for r in runs))},
           "final": {"rule": "gbest dengan fitness validasi tertinggi di antara semua seed",
                     "seed": winner["seed"], "hparams": winner["gbest_hparams"],
                     "fitness": winner["gbest_fitness"]}}, f"pso_{tag}.json")
log(f"FINAL hp: {winner['gbest_hparams']} val score {winner['gbest_fitness']:.4f}")
