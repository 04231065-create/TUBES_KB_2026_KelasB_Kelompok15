"""Uji unit inti. Jalankan: python -m pytest tests/  (atau python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from occupancy import fuzzy  # noqa: E402
from occupancy.data import make_windows  # noqa: E402
from occupancy.ga import GeneticFeatureSelector  # noqa: E402
from occupancy.metrics import classification_report  # noqa: E402
from occupancy.pso import PSOOptimizer, decode  # noqa: E402
from occupancy.rnn import gradient_check  # noqa: E402


def test_bptt_gradient():
    assert gradient_check() < 1e-5


def test_windows_do_not_cross_groups():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10)
    g = np.array([0] * 4 + [1] * 6)
    Xw, yw, end = make_windows(X, y, g, 3)
    assert len(Xw) == (4 - 2) + (6 - 2)
    for w, e in zip(Xw, end):
        rows = (w[:, 0] / 2).astype(int)
        assert len(set(g[rows])) == 1 and rows[-1] == e


def test_metrics():
    r = classification_report([0, 0, 1, 2, 3], [0, 1, 1, 2, 3])
    assert abs(r["accuracy"] - 0.8) < 1e-9
    assert r["confusion_matrix"][0][1] == 1


def test_ga_finds_target_mask():
    target = np.array([1, 0, 1, 1, 0, 0, 1, 0], bool)
    fit = lambda m: {"fitness": float((m == target).mean()), "f1": 0.0, "n_features": int(m.sum())}  # noqa: E731
    res = GeneticFeatureSelector(8, fit, pop_size=12, generations=30, patience=30, seed=1,
                                 log=lambda *_: None).run()
    assert res["best_fitness"] == 1.0


def test_pso_optimizes_quadratic():
    bounds = {"hidden": [8, 96], "log10_lr": [-3.5, -1.5], "seq_len": [4, 30], "dropout": [0, 0.5],
              "log2_batch": [4, 7]}
    fit = lambda hp: {"fitness": -((hp["hidden"] - 40) ** 2) / 100 - (np.log10(hp["lr"]) + 2.5) ** 2}  # noqa: E731
    res = PSOOptimizer(bounds, fit, swarm_size=12, iterations=30, patience=30, seed=0,
                       log=lambda *_: None).run()
    assert abs(res["gbest_hparams"]["hidden"] - 40) <= 3
    assert decode([10, -2, 5, 0.1, 5])["batch_size"] == 32


def test_fuzzy_rule_base():
    ch = fuzzy.check_rule_base()
    assert ch["complete"] and ch["conflicts"] == 0 and ch["label_monotonic_violations"] == 0
    assert fuzzy.infer(0, 400, 22)["decision"] == "Minimum"
    assert fuzzy.infer(3, 1300, 29)["decision"] == "High"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("PASS", name)
