"""W15 - Validasi Fuzzy HVAC: konsistensi rule, skenario ekstrem, keputusan pada test set.

Catatan: keputusan fuzzy TIDAK disetarakan dengan akurasi NN. Yang diuji adalah
(1) konsistensi & monotonisitas rule base, (2) kewajaran pada skenario ekstrem,
(3) seberapa sering keputusan dari occupancy PREDIKSI sama dengan keputusan dari
occupancy SEBENARNYA (dengan CO2 & suhu yang sama), dan (4) proksi energi.
"""
import numpy as np
from _common import Logger, load_json, save_json

from occupancy import fuzzy
from occupancy.data import OUT_DIR, DataBundle, load_config
from occupancy.pipeline import TEMP_COLS

log = Logger("06_fuzzy_eval")
cfg = load_config()
b = DataBundle(cfg)
comp = load_json("comparison.json")["ga_pso_rnn"]
tp = comp["test_predictions"]
rows = np.array(tp["row_index"])
prob = np.array(tp["prob"])
y_true = np.array(tp["y_true"])
co2 = b.df["S5_CO2"].to_numpy()[rows]
temp = b.df[TEMP_COLS].to_numpy().mean(axis=1)[rows]
exp_occ = fuzzy.expected_occupancy(prob)

check = fuzzy.check_rule_base()
log(f"rule check: {check}")

dec_pred, dec_true, pct_pred, pct_true = [], [], [], []
for i in range(len(rows)):
    rp = fuzzy.infer(exp_occ[i], co2[i], temp[i], 301)
    rt = fuzzy.infer(float(y_true[i]), co2[i], temp[i], 301)
    dec_pred.append(rp["decision"]); pct_pred.append(rp["hvac_percent"])
    dec_true.append(rt["decision"]); pct_true.append(rt["hvac_percent"])
dec_pred, dec_true = np.array(dec_pred), np.array(dec_true)
pct_pred, pct_true = np.array(pct_pred), np.array(pct_true)
L = fuzzy.OUT_LEVELS
cm = np.zeros((4, 4), int)
for a, p in zip(dec_true, dec_pred):
    cm[L.index(a), L.index(p)] += 1
rank = {k: i for i, k in enumerate(L)}
diff = np.array([rank[p] - rank[t] for p, t in zip(dec_pred, dec_true)])

scenarios = [
    ("Ruang kosong, CO2 rendah, sejuk", 0.0, 400, 22.0),
    ("Ruang kosong, CO2 tinggi (sisa), hangat", 0.0, 1200, 28.0),
    ("1 orang, normal", 1.0, 650, 25.0),
    ("2 orang, CO2 naik, sedikit hangat", 2.0, 900, 26.2),
    ("3 orang, CO2 tinggi, hangat", 3.0, 1300, 29.0),
    ("3 orang, CO2 rendah, dingin", 3.0, 400, 20.0),
    ("Batas bawah semua input", 0.0, 300, 18.0),
    ("Batas atas semua input", 3.0, 1500, 32.0),
]
scen = []
for name, o, c, t in scenarios:
    r = fuzzy.infer(o, c, t)
    scen.append({"scenario": name, "occupancy": o, "co2": c, "temperature": t,
                 "hvac_percent": round(r["hvac_percent"], 2), "decision": r["decision"],
                 "top_rule": r["fired_rules"][0] if r["fired_rules"] else None})
    log(f"{name}: {r['hvac_percent']:.1f}% {r['decision']}")

out = {
    "rule_check": check, "rules": fuzzy.RULES, "scenarios": scen,
    "test_n": int(len(rows)),
    "decision_agreement": float((dec_pred == dec_true).mean()),
    "decision_within_one_level": float((np.abs(diff) <= 1).mean()),
    "under_ventilation_rate": float((diff < 0).mean()),
    "over_ventilation_rate": float((diff > 0).mean()),
    "decision_confusion_true_vs_pred": cm.tolist(),
    "decision_distribution_pred": {k: int((dec_pred == k).sum()) for k in L},
    "decision_distribution_true": {k: int((dec_true == k).sum()) for k in L},
    "mean_hvac_pred": float(pct_pred.mean()), "mean_hvac_true": float(pct_true.mean()),
    "mae_hvac_percent": float(np.abs(pct_pred - pct_true).mean()),
    "energy_proxy_vs_always_high": float(1 - pct_pred.mean() / 100.0),
    "series": {"row_index": rows.tolist(), "hvac_pred": np.round(pct_pred, 2).tolist(),
               "hvac_true": np.round(pct_true, 2).tolist(), "exp_occ": np.round(exp_occ, 3).tolist()},
}
save_json(out, "fuzzy_eval.json")
log(f"agreement {out['decision_agreement']:.4f} within1 {out['decision_within_one_level']:.4f} "
    f"under {out['under_ventilation_rate']:.4f} over {out['over_ventilation_rate']:.4f} "
    f"MAE% {out['mae_hvac_percent']:.2f} mean pred {out['mean_hvac_pred']:.1f}")

# contoh CSV untuk demo upload di dashboard: potongan test KONTINU (satu sesi) dengan variasi kelas terbanyak
test = b.df[b.df["split"] == "test"]
best_sess = max(test["session"].unique(),
                key=lambda s: (test.loc[test["session"] == s, cfg["target"]].nunique(), (test["session"] == s).sum()))
seg = test[test["session"] == best_sess]
y = seg[cfg["target"]].to_numpy()
chg = np.where(np.diff(y) != 0)[0]
start = int(np.clip((chg[0] - 30) if len(chg) else 0, 0, max(len(seg) - 240, 0)))
sample = seg.iloc[start:start + 240][["Date", "Time"] + b.features]
(OUT_DIR / "data").mkdir(exist_ok=True)
sample.to_csv(OUT_DIR / "data" / "sample_input.csv", index=False)
log(f"sample_input.csv rows {len(sample)} from test session {best_sess}, classes "
    f"{sorted(set(y[start:start + 240].tolist()))}")
