"""Mamdani Fuzzy Inference System untuk rekomendasi HVAC.

Input linguistik:
  1. Occupancy  (perkiraan jumlah penghuni dari RNN, E = sum_k k * p_k, 0..3 orang)
                 Empty | Few | Full
  2. CO2        (ppm, sensor S5_CO2, 300..1500)     Low | Normal | High
  3. Temperature(rata-rata S1..S4_Temp, 18..32 C)   Cool | Comfortable | Warm
Output:
  HVAC level (0..100 %)                            Minimum | Low | Medium | High

Inferensi : AND = min, implikasi = min (clipping), agregasi = max, defuzzifikasi = centroid.
Rule base : 27 aturan (3x3x3, lengkap & tanpa konflik), disusun dengan logika
            level = base(Occupancy) + adj(CO2) + adj(Temperature), di-clip ke 0..3.
            base: Empty=0, Few=1, Full=2 ; CO2: Low=0, Normal=0, High=+1 ;
            Temp: Cool=-1, Comfortable=0, Warm=+1.
"""
from __future__ import annotations

import itertools

import numpy as np


def trimf(x, a, b, c):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12), 1.0)
    right = np.where(c > b, (c - x) / (c - b + 1e-12), 1.0)
    return np.clip(np.minimum(left, right), 0, 1) * ((x >= a) & (x <= c))


def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    up = np.ones_like(x) if b == a else (x - a) / (b - a)
    down = np.ones_like(x) if d == c else (d - x) / (d - c)
    return np.clip(np.minimum(np.minimum(up, 1.0), down), 0, 1)


INPUTS = {
    "occupancy": {
        "range": (0.0, 3.0), "unit": "orang",
        "terms": {
            "Empty": ("trap", (0.0, 0.0, 0.2, 0.8)),
            "Few": ("tri", (0.3, 1.2, 2.1)),
            "Full": ("trap", (1.6, 2.4, 3.0, 3.0)),
        },
    },
    "co2": {
        "range": (300.0, 1500.0), "unit": "ppm",
        "terms": {
            "Low": ("trap", (300.0, 300.0, 450.0, 600.0)),
            "Normal": ("tri", (450.0, 700.0, 950.0)),
            "High": ("trap", (800.0, 1050.0, 1500.0, 1500.0)),
        },
    },
    "temperature": {
        "range": (18.0, 32.0), "unit": "C",
        "terms": {
            "Cool": ("trap", (18.0, 18.0, 22.0, 24.0)),
            "Comfortable": ("tri", (23.0, 25.0, 26.5)),
            "Warm": ("trap", (25.5, 27.0, 32.0, 32.0)),
        },
    },
}

OUTPUT = {
    "range": (0.0, 100.0), "unit": "%",
    "terms": {
        "Minimum": ("trap", (0.0, 0.0, 10.0, 25.0)),
        "Low": ("tri", (15.0, 35.0, 55.0)),
        "Medium": ("tri", (45.0, 62.0, 80.0)),
        "High": ("trap", (70.0, 85.0, 100.0, 100.0)),
    },
}

OUT_LEVELS = ["Minimum", "Low", "Medium", "High"]
_BASE = {"Empty": 0, "Few": 1, "Full": 2}
_CO2 = {"Low": 0, "Normal": 0, "High": 1}
_TEMP = {"Cool": -1, "Comfortable": 0, "Warm": 1}


def mf(spec, x):
    kind, p = spec
    return trimf(x, *p) if kind == "tri" else trapmf(x, *p)


def build_rules():
    rules = []
    for o, c, t in itertools.product(_BASE, _CO2, _TEMP):
        lvl = int(np.clip(_BASE[o] + _CO2[c] + _TEMP[t], 0, 3))
        rules.append({"occupancy": o, "co2": c, "temperature": t, "hvac": OUT_LEVELS[lvl]})
    return rules


RULES = build_rules()


def fuzzify(name, x):
    return {term: float(mf(spec, x)) for term, spec in INPUTS[name]["terms"].items()}


def infer(occupancy: float, co2: float, temperature: float, n_points: int = 501):
    """Kembalikan dict berisi nilai crisp, label keputusan, derajat keanggotaan input,
    kekuatan aturan dan kurva agregasi (untuk plot di dashboard)."""
    lo, hi = INPUTS["occupancy"]["range"]
    occupancy = float(np.clip(occupancy, lo, hi))
    lo, hi = INPUTS["co2"]["range"]
    co2 = float(np.clip(co2, lo, hi))
    lo, hi = INPUTS["temperature"]["range"]
    temperature = float(np.clip(temperature, lo, hi))
    mu = {"occupancy": fuzzify("occupancy", occupancy), "co2": fuzzify("co2", co2),
          "temperature": fuzzify("temperature", temperature)}
    u = np.linspace(*OUTPUT["range"], n_points)
    agg = np.zeros_like(u)
    fired = []
    for i, r in enumerate(RULES):
        s = min(mu["occupancy"][r["occupancy"]], mu["co2"][r["co2"]],
                mu["temperature"][r["temperature"]])
        if s > 0:
            agg = np.maximum(agg, np.minimum(s, mf(OUTPUT["terms"][r["hvac"]], u)))
            fired.append({"rule": i + 1, **r, "strength": float(s)})
    if agg.sum() == 0:
        crisp = 0.0
    else:
        crisp = float((u * agg).sum() / agg.sum())
    out_mu = {term: float(mf(spec, crisp)) for term, spec in OUTPUT["terms"].items()}
    label = max(out_mu, key=out_mu.get)
    fired.sort(key=lambda d: -d["strength"])
    return {"hvac_percent": crisp, "decision": label, "membership_in": mu,
            "membership_out": out_mu, "fired_rules": fired, "universe": u, "aggregated": agg}


def expected_occupancy(prob: np.ndarray) -> np.ndarray:
    prob = np.atleast_2d(prob)
    return prob @ np.arange(prob.shape[1])


def explain(res: dict, occupancy: float, co2: float, temperature: float) -> str:
    top = res["fired_rules"][:3]
    parts = [f"IF Occupancy={r['occupancy']} AND CO2={r['co2']} AND Temp={r['temperature']} "
             f"THEN HVAC={r['hvac']} (alpha={r['strength']:.2f})" for r in top]
    return (f"Perkiraan penghuni {occupancy:.2f} orang, CO2 {co2:.0f} ppm, suhu {temperature:.2f} C. "
            f"Aturan paling aktif: " + "; ".join(parts) +
            f". Hasil defuzzifikasi centroid = {res['hvac_percent']:.1f}% -> {res['decision']}.")


def check_rule_base():
    """Uji konsistensi: lengkap (27 kombinasi unik), tidak ada konflik, monoton."""
    keys = [(r["occupancy"], r["co2"], r["temperature"]) for r in RULES]
    complete = len(set(keys)) == 27 == len(keys)
    conflicts = 0
    seen = {}
    for k, r in zip(keys, RULES):
        if k in seen and seen[k] != r["hvac"]:
            conflicts += 1
        seen[k] = r["hvac"]
    # monotonicity: jika occupancy / CO2 / suhu naik, LEVEL keputusan tidak boleh turun.
    # Nilai crisp centroid boleh turun sedikit di zona transisi (sifat bawaan centroid
    # ketika satu himpunan output di-clip pada alpha berbeda), jadi besarnya juga dicatat.
    grid_o = np.linspace(0, 3, 13)
    grid_c = np.linspace(300, 1500, 13)
    grid_t = np.linspace(18, 32, 15)
    rank = {k: i for i, k in enumerate(OUT_LEVELS)}
    label_viol, crisp_viol, total, max_drop = 0, 0, 0, 0.0

    def sweep(results):
        nonlocal label_viol, crisp_viol, total, max_drop
        vals = np.array([r["hvac_percent"] for r in results])
        lv = np.array([rank[r["decision"]] for r in results])
        d = np.diff(vals)
        label_viol += int(np.sum(np.diff(lv) < 0))
        crisp_viol += int(np.sum(d < -0.5))
        max_drop = max(max_drop, float(-d.min()) if d.min() < 0 else 0.0)
        total += len(vals) - 1

    for c in grid_c[::3]:
        for t in grid_t[::3]:
            sweep([infer(o, c, t, 201) for o in grid_o])
    for o in grid_o[::3]:
        for t in grid_t[::3]:
            sweep([infer(o, c, t, 201) for c in grid_c])
    for o in grid_o[::3]:
        for c in grid_c[::3]:
            sweep([infer(o, c, t, 201) for t in grid_t])
    return {"n_rules": len(RULES), "complete": bool(complete), "conflicts": conflicts,
            "monotonic_checks": total, "label_monotonic_violations": label_viol,
            "crisp_drops_over_0p5": crisp_viol, "max_crisp_drop_pct": round(max_drop, 2)}
