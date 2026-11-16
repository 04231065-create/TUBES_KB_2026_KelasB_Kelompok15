"""Pipeline inference end-to-end: data mentah -> preprocessing -> RNN -> Fuzzy HVAC.

Artefak yang dimuat (folder models/):
  <name>.npz          bobot RNN + metadata (fitur terpilih, hyperparameter)
  scaler.json         mean/std dari split TRAIN (16 sensor)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import fuzzy
from .data import ROOT, StandardScaler1D, load_config
from .rnn import ElmanRNN

TEMP_COLS = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp"]


class OccupancyPipeline:
    def __init__(self, model_name: str = "ga_pso_rnn", models_dir: str | Path | None = None):
        self.cfg = load_config()
        models_dir = Path(models_dir) if models_dir else ROOT / "models"
        self.model, self.meta = ElmanRNN.load(models_dir / f"{model_name}.npz")
        with open(models_dir / "scaler.json", "r", encoding="utf-8") as f:
            self.scaler = StandardScaler1D.from_dict(json.load(f))
        self.all_features = list(self.cfg["sensor_features"])
        self.features = list(self.meta["features"])
        self.feat_idx = [self.all_features.index(f) for f in self.features]
        self.T = int(self.meta["hparams"]["seq_len"])

    def _scaled(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.all_features if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom berikut tidak ada di input: {missing}")
        X = df[self.all_features].to_numpy(float)
        if np.isnan(X).any():
            X = pd.DataFrame(X).ffill().bfill().to_numpy()
        return self.scaler.transform(X)[:, self.feat_idx]

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Setiap baris ke-t (t >= T-1) diprediksi dari jendela T baris terakhir."""
        if len(df) < self.T:
            raise ValueError(f"Butuh minimal {self.T} baris berurutan (seq_len model).")
        Xs = self._scaled(df)
        idx = np.arange(self.T - 1, len(df))
        Xw = np.stack([Xs[i - self.T + 1:i + 1] for i in idx])
        prob = self.model.predict_proba(Xw)
        exp_occ = fuzzy.expected_occupancy(prob)
        rows = []
        for j, i in enumerate(idx):
            co2 = float(df["S5_CO2"].iloc[i])
            temp = float(df[TEMP_COLS].iloc[i].mean())
            res = fuzzy.infer(exp_occ[j], co2, temp, 301)
            rows.append({
                "row": int(i), "pred_occupancy": int(prob[j].argmax()),
                "confidence": float(prob[j].max()), "expected_occupancy": float(exp_occ[j]),
                **{f"p_{k}": float(prob[j, k]) for k in range(prob.shape[1])},
                "co2_ppm": co2, "temp_mean_c": temp,
                "hvac_percent": res["hvac_percent"], "hvac_decision": res["decision"],
            })
        return pd.DataFrame(rows)

    def predict_steady(self, readings: dict) -> dict:
        """Input manual: satu set nilai sensor diasumsikan konstan sepanjang jendela T
        (kondisi tunak). Kembalikan probabilitas, prediksi dan keputusan fuzzy."""
        row = {c: float(readings.get(c, 0.0)) for c in self.all_features}
        df = pd.DataFrame([row] * self.T)
        Xs = self._scaled(df)[None]
        prob = self.model.predict_proba(Xs)[0]
        exp_occ = float(fuzzy.expected_occupancy(prob)[0])
        temp = float(np.mean([row[c] for c in TEMP_COLS]))
        res = fuzzy.infer(exp_occ, row["S5_CO2"], temp)
        return {"prob": prob, "pred": int(prob.argmax()), "expected_occupancy": exp_occ,
                "temp_mean": temp, "fuzzy": res,
                "explanation": fuzzy.explain(res, exp_occ, row["S5_CO2"], temp)}
