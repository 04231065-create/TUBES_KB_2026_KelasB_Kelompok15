"""W12 - Baseline RNN: 16 sensor, hyperparameter default (tanpa optimasi)."""
import json

from _common import MODELS, Logger, save_json

from occupancy.data import DataBundle, load_config
from occupancy.trainer import train_eval

log = Logger("02_baseline")
cfg = load_config()
b = DataBundle(cfg)
with open(MODELS / "scaler.json", "w", encoding="utf-8") as f:
    json.dump({**b.scaler.to_dict(), "features": b.features, "fitted_on": "train split only"}, f, indent=2)

hp = cfg["baseline_hparams"]
tr = cfg["train"]
model, out = train_eval(b, list(range(16)), hp, seed=cfg["global_seed"], max_epochs=tr["max_epochs"],
                        patience=tr["patience"], l2=tr["l2"], clip=tr["clip"], verbose=True)
log(f"baseline val: acc={out['val']['accuracy']:.4f} f1={out['val']['f1_macro']:.4f} "
    f"epochs={out['epochs_run']} time={out['train_time_s']:.1f}s params={model.n_params()}")
save_json({"hparams": hp, "features": b.features, "val": out["val"], "history": model.history,
           "n_params": model.n_params(), "seed": cfg["global_seed"]}, "baseline.json")
