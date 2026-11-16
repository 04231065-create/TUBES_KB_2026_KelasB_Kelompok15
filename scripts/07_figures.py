"""Membuat semua figur PNG untuk laporan (artifacts/figures/)."""
import numpy as np
from _common import ART, Logger, load_json

from occupancy import fuzzy, viz
from occupancy.data import DataBundle, load_config

log = Logger("07_figures")
FIG = ART / "figures"
cfg = load_config()
b = DataBundle(cfg)
eda, ga = load_json("eda.json"), load_json("ga_results.json")
pa, pg = load_json("pso_all.json"), load_json("pso_ga.json")
comp, fe = load_json("comparison.json"), load_json("fuzzy_eval.json")


def save(fig, name):
    fig.savefig(FIG / name, dpi=170)
    viz.plt.close(fig)
    log(f"saved {name}")


save(viz.architecture(), "architecture.png")
save(viz.class_distribution(eda["class_counts"]), "class_distribution.png")
save(viz.correlation_heatmap(eda["corr_matrix"], eda["corr_labels"]), "correlation.png")
big = max(b.df["session"].unique(), key=lambda s: (b.df["session"] == s).sum())
g = b.df[b.df["session"] == big]
save(viz.sensor_timeseries(g, ["S1_Temp", "S1_Light", "S1_Sound", "S5_CO2", "S6_PIR"],
                           title=f"Contoh deret sensor - sesi {big}"), "sensor_timeseries.png")
save(viz.ga_convergence(ga), "ga_convergence.png")
save(viz.ga_frequency(ga), "ga_frequency.png")
save(viz.pso_convergence([pa, pg], ["PSO+NN (16 sensor)", "GA+PSO+NN"]), "pso_convergence.png")
save(viz.pso_particles(pg, "GA+PSO+NN"), "pso_particles.png")
for n, r in comp.items():
    save(viz.training_curves(r["selected_model"]["history"], f"({r['code']}. {r['label']})"), f"curves_{n}.png")
    save(viz.confusion(r["selected_model"]["test"]["confusion_matrix"],
                       title=f"Confusion matrix test - {r['code']}. {r['label']}"), f"cm_{n}.png")
save(viz.comparison_bars(comp, "f1_macro", "test"), "comparison_f1_test.png")
save(viz.comparison_bars(comp, "accuracy", "test"), "comparison_acc_test.png")
save(viz.memberships(), "fuzzy_memberships.png")
ex = fuzzy.infer(2.0, 900, 26.2)
save(viz.memberships({"occupancy": 2.0, "co2": 900, "temperature": 26.2, "hvac": ex["hvac_percent"]}),
     "fuzzy_example_memberships.png")
save(viz.fuzzy_output(ex), "fuzzy_example_output.png")
tp = comp["ga_pso_rnn"]["test_predictions"]
rows = np.array(tp["row_index"])
ts = b.df["timestamp"].to_numpy()[rows]
sess = b.df["session"].to_numpy()[rows]
# pilih sesi test yang paling banyak variasi kelas
best_s = max(np.unique(sess), key=lambda s: len(np.unique(np.array(tp["y_true"])[sess == s])) * 1e6 +
             (sess == s).sum())
m = sess == best_s
save(viz.test_timeline(ts[m], np.array(tp["y_true"])[m], np.array(tp["y_pred"])[m],
                       np.array(fe["series"]["hvac_pred"])[m],
                       title=f"Test set sesi {best_s}: aktual vs prediksi (D) & HVAC fuzzy"), "test_timeline.png")
cm = np.array(fe["decision_confusion_true_vs_pred"])
save(viz.confusion(cm, fuzzy.OUT_LEVELS, "Keputusan HVAC: dari occupancy aktual vs prediksi"),
     "fuzzy_decision_cm.png")
