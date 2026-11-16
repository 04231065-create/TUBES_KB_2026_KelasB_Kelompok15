"""Plot matplotlib yang dipakai bersama oleh dashboard Streamlit dan laporan PDF."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from . import fuzzy  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5",
                                                     "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_br", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b8302f"])
CLASS_LABELS = ["0 orang", "1 orang", "2 orang", "3 orang"]

HI_TEXT = "white"  # warna teks di sel heatmap yang pekat


def _apply_rc():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "grid.color": GRID, "axes.titlecolor": INK, "text.color": INK, "legend.labelcolor": INK2,
    })


def use_dark():
    """Tema gelap untuk dashboard (laporan PDF tetap memakai tema terang)."""
    global SURFACE, INK, INK2, GRID, SERIES, SEQ, DIV, HI_TEXT
    SURFACE, INK, INK2, GRID = "#0d1526", "#e6edf7", "#9fb3d1", "#1f2c45"
    SERIES = ["#4c9bf0", "#f07a45", "#2cc590", "#f0b429", "#e77fb0", "#3fbf3f", "#9a8cf0", "#ef6b6b"]
    SEQ = LinearSegmentedColormap.from_list("seq_dark", ["#111c33", "#16305c", "#1c5cab", "#3987e5",
                                                         "#86b6ef", "#d5e7fd"])
    DIV = LinearSegmentedColormap.from_list("div_dark", ["#3987e5", "#1f4f8f", "#1b2232", "#8f2f2e", "#e05a59"])
    HI_TEXT = "#0b1220"
    _apply_rc()


plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 110,
})


def _fig(w=6.4, h=3.4):
    return plt.subplots(figsize=(w, h), constrained_layout=True)


def architecture():
    """Diagram pipeline hybrid (untuk laporan)."""
    fig, ax = plt.subplots(figsize=(9.0, 2.6), constrained_layout=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    boxes = [
        (1.0, "UCI 864\n16 sensor\n30 detik", "#eef4fc", SERIES[0]),
        (15.3, "Preprocessing\nsplit kronologis\nscaler (train)\nwindow T", "#eef4fc", SERIES[0]),
        (29.6, "GA\nseleksi sensor\n16-bit", "#fdeee7", SERIES[1]),
        (43.9, "PSO\nhidden, lr, T,\ndropout, batch", "#e6f6f0", SERIES[2]),
        (58.2, "Elman RNN\nP(0..3 orang)", "#eef4fc", SERIES[0]),
        (72.5, "Fuzzy Mamdani\nOcc + CO2 + Suhu", "#fdf4e0", SERIES[3]),
        (86.8, "HVAC\nMin/Low/\nMed/High", "#eef4fc", INK2),
    ]
    w = 10.6
    for x, t, fc, ec in boxes:
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 6), w, 18, boxstyle="round,pad=0.4",
                                                       fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, 15, t, ha="center", va="center", fontsize=7.5, color=INK)
    for (x, *_), (x2, *_) in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(x2 - 0.6, 15), xytext=(x + w + 0.6, 15),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
    ax.text(50, 1.5, "Streamlit dashboard: Overview · Dataset · GA · PSO · NN · Fuzzy · Prediction · Comparison",
            ha="center", fontsize=8, color=INK2)
    return fig


def class_distribution(counts: dict):
    fig, ax = _fig(5.2, 3.0)
    k = [int(x) for x in counts]
    v = [counts[x] for x in counts]
    bars = ax.bar([CLASS_LABELS[i] for i in k], v, color=SERIES[0], width=0.6, edgecolor=SURFACE,
                  linewidth=2)
    tot = sum(v)
    for b_, val in zip(bars, v):
        ax.text(b_.get_x() + b_.get_width() / 2, val, f"{val}\n({val / tot:.1%})", ha="center",
                va="bottom", fontsize=8, color=INK2)
    ax.set_ylabel("Jumlah sampel")
    ax.set_title("Distribusi target Room_Occupancy_Count")
    ax.set_ylim(0, max(v) * 1.22)
    ax.grid(axis="x", visible=False)
    return fig


def correlation_heatmap(corr, labels):
    corr = np.asarray(corr)
    fig, ax = _fig(6.8, 5.6)
    im = ax.imshow(corr, cmap=DIV, vmin=-1, vmax=1)
    short = [l.replace("Room_Occupancy_Count", "Occupancy") for l in labels]
    ax.set_xticks(range(len(labels)), short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), short, fontsize=7)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if abs(corr[i, j]) >= 0.5 and i != j:
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(corr[i, j]) > 0.75 else INK)  # DIV gelap: ujung tetap pekat
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Matriks korelasi sensor & target")
    return fig


def sensor_timeseries(df, cols, target="Room_Occupancy_Count", title=None):
    fig, axes = plt.subplots(len(cols) + 1, 1, figsize=(7.2, 1.25 * (len(cols) + 1) + 0.4),
                             sharex=True, constrained_layout=True)
    for ax, c, colr in zip(axes, cols, SERIES):
        ax.plot(df["timestamp"], df[c], color=colr, linewidth=1.2)
        ax.set_ylabel(c, fontsize=7, rotation=0, ha="right", va="center")
    axes[-1].step(df["timestamp"], df[target], color=INK, linewidth=1.2, where="post")
    axes[-1].set_ylabel("Occupancy", fontsize=7, rotation=0, ha="right", va="center")
    axes[-1].set_yticks([0, 1, 2, 3])
    if title:
        axes[0].set_title(title)
    fig.autofmt_xdate()
    return fig


def ga_convergence(ga: dict):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    for i, r in enumerate(ga["runs"]):
        h = r["history"]
        g = [x["generation"] for x in h]
        axes[0].plot(g, [x["best_fitness"] for x in h], color=SERIES[i % 8], label=f"seed {r['seed']}")
        axes[1].plot(g, [x["mean_fitness"] for x in h], color=SERIES[i % 8], label=f"seed {r['seed']}")
    axes[0].set_title("GA: best fitness per generasi")
    axes[1].set_title("GA: rata-rata fitness populasi")
    for ax in axes:
        ax.set_xlabel("Generasi")
        ax.set_ylabel("Fitness")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(fontsize=7)
    return fig


def ga_frequency(ga: dict):
    feats = ga["features"]
    freq = [ga["selection_frequency"][f] for f in feats]
    final = set(ga["final_selection"]["features"])
    fig, ax = _fig(7.2, 3.0)
    colors = [SERIES[0] if f in final else "#c9c8c2" for f in feats]
    ax.bar(feats, freq, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Frekuensi terpilih")
    ax.set_title("Frekuensi fitur terpilih di semua seed GA (biru = subset final)")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.grid(axis="x", visible=False)
    return fig


def pso_convergence(pso_list: list[dict], titles: list[str]):
    fig, axes = plt.subplots(1, len(pso_list), figsize=(4.2 * len(pso_list), 3.2),
                             constrained_layout=True, squeeze=False)
    for ax, pso, t in zip(axes[0], pso_list, titles):
        for i, r in enumerate(pso["runs"]):
            h = r["history"]
            ax.plot([x["iteration"] for x in h], [x["gbest_fitness"] for x in h], color=SERIES[i % 8],
                    marker="o", markersize=4, label=f"seed {r['seed']}")
        ax.set_title(f"PSO gbest - {t}")
        ax.set_xlabel("Iterasi")
        ax.set_ylabel("gbest (skor CV: F1 − log-loss)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=7)
    return fig


def pso_particles(pso: dict, title: str):
    r = max(pso["runs"], key=lambda x: x["gbest_fitness"])
    P = r["particles"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    it = np.array([p["iteration"] for p in P])
    f = np.array([p["fitness"] for p in P])
    axes[0].scatter(it + np.random.default_rng(0).uniform(-0.15, 0.15, len(it)), f, s=14,
                    color=SERIES[0], alpha=0.7)
    axes[0].set_xlabel("Iterasi")
    axes[0].set_ylabel("Fitness partikel (skor CV)")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_title(f"Sebaran fitness partikel ({title}, seed {r['seed']})")
    sc = axes[1].scatter([p["hidden"] for p in P], [p["log10_lr"] for p in P], c=it, cmap=SEQ, s=16)
    axes[1].set_xlabel("hidden units")
    axes[1].set_ylabel("log10(learning rate)")
    axes[1].set_title("Lintasan partikel (warna = iterasi)")
    fig.colorbar(sc, ax=axes[1], label="iterasi")
    return fig


def training_curves(hist: dict, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), constrained_layout=True)
    e = hist["epoch"]
    axes[0].plot(e, hist["train_loss"], color=SERIES[0], label="train")
    axes[0].plot(e, hist["val_loss"], color=SERIES[1], label="validation")
    axes[0].set_title(f"Loss {title}")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[1].plot(e, hist["train_acc"], color=SERIES[0], label="train acc")
    axes[1].plot(e, hist["val_acc"], color=SERIES[1], label="val acc")
    axes[1].plot(e, hist["val_f1"], color=SERIES[2], label="val macro-F1")
    axes[1].set_title(f"Akurasi / F1 {title}")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=7)
    return fig


def confusion(cm, labels=CLASS_LABELS, title="Confusion matrix (test)", normalize=True):
    cm = np.asarray(cm, float)
    fig, ax = _fig(4.2, 3.6)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    ax.imshow(shown, cmap=SEQ, vmin=0, vmax=1 if normalize else None)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{int(cm[i, j])}\n{shown[i, j]:.0%}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color=HI_TEXT if shown[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, fontsize=7.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(title)
    return fig


def memberships(point: dict | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), constrained_layout=True)
    specs = [("occupancy", fuzzy.INPUTS["occupancy"]), ("co2", fuzzy.INPUTS["co2"]),
             ("temperature", fuzzy.INPUTS["temperature"]), ("hvac", fuzzy.OUTPUT)]
    titles = ["Input: Occupancy (orang)", "Input: CO2 (ppm)", "Input: Temperature (C)",
              "Output: HVAC level (%)"]
    for ax, (name, spec), t in zip(axes.ravel(), specs, titles):
        x = np.linspace(*spec["range"], 600)
        for i, (term, s) in enumerate(spec["terms"].items()):
            ax.plot(x, fuzzy.mf(s, x), color=SERIES[i], label=term)
        if point and name in point:
            ax.axvline(point[name], color=INK, linestyle="--", linewidth=1)
        ax.set_title(t)
        ax.set_ylim(-0.03, 1.08)
        ax.set_ylabel("derajat")
        ax.legend(fontsize=7, loc="center right")
    return fig


def fuzzy_output(res: dict):
    fig, ax = _fig(6.0, 2.8)
    u = res["universe"]
    for i, (term, s) in enumerate(fuzzy.OUTPUT["terms"].items()):
        ax.plot(u, fuzzy.mf(s, u), color=SERIES[i], linewidth=1.2, alpha=0.6, label=term)
    ax.fill_between(u, 0, res["aggregated"], color=SERIES[0], alpha=0.25, label="agregasi")
    ax.axvline(res["hvac_percent"], color=INK, linewidth=2,
               label=f"centroid {res['hvac_percent']:.1f}%")
    ax.set_xlabel("HVAC level (%)")
    ax.set_ylabel("derajat")
    ax.set_title(f"Agregasi Mamdani & defuzzifikasi -> {res['decision']}")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    return fig


def comparison_bars(comp: dict, metric="f1_macro", split="test"):
    names = list(comp)
    labels = [f"{comp[n]['code']}. {comp[n]['label']}" for n in names]
    mean = [comp[n]["summary"][metric][split]["mean"] for n in names]
    std = [comp[n]["summary"][metric][split]["std"] for n in names]
    fig, ax = _fig(6.4, 3.2)
    bars = ax.bar(labels, mean, yerr=std, color=SERIES[0], capsize=4, width=0.6,
                  edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": INK2, "elinewidth": 1})
    for b_, m, s in zip(bars, mean, std):
        ax.text(b_.get_x() + b_.get_width() / 2, m + s + 0.004, f"{m:.4f}", ha="center", fontsize=8,
                color=INK2)
    lo = min(m - s for m, s in zip(mean, std))
    ax.set_ylim(max(0, lo - 0.05), min(1.0, max(mean) + max(std) + 0.03))
    ax.set_ylabel(f"{metric} ({split})")
    ax.set_title(f"Perbandingan 4 konfigurasi - {metric} {split} (mean +- std, 5 seed)")
    ax.grid(axis="x", visible=False)
    return fig


def test_timeline(ts, y_true, y_pred, hvac=None, title="Test set: aktual vs prediksi"):
    n = 3 if hvac is not None else 2
    fig, axes = plt.subplots(n, 1, figsize=(7.6, 1.6 * n + 0.4), sharex=True, constrained_layout=True)
    axes[0].step(ts, y_true, where="post", color=INK, linewidth=1.3, label="aktual")
    axes[0].step(ts, y_pred, where="post", color=SERIES[1], linewidth=1.1, label="prediksi", alpha=0.85)
    axes[0].set_ylabel("orang")
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    axes[0].set_title(title)
    err = (np.asarray(y_pred) != np.asarray(y_true)).astype(int)
    axes[1].fill_between(ts, 0, err, step="post", color="#d03b3b", alpha=0.6)
    axes[1].set_ylabel("salah")
    axes[1].set_yticks([0, 1])
    if hvac is not None:
        axes[2].plot(ts, hvac, color=SERIES[0], linewidth=1.3)
        axes[2].set_ylabel("HVAC %")
        axes[2].set_ylim(0, 100)
    fig.autofmt_xdate()
    return fig
