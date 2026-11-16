"""Dashboard Streamlit - Smart Room Occupancy Estimation (Kelas B, Kelompok 15).

Jalankan:  streamlit run app.py
Dashboard hanya melakukan INFERENCE dengan artefak di models/ dan menampilkan log
eksperimen di artifacts/. Optimasi GA/PSO dan training dilakukan offline (scripts/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from occupancy import fuzzy, viz  # noqa: E402
from occupancy.data import DataBundle, load_config  # noqa: E402
from occupancy.pipeline import TEMP_COLS, OccupancyPipeline  # noqa: E402

st.set_page_config(page_title="Smart Room Occupancy - Kelompok 15", page_icon="🏠", layout="wide")

MEMBERS = [
    ("04231011", "Arief Rahmansyah Firdaus"),
    ("04231018", "Davin Oentario Adytiawarman"),
    ("04231065", "Musta'in Yusuf"),
    ("04231063", "Muhammad Rozul Faza"),
]
MODEL_NAMES = {"baseline_rnn": "A. Baseline NN", "ga_rnn": "B. GA + NN", "pso_rnn": "C. PSO + NN",
               "ga_pso_rnn": "D. GA + PSO + NN (final)"}


# ------------------------------------------------------------------ loaders
@st.cache_data
def art(name: str):
    with open(ROOT / "artifacts" / name, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def bundle():
    return DataBundle(load_config())


@st.cache_resource
def pipe(name: str = "ga_pso_rnn"):
    return OccupancyPipeline(name)


def show(fig):
    st.pyplot(fig, use_container_width=True)
    viz.plt.close(fig)



STYLE = """
<style>
[data-testid="stSidebar"] {background: linear-gradient(180deg, #eaf2fd 0%, #f6f9fe 100%); border-right: 1px solid #d7e4f7;}
[data-testid="stSidebar"] h1 {color: #14213d;}
.kb-banner {background: linear-gradient(120deg, #14213d 0%, #1c5cab 65%, #2a78d6 100%); color: #f7f9fc;
  padding: 26px 32px; border-radius: 16px; margin: 4px 0 22px 0; box-shadow: 0 4px 14px rgba(20,33,61,.18);}
.kb-banner .kb-eyebrow {font-size: .82rem; letter-spacing: .14em; text-transform: uppercase; color: #a9c9f2; font-weight: 600;}
.kb-banner .kb-title {font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 4px 0 2px 0; color: #ffffff;}
.kb-banner .kb-sub {font-size: 1.02rem; color: #dbe7f8; margin-top: 6px;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e3e8f0; border-left: 5px solid #2a78d6;
  border-radius: 12px; padding: 12px 16px; box-shadow: 0 1px 4px rgba(20,33,61,.06);}
[data-testid="stMetricValue"] {color: #14213d;}
h2, h3, h4 {color: #14213d;}
.stTabs [aria-selected="true"] {color: #1c5cab !important;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def banner(title: str, subtitle: str = "", eyebrow: str = "Kelompok 15 · Smart Room Occupancy"):
    sub = f'<div class="kb-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="kb-banner"><div class="kb-eyebrow">{eyebrow}</div>'
                f'<div class="kb-title">{title}</div>{sub}</div>', unsafe_allow_html=True)


cfg = load_config()
st.sidebar.title("🏠 Smart Room Occupancy")
st.sidebar.caption("Kecerdasan Buatan TE2514031 · Kelas B · Kelompok 15")
page = st.sidebar.radio("Halaman", [
    "1. Overview", "2. Dataset Explorer", "3. GA Optimization", "4. PSO Optimization",
    "5. Neural Network", "6. Fuzzy Inference", "7. Prediction / Simulation", "8. Model Comparison"])
st.sidebar.markdown("---")
st.sidebar.caption("Pipeline: UCI data → preprocessing → GA (seleksi sensor) → PSO (hyperparameter) "
                   "→ RNN (prediksi) → Fuzzy (rekomendasi HVAC)")

# ================================================================== 1. OVERVIEW
if page.startswith("1"):
    banner("Smart Room Occupancy Estimation", "Hybrid Intelligent System: GA + PSO + RNN + Fuzzy Logic · "
           "Kecerdasan Buatan TE2514031 · Kelas B", eyebrow="Tugas Besar · UAS")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
**Problem.** Sistem HVAC gedung sering menyala penuh walaupun ruangan kosong. Jika jumlah penghuni
dapat diperkirakan dari sensor lingkungan yang murah dan tidak invasif (suhu, cahaya, suara, CO₂, PIR),
HVAC dapat diatur sesuai kebutuhan sehingga energi lebih hemat tanpa mengorbankan kenyamanan.

**Tujuan.**
1. Mengklasifikasikan jumlah penghuni ruangan (0, 1, 2, 3 orang) dari jendela data sensor dengan **RNN**.
2. Memilih subset sensor yang paling informatif dengan **Genetic Algorithm**.
3. Menala hyperparameter RNN dengan **Particle Swarm Optimization**.
4. Mengubah prediksi menjadi **rekomendasi HVAC** (Minimum · Low · Medium · High) dengan **Fuzzy Logic**.

**Dataset.** [Room Occupancy Estimation – UCI ID 864](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)
(Singh & Chaudhari, 2018; CC BY 4.0). Ruang 6 m × 4.6 m, 7 node sensor, sampling 30 detik,
target `Room_Occupancy_Count` (0–3).
""")
    with c2:
        st.markdown("**Anggota Kelompok 15**")
        st.table(pd.DataFrame(MEMBERS, columns=["NIM", "Nama"]))
        st.markdown("**Metrik utama:** macro-F1 (kelas tidak seimbang), didukung accuracy, "
                    "precision, recall, confusion matrix.")
    st.markdown("#### Diagram sistem")
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6", fontname="Helvetica", fontsize=11];
  data [label="UCI 864\\n16 sensor, 30 s"]; pre [label="Preprocessing\\nsplit kronologis 60/20/20\\nscaler fit di train\\nwindowing T langkah"];
  ga [label="GA\\nseleksi sensor\\nkromosom 16-bit", fillcolor="#fdeee7", color="#eb6834"];
  pso [label="PSO\\nhidden, lr, T,\\ndropout, batch", fillcolor="#e6f6f0", color="#1baf7a"];
  rnn [label="Elman RNN\\nP(0..3 orang)"]; fis [label="Fuzzy Mamdani\\nOccupancy + CO2 + Suhu", fillcolor="#fdf4e0", color="#eda100"];
  out [label="Rekomendasi HVAC\\nMinimum/Low/Medium/High"]; dash [label="Streamlit\\ndashboard", shape=component];
  data->pre->ga->pso->rnn->fis->out->dash; pre->rnn [style=dashed, label="fitur terpilih"]; }
""")
    comp = art("comparison.json")
    fe = art("fuzzy_eval.json")
    d = comp["ga_pso_rnn"]["summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sensor dipakai (GA)", f"{comp['ga_pso_rnn']['n_features']} / 16")
    m2.metric("Test macro-F1 (D)", f"{d['f1_macro']['test']['mean']:.4f}")
    m3.metric("Test accuracy (D)", f"{d['accuracy']['test']['mean']:.4f}")
    m4.metric("Kesesuaian keputusan HVAC", f"{fe['decision_agreement']:.1%}")

# ================================================================== 2. DATASET
elif page.startswith("2"):
    banner("Dataset Explorer", "UCI Room Occupancy Estimation (ID 864) · statistik, split kronologis, korelasi")
    eda = art("eda.json")
    b = bundle()
    df = b.df
    c = st.columns(5)
    c[0].metric("Baris", f"{eda['n_rows']:,}")
    c[1].metric("Fitur sensor", eda["n_sensor_features"])
    c[2].metric("Missing values", sum(eda["missing_values"].values()))
    c[3].metric("Sesi kontinu", eda["n_sessions"])
    c[4].metric("Imbalance (max/min)", f"{eda['imbalance_ratio_max_min']:.1f}×")
    t1, t2, t3, t4 = st.tabs(["Statistik", "Target & split", "Korelasi", "Deret waktu & distribusi"])
    with t1:
        st.dataframe(pd.DataFrame(eda["describe"]).T, use_container_width=True)
        st.write("Missing values per kolom:", pd.Series(eda["missing_values"]).to_frame("missing").T)
        st.caption(f"Periode: {eda['time_start']} s.d. {eda['time_end']} · median sampling "
                   f"{eda['sampling_median_s']:.0f} detik · duplikat timestamp: {eda['duplicated_rows']}")
    with t2:
        show(viz.class_distribution(eda["class_counts"]))
        rows = []
        for sp, v in eda["split_counts"].items():
            rows.append({"split": sp, "n": v["n"], **{f"kelas {k}": v["class_counts"].get(str(k), 0)
                                                      for k in range(4)}, "mulai": v["start"], "akhir": v["end"]})
        st.markdown("**Split kronologis per sesi (60/20/20)** – tidak ada data masa depan di train.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(eda["sessions"]).drop(columns=["class_counts"]), hide_index=True,
                     use_container_width=True)
    with t3:
        show(viz.correlation_heatmap(eda["corr_matrix"], eda["corr_labels"]))
        st.bar_chart(pd.Series(eda["corr_with_target"]).sort_values(), horizontal=True)
    with t4:
        sess = st.selectbox("Sesi", sorted(df["session"].unique()))
        cols = st.multiselect("Sensor", b.features, default=["S1_Light", "S1_Sound", "S5_CO2", "S6_PIR"])
        g = df[df["session"] == sess]
        if cols:
            show(viz.sensor_timeseries(g, cols[:7], title=f"Sesi {sess}"))
        f = st.selectbox("Distribusi fitur per kelas", b.features, index=b.features.index("S5_CO2"))
        st.dataframe(df.groupby(cfg["target"])[f].describe(), use_container_width=True)

# ================================================================== 3. GA
elif page.startswith("3"):
    banner("Genetic Algorithm – Seleksi Sensor", "Kromosom 16-bit · fitness CV forward-chaining · 5 seed")
    ga = art("ga_results.json")
    gc = ga["config"]
    st.latex(r"F(c)=\overline{\big[\mathrm{macroF1}_{val,k}-\mathrm{logloss}_{val,k}\big]}_{fold\,k}\big(\mathrm{RNN}(c)\big)-\lambda\frac{\sum_i c_i}{16},"
             rf"\quad \lambda={gc['lambda_size']}")
    st.markdown("Kromosom = vektor biner 16 gen (1 = sensor dipakai). RNN fitness: hidden "
                f"{gc['fitness_model']['hidden']}, T={gc['fitness_model']['seq_len']}, "
                f"maks {gc['fitness_model']['max_epochs']} epoch, seed training tetap → fitness deterministik. "
                "Skor dirata-rata atas **3 fold CV forward-chaining** di periode train+val "
                f"({', '.join(f'[0,{p:.0%})→[{p:.0%},{q:.0%})' for p, q in cfg['cv_folds'])} dari setiap sesi). "
                "Log-loss dipakai karena macro-F1 validasi cepat jenuh (≈1.0). "
                "Hanya data **train + validation**; test tidak pernah dipakai.")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.table(pd.DataFrame({"parameter": ["populasi", "generasi maks", "seleksi", "crossover",
                                             "mutasi", "elitism", "stopping", "seed"],
                               "nilai": [gc["pop_size"], gc["generations"], f"tournament k={gc['tournament_k']}",
                                         f"uniform, pc={gc['p_crossover']}", f"bit-flip, pm={gc['p_mutation']}",
                                         gc["elitism"], f"tanpa perbaikan {gc['patience']} generasi",
                                         ", ".join(map(str, gc["seeds"]))]}))
    with c2:
        s = ga["stats"]
        st.metric("Best fitness (best / mean ± std)",
                  f"{s['best_fitness_best']:.4f}", f"{s['best_fitness_mean']:.4f} ± {s['best_fitness_std']:.4f}")
        st.write(f"Total evaluasi fitness: **{s['total_evals']}** · waktu total {s['total_time_s']:.0f} s")
    show(viz.ga_convergence(ga))
    show(viz.ga_frequency(ga))
    rows = [{"seed": r["seed"], "kromosom": r["best_chromosome"], "fitness": round(r["best_fitness"], 4),
             "F1 val": round(r["best_info"]["f1"], 4), "n fitur": r["best_info"]["n_features"],
             "evaluasi": r["n_evals"], "fitur": ", ".join(r["selected_features"])} for r in ga["runs"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    fin = ga["final_selection"]
    sel = set(fin["features"])
    c1, c2 = st.columns(2)
    c1.success("**Terpilih (" + str(len(sel)) + "):** " + ", ".join(fin["features"]))
    c2.warning("**Dieliminasi:** " + ", ".join([f for f in ga["features"] if f not in sel]))
    st.caption(f"Aturan pemilihan subset final: {fin['rule']} (seed {fin['seed']}).")

# ================================================================== 4. PSO
elif page.startswith("4"):
    banner("Particle Swarm Optimization – Hyperparameter RNN", "5 dimensi: hidden, learning rate, T, dropout, batch · 3 seed per run")
    pa, pg = art("pso_all.json"), art("pso_ga.json")
    pc = pa["config"]
    st.latex(r"v_{i}^{t+1}=w\,v_i^t+c_1r_1(p_i-x_i^t)+c_2r_2(g-x_i^t),\qquad x_i^{t+1}=x_i^t+v_i^{t+1}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Search space (partikel 5 dimensi)**")
        st.table(pd.DataFrame([{"dimensi": k, "batas bawah": v[0], "batas atas": v[1]}
                               for k, v in pc["bounds"].items()]))
    with c2:
        st.markdown("**Parameter swarm**")
        st.table(pd.DataFrame({"parameter": ["swarm", "iterasi maks", "w", "c1, c2", "v_max", "stopping",
                                             "fitness", "seed"],
                               "nilai": [pc["swarm_size"], pc["iterations"], f"{pc['w_start']} → {pc['w_end']} (linear)",
                                         f"{pc['c1']}, {pc['c2']}", f"{pc['vmax_frac']}·(ub−lb)",
                                         f"gbest stagnan {pc['patience']} iterasi", "mean fold CV (macro-F1 − log-loss)",
                                         ", ".join(map(str, pc["seeds"]))]}))
    show(viz.pso_convergence([pa, pg], ["PSO+NN (16 sensor)", "GA+PSO+NN"]))
    which = st.radio("Detail partikel", ["PSO+NN", "GA+PSO+NN"], horizontal=True)
    show(viz.pso_particles(pa if which == "PSO+NN" else pg, which))
    rows = []
    for tag, p in [("PSO+NN", pa), ("GA+PSO+NN", pg)]:
        for r in p["runs"]:
            rows.append({"run": tag, "seed": r["seed"], "gbest F1 val": round(r["gbest_fitness"], 4),
                         **r["gbest_hparams"], "evaluasi": r["n_evals"], "waktu (s)": round(r["time_s"], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    hs = lambda h: (f"hidden={h['hidden']}, lr={h['lr']:.5f}, T={h['seq_len']}, "  # noqa: E731
                    f"dropout={h['dropout']:.3f}, batch={h['batch_size']}")
    c1.success(f"**Hyperparameter terbaik PSO+NN:** {hs(pa['final']['hparams'])} "
               f"(gbest mean±std {pa['stats']['gbest_mean']:.4f} ± {pa['stats']['gbest_std']:.4f})")
    c2.success(f"**Hyperparameter terbaik GA+PSO+NN:** {hs(pg['final']['hparams'])} "
               f"(gbest mean±std {pg['stats']['gbest_mean']:.4f} ± {pg['stats']['gbest_std']:.4f})")

# ================================================================== 5. NN
elif page.startswith("5"):
    banner("Neural Network – Elman RNN", "Ditulis dari nol dengan NumPy · BPTT + Adam · early stopping")
    comp = art("comparison.json")
    name = st.selectbox("Model", list(MODEL_NAMES), index=3, format_func=MODEL_NAMES.get)
    r = comp[name]
    hp = r["hparams"]
    F, H = r["n_features"], hp["hidden"]
    st.markdown(f"""
**Arsitektur:** input jendela `T={hp['seq_len']}` langkah × `{F}` sensor →
`h_t = tanh(x_t·W_x + h_(t−1)·W_h + b_h)` dengan **{H} hidden units** → dropout {hp['dropout']:.2f} pada `h_T`
→ Dense(4) + softmax → P(0,1,2,3 orang).
Jumlah parameter: **{r['n_params']:,}** = {F}×{H} + {H}×{H} + {H} + {H}×4 + 4.
Optimizer Adam (lr={hp['lr']:.5f}), batch {hp['batch_size']}, weighted cross-entropy (bobot kelas dari train),
L2 {cfg['train']['l2']}, gradient clipping {cfg['train']['clip']}, early stopping (patience {cfg['train']['patience']}) pada validation loss.

*Alasan RNN:* sensor adalah deret waktu 30 detik; kenaikan CO₂, cahaya & suara saat orang masuk bersifat
temporal sehingga model perlu membaca beberapa langkah sebelumnya.""")
    sm = r["selected_model"]
    show(viz.training_curves(sm["history"], f"({MODEL_NAMES[name]}, seed {r['selected_seed']})"))
    c1, c2 = st.columns([1, 1])
    with c1:
        show(viz.confusion(sm["test"]["confusion_matrix"]))
    with c2:
        t = sm["test"]
        st.dataframe(pd.DataFrame({"kelas": viz.CLASS_LABELS, "precision": t["precision_per_class"],
                                   "recall": t["recall_per_class"], "F1": t["f1_per_class"],
                                   "support": t["support"]}).round(4), hide_index=True, use_container_width=True)
        s = r["summary"]
        st.dataframe(pd.DataFrame({k: {"val (mean±std)": f"{s[k]['val']['mean']:.4f} ± {s[k]['val']['std']:.4f}",
                                       "test (mean±std)": f"{s[k]['test']['mean']:.4f} ± {s[k]['test']['std']:.4f}"}
                                   for k in s}).T, use_container_width=True)
        st.caption("mean ± std dari 5 seed training; plot & confusion matrix dari seed dengan skor validasi (F1 − log-loss) terbaik.")

# ================================================================== 6. FUZZY
elif page.startswith("6"):
    banner("Fuzzy Inference System – Rekomendasi HVAC", "Mamdani · 3 input · 27 aturan · defuzzifikasi centroid")
    st.markdown("Mamdani · AND = min · implikasi = min · agregasi = max · defuzzifikasi = **centroid**. "
                "Rule disusun dari: `level = base(Occupancy) + adj(CO2) + adj(Suhu)` (Empty=0, Few=1, Full=2; "
                "CO2 High=+1; Suhu Cool=−1, Warm=+1), di-clip ke Minimum…High.")
    c = st.columns(3)
    o = c[0].slider("Occupancy (perkiraan orang)", 0.0, 3.0, 1.5, 0.05)
    co2 = c[1].slider("CO₂ (ppm)", 300, 1500, 700, 10)
    tmp = c[2].slider("Suhu rata-rata (°C)", 18.0, 32.0, 25.5, 0.1)
    res = fuzzy.infer(o, co2, tmp)
    m1, m2 = st.columns(2)
    m1.metric("HVAC level", f"{res['hvac_percent']:.1f} %")
    m2.metric("Keputusan", res["decision"])
    show(viz.memberships({"occupancy": o, "co2": co2, "temperature": tmp, "hvac": res["hvac_percent"]}))
    show(viz.fuzzy_output(res))
    st.info(fuzzy.explain(res, o, co2, tmp))
    st.markdown("**Derajat keanggotaan input**")
    st.dataframe(pd.DataFrame(res["membership_in"]).T.round(3), use_container_width=True)
    st.markdown("**Rule base (27 aturan)** – aturan yang aktif ditandai α > 0")
    act = {f["rule"]: f["strength"] for f in res["fired_rules"]}
    rb = pd.DataFrame(fuzzy.RULES)
    rb.insert(0, "R", range(1, len(rb) + 1))
    rb["α"] = [round(act.get(i, 0.0), 3) for i in rb["R"]]
    st.dataframe(rb, hide_index=True, use_container_width=True, height=300)
    fe = art("fuzzy_eval.json")
    st.markdown("**Validasi keputusan**")
    ch = fe["rule_check"]
    st.write(f"Rule lengkap: {ch['complete']} · konflik: {ch['conflicts']} · pelanggaran monoton label: "
             f"{ch['label_monotonic_violations']}/{ch['monotonic_checks']} · penurunan crisp maks "
             f"{ch['max_crisp_drop_pct']}% (efek centroid di zona transisi).")
    st.dataframe(pd.DataFrame(fe["scenarios"]).drop(columns=["top_rule"]), hide_index=True,
                 use_container_width=True)
    st.caption("Keputusan fuzzy bukan ukuran akurasi NN; ia diuji lewat konsistensi rule, skenario ekstrem, "
               "dan kesesuaian keputusan dari occupancy prediksi vs occupancy sebenarnya.")

# ================================================================== 7. PREDICTION
elif page.startswith("7"):
    banner("Prediction / Simulation", "Input manual, jendela test, atau upload CSV → RNN → keputusan HVAC")
    name = st.selectbox("Model", list(MODEL_NAMES), index=3, format_func=MODEL_NAMES.get)
    p = pipe(name)
    st.caption(f"Model memakai {len(p.features)} sensor: {', '.join(p.features)} · panjang jendela T = {p.T}")
    tab1, tab2, tab3 = st.tabs(["Input manual", "Pilih jendela dari test set", "Upload CSV"])
    b = bundle()
    with tab1:
        st.markdown("Nilai sensor dianggap **konstan selama T langkah** (kondisi tunak). "
                    "Sensor yang tidak dipilih GA (tanpa ✓) tidak memengaruhi prediksi.")
        st.caption("Preset = rata-rata data train per kelas (PIR dibulatkan ke 0/1). Untuk kasus dinamis, "
                   "misalnya saat orang masuk/keluar, gunakan tab 'Pilih jendela dari test set'.")
        presets = {"Ruang kosong": 0, "1 orang": 1, "2 orang": 2, "3 orang": 3}
        pr = st.radio("Isi otomatis dari rata-rata kelas (data train)", list(presets), horizontal=True)
        tr = b.df[(b.df["split"] == "train") & (b.df[cfg["target"]] == presets[pr])]
        med = tr[b.features].mean()
        vals = {}
        cols = st.columns(4)
        for i, f in enumerate(b.features):
            lo, hi = float(b.df[f].min()), float(b.df[f].max())
            if f.endswith("PIR"):
                vals[f] = cols[i % 4].selectbox(f, [0, 1], index=int(round(med[f])), key=f"{pr}{f}")
            else:
                pad = 0.1 * (hi - lo) if hi > lo else 1.0
                vals[f] = cols[i % 4].number_input(f + (" ✓" if f in p.features else ""),
                                                   float(lo - pad), float(hi + pad), float(med[f]),
                                                   key=f"{pr}{f}", format="%.3f")
        out = p.predict_steady(vals)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediksi occupancy", f"{out['pred']} orang", f"conf {out['prob'].max():.1%}")
        c2.metric("Expected occupancy", f"{out['expected_occupancy']:.2f}")
        c3.metric("Rekomendasi HVAC", out["fuzzy"]["decision"], f"{out['fuzzy']['hvac_percent']:.1f} %")
        st.bar_chart(pd.Series(out["prob"], index=viz.CLASS_LABELS, name="probabilitas"))
        st.info(out["explanation"])
    with tab2:
        test = b.df[b.df["split"] == "test"]
        sess_opts = sorted(test["session"].unique())
        rich = max(sess_opts, key=lambda s: (test.loc[test["session"] == s, cfg["target"]].nunique(),
                                             (test["session"] == s).sum()))
        sess = st.selectbox("Sesi (bagian test)", sess_opts, index=sess_opts.index(rich))
        g = test[test["session"] == sess]
        if len(g) < p.T:
            st.warning("Sesi terlalu pendek.")
        else:
            pos = st.slider("Indeks akhir jendela", p.T - 1, len(g) - 1, min(len(g) - 1, p.T + 60))
            win = g.iloc[pos - p.T + 1:pos + 1]
            res = p.predict_df(win).iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Waktu", str(win["timestamp"].iloc[-1]))
            c2.metric("Aktual", f"{int(win[cfg['target']].iloc[-1])} orang")
            c3.metric("Prediksi", f"{int(res['pred_occupancy'])} orang", f"conf {res['confidence']:.1%}")
            c4.metric("HVAC", res["hvac_decision"], f"{res['hvac_percent']:.1f} %")
            st.line_chart(win.set_index("timestamp")[[f for f in ["S1_Light", "S5_CO2", "S1_Sound"]]])
            full = p.predict_df(g)
            ts = g["timestamp"].iloc[full["row"]].to_numpy()
            show(viz.test_timeline(ts, g[cfg["target"]].iloc[full["row"]].to_numpy(),
                                   full["pred_occupancy"].to_numpy(), full["hvac_percent"].to_numpy(),
                                   title=f"Sesi {sess} (test): aktual vs prediksi & HVAC"))
    with tab3:
        st.markdown("Unggah CSV berisi kolom 16 sensor (format UCI, urut waktu, ≥ T baris). "
                    "Contoh: `data/sample_input.csv`.")
        up = st.file_uploader("CSV sensor", type=["csv"])
        use_sample = st.checkbox("Pakai sample_input.csv", value=up is None)
        dfu = None
        if up is not None:
            dfu = pd.read_csv(up)
        elif use_sample:
            dfu = pd.read_csv(ROOT / "data" / "sample_input.csv")
        if dfu is not None:
            try:
                res = p.predict_df(dfu)
                st.dataframe(res.round(3), hide_index=True, use_container_width=True, height=280)
                st.line_chart(res.set_index("row")[["expected_occupancy"]])
                st.line_chart(res.set_index("row")[["hvac_percent"]])
                st.write(res["hvac_decision"].value_counts())
                st.download_button("Unduh hasil (CSV)", res.to_csv(index=False), "prediksi_hvac.csv",
                                   "text/csv")
            except ValueError as e:
                st.error(str(e))

# ================================================================== 8. COMPARISON
else:
    banner("Model Comparison", "Baseline vs GA+NN vs PSO+NN vs GA+PSO+NN · mean ± std 5 seed")
    comp = art("comparison.json")
    fe = art("fuzzy_eval.json")
    rows = []
    for n, r in comp.items():
        s = r["summary"]
        rows.append({"model": f"{r['code']}. {r['label']}", "n fitur": r["n_features"],
                     "hidden": r["hparams"]["hidden"], "lr": round(r["hparams"]["lr"], 5),
                     "T": r["hparams"]["seq_len"], "dropout": round(r["hparams"]["dropout"], 3),
                     "batch": r["hparams"]["batch_size"], "params": r["n_params"],
                     "val F1": f"{s['f1_macro']['val']['mean']:.4f} ± {s['f1_macro']['val']['std']:.4f}",
                     "test F1": f"{s['f1_macro']['test']['mean']:.4f} ± {s['f1_macro']['test']['std']:.4f}",
                     "test acc": f"{s['accuracy']['test']['mean']:.4f} ± {s['accuracy']['test']['std']:.4f}",
                     "test prec": f"{s['precision_macro']['test']['mean']:.4f}",
                     "test rec": f"{s['recall_macro']['test']['mean']:.4f}",
                     "waktu train (s)": round(r["train_time_mean_s"], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Semua konfigurasi: split, epoch maksimum, patience, dan 5 seed training yang sama. "
               "mean ± std dari 5 seed.")
    metric = st.selectbox("Metrik", ["f1_macro", "accuracy", "precision_macro", "recall_macro"])
    split = st.radio("Split", ["test", "val"], horizontal=True)
    show(viz.comparison_bars(comp, metric, split))
    st.markdown("**E. GA + PSO + NN + Fuzzy (sistem final)**")
    c = st.columns(4)
    c[0].metric("Keputusan sama vs occupancy aktual", f"{fe['decision_agreement']:.1%}")
    c[1].metric("Selisih ≤ 1 level", f"{fe['decision_within_one_level']:.1%}")
    c[2].metric("Under-ventilation", f"{fe['under_ventilation_rate']:.1%}")
    c[3].metric("Rata-rata HVAC", f"{fe['mean_hvac_pred']:.1f} %")
    st.caption("Model E tidak menambah akurasi klasifikasi; fuzzy adalah lapisan keputusan. Proksi energi: "
               f"rata-rata level HVAC {fe['mean_hvac_pred']:.1f}% vs 100% bila HVAC selalu High.")
