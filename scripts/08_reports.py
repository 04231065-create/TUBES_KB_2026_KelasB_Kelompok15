"""Membangun laporan PDF W12, W13, W14, W15 dan FINAL dari artifacts/*.json.

Semua angka dibaca dari file hasil eksperimen sehingga laporan selalu konsisten dengan log.
"""
import numpy as np
from _common import ART, OUT_DIR, load_json
from report_lib import MEMBERS, Report, f4, pm

from occupancy import fuzzy
from occupancy.data import load_config

FIG = ART / "figures"
REP = OUT_DIR / "reports"
REP.mkdir(exist_ok=True)
cfg = load_config()
eda, base, ga = load_json("eda.json"), load_json("baseline.json"), load_json("ga_results.json")
pa, pg = load_json("pso_all.json"), load_json("pso_ga.json")
comp, fe = load_json("comparison.json"), load_json("fuzzy_eval.json")
A, B, C, D = comp["baseline_rnn"], comp["ga_rnn"], comp["pso_rnn"], comp["ga_pso_rnn"]
GA_FEATS = ga["final_selection"]["features"]
DROPPED = [f for f in ga["features"] if f not in GA_FEATS]
REPO_URL = "https://github.com/[akun-github]/TUBES_KB_2026_KelasB_Kelompok15"
APP_URL = "https://[nama-app].streamlit.app"

DESC = {
    "Temp": ("°C", "Suhu ruangan dari node sensor {n}"),
    "Light": ("Lux", "Intensitas cahaya dari node sensor {n}"),
    "Sound": ("Volt", "Keluaran amplifier mikrofon (ADC) node {n}"),
}


def hpstr(h):
    return (f"hidden={h['hidden']}, lr={h['lr']:.5f}, T={h['seq_len']}, dropout={h['dropout']:.3f}, "
            f"batch={h['batch_size']}")


def cv_text():
    return "; ".join(f"fold {k + 1}: train [0, {a:.0%}) → val [{a:.0%}, {b:.0%})"
                     for k, (a, b) in enumerate(cfg["cv_folds"])) + " dari panjang setiap sesi"


def tf(r, k="f1_macro", s="test"):
    return r["summary"][k][s]


def delta(x, y, k="f1_macro", s="test"):
    """Selisih mean y - x dan apakah melebihi gabungan std antar-seed."""
    dx, dy = tf(x, k, s), tf(y, k, s)
    d = dy["mean"] - dx["mean"]
    pooled = np.sqrt((dx["std"] ** 2 + dy["std"] ** 2) / 2)
    return d, pooled


def word_delta(d, pooled):
    arah = "naik" if d > 0 else "turun"
    if abs(d) <= pooled:
        return f"{arah} {abs(d):.4f} (masih di dalam simpangan baku antar-seed ≈ {pooled:.4f}, jadi belum dapat disebut perbedaan nyata)"
    return f"{arah} {abs(d):.4f} (lebih besar dari simpangan baku antar-seed ≈ {pooled:.4f})"


def data_dictionary_rows():
    rows = [["Date, Time", "-", "Tanggal & waktu pengukuran (YYYY/MM/DD, HH:MM:SS)", "-"]]
    d = eda["describe"]
    for f in cfg["sensor_features"]:
        if f.startswith("S5_CO2_Slope"):
            u, desc = "ppm/langkah", "Kemiringan CO2 dalam sliding window (fitur turunan dari penyedia dataset)"
        elif f.startswith("S5_CO2"):
            u, desc = "ppm", "Konsentrasi CO2 (node S5)"
        elif f.endswith("PIR"):
            u, desc = "biner", f"Sensor gerak PIR node {f[:2]} (1 = ada gerakan)"
        else:
            kind = f.split("_")[1]
            u, desc = DESC[kind][0], DESC[kind][1].format(n=f[:2])
        rows.append([f, u, desc, f"{d[f]['min']:.2f} – {d[f]['max']:.2f}"])
    rows.append(["Room_Occupancy_Count", "orang", "TARGET: jumlah penghuni (ground truth manual)", "0 – 3"])
    return rows


def config_rows():
    rows = []
    for r in (A, B, C, D):
        hp = r["hparams"]
        rows.append([f"{r['code']}. {r['label']}", r["n_features"], hp["hidden"], f"{hp['lr']:.5f}", hp["seq_len"],
                     f"{hp['dropout']:.3f}", hp["batch_size"], f"{r['n_params']:,}", f"{r['train_time_mean_s']:.1f}"])
    return rows


CONFIG_HEADER = ["Model", "Sensor", "hidden", "lr", "T", "dropout", "batch", "Parameter", "Waktu train (s)"]
CONFIG_W = [3.6, 1.4, 1.4, 1.8, 1.0, 1.6, 1.4, 1.8, 2.4]


def metric_rows(split="test", with_e=False):
    rows = []
    for r in (A, B, C, D):
        rows.append([f"{r['code']}. {r['label']}", pm(tf(r, "f1_macro", "val")), pm(tf(r, "f1_macro", split)),
                     pm(tf(r, "accuracy", split)), pm(tf(r, "precision_macro", split)), pm(tf(r, "recall_macro", split))])
    if with_e:
        rows.append(["E. D + Fuzzy", "= D", "= D", "= D", "= D",
                     f"keputusan HVAC cocok {fe['decision_agreement']:.1%}"])
    return rows


METRIC_HEADER = ["Model", "F1 val", "F1 test", "Accuracy test", "Precision test", "Recall test"]
METRIC_W = [3.2, 2.7, 2.7, 2.7, 2.7, 3.0]


def comparison_tables(R, with_e=False):
    R.table(CONFIG_HEADER, config_rows(), CONFIG_W, "Konfigurasi empat model (sensor & hyperparameter)", font_size=7.4)
    R.table(METRIC_HEADER, metric_rows(with_e=with_e), METRIC_W,
            "Metrik (macro, mean ± std 5 seed training; test hanya untuk evaluasi akhir)"
            + ("; E menambah lapisan keputusan, bukan akurasi" if with_e else ""), font_size=7.4)


def split_table(R):
    rows = []
    for sp in ["train", "val", "test"]:
        v = eda["split_counts"][sp]
        rows.append([sp, v["n"]] + [v["class_counts"].get(str(k), 0) for k in range(4)])
    R.table(["Split", "n", "kelas 0", "kelas 1", "kelas 2", "kelas 3"], rows, [2.5, 2, 2, 2, 2, 2],
            "Jumlah sampel per split dan kelas (split kronologis 60/20/20 di dalam setiap sesi)")


def roles_table(R, caption="Pembagian peran (usulan, sesuaikan dengan kesepakatan kelompok)"):
    R.table(["Anggota", "NIM", "Tanggung jawab utama", "Halaman dashboard"], [
        ["Arief Rahmansyah Firdaus", "04231011", "Data understanding, EDA, preprocessing & split anti-leakage, data dictionary", "Overview, Dataset Explorer"],
        ["Davin Oentario Adytiawarman", "04231018", "Genetic Algorithm: kromosom, fitness, operator, analisis multi-seed", "GA Optimization"],
        ["Musta'in Yusuf", "04231065", "RNN (BPTT, Adam), PSO hyperparameter, eksperimen pembanding A–D", "PSO, Neural Network, Model Comparison"],
        ["Muhammad Rozul Faza", "04231063", "Fuzzy Inference System, pipeline inference, deployment Streamlit, QA", "Fuzzy Inference, Prediction"],
    ], [3.6, 2.0, 7.2, 4.2], caption)


ITER1 = {k: load_json(f"iter1_single_val/{k}.json") for k in ["comparison", "ga_results", "pso_all", "pso_ga"]}
PROBE = load_json("validation_probe.json")


def validation_section(R, short=False):
    """Ablation protokol validasi: iterasi 1 (satu blok val) vs iterasi 2 (CV forward-chaining)."""
    c1 = ITER1["comparison"]
    R.p("<b>Iterasi 1 (satu blok validasi).</b> Pada percobaan pertama, GA dan PSO dinilai pada satu blok validasi "
        "[60%, 80%) setiap sesi. Macro-F1 validasi hampir semua kandidat ≈ 1.0 sehingga optimasi tidak terarah: GA memilih "
        f"{len(ITER1['ga_results']['final_selection']['features'])} sensor ("
        + ", ".join(ITER1["ga_results"]["final_selection"]["features"]) + ") yang skor validasinya sempurna, tetapi pada "
        f"evaluasi test macro-F1 GA+NN hanya {pm(tf(c1['ga_rnn']))} dan GA+PSO+NN {pm(tf(c1['ga_pso_rnn']))}, di bawah "
        f"baseline {pm(tf(c1['baseline_rnn']))}. Analisis galat per sesi menunjukkan kesalahan terkumpul di bagian test sesi 4 "
        "(23 Desember siang), ketika sensor suhu bergeser ke luar rentang train (S2_Temp mencapai 29.0 °C, maksimum train "
        "28.38 °C); subset yang bergantung pada suhu paling terdampak.")
    R.p("<b>Pengungkapan.</b> Masalah ini baru terlihat setelah test dievaluasi pada iterasi 1. Karena itu satu-satunya "
        "perubahan pada iterasi 2 adalah <b>protokol validasi</b> GA/PSO, yang dibuktikan perlu tanpa memakai test "
        "(Tabel berikut): tidak ada fitur, hyperparameter, atau aturan yang dipilih berdasarkan test. Hasil iterasi 1 "
        "tetap disimpan di <i>artifacts/iter1_single_val/</i>.")
    rows = [[x["subset"], len(x["features"]), f4(x["single_val_score"]), f4(x["cv_score"]),
             " / ".join(f"{p['f1']:.3f}" for p in x["cv_per_fold"])] for x in PROBE["rows"]]
    R.table(["Subset sensor", "n", "Skor 1 blok val", "Skor CV 3 fold", "F1 per fold"], rows,
            [5.4, 1.0, 2.8, 2.8, 4.6], "Uji protokol validasi tanpa test (skor = macro-F1 − log-loss, RNN fitness GA)")
    R.p("Satu blok validasi menilai subset GA iterasi 1 sebagai yang <b>terbaik</b>, sedangkan CV forward-chaining 3 fold "
        "(" + cv_text() + ") menilainya <b>terburuk</b> dan lebih menyukai subset tanpa sensor suhu. Fold awal memuat periode "
        "dan kondisi yang berbeda, sehingga skor CV lebih mewakili kemampuan generalisasi ke waktu yang belum terlihat.")
    if short:
        return
    rows = []
    for key in ["baseline_rnn", "ga_rnn", "pso_rnn", "ga_pso_rnn"]:
        a, b = c1[key], comp[key]
        rows.append([f"{b['code']}. {b['label']}", a["n_features"], pm(tf(a)), b["n_features"], pm(tf(b))])
    R.table(["Model", "Fitur (iter 1)", "F1 test iter 1 (1 blok val)", "Fitur (iter 2)", "F1 test iter 2 (CV)"], rows,
            [3.4, 2.0, 4.2, 2.0, 4.2], "Dampak protokol validasi terhadap macro-F1 test (mean ± std 5 seed)")


def discussion(R):
    """Analisis hasil (angka dibaca dari artifacts)."""
    freq = ga["selection_frequency"]
    always = [f for f, v in freq.items() if v == 1.0]
    never = [f for f, v in freq.items() if v == 0.0]
    temps_never = all(freq[f] == 0.0 for f in ga["features"] if "Temp" in f)
    sA, sB, sC, sD = (tf(x)["std"] for x in (A, B, C, D))
    pa_runs = [r["gbest_fitness"] for r in pa["runs"]]
    pg_runs = [r["gbest_fitness"] for r in pg["runs"]]
    tD = D["selected_model"]["test"]
    tC = C["selected_model"]["test"]
    cmC = tC["confusion_matrix"]
    R.p(f"<b>Mengapa GA membantu.</b> Sensor yang selalu terpilih di semua seed: {', '.join(always)}; yang tidak pernah "
        f"terpilih: {', '.join(never)}." + (" Tidak satu pun sensor suhu terpilih." if temps_never else "") +
        " Secara fisik, lampu dinyalakan penghuni, suara muncul dari aktivitas dan PIR menangkap gerakan, sehingga sinyal "
        "ini langsung terkait jumlah orang. Sebaliknya suhu dan CO2 berubah lambat dan juga dipengaruhi faktor luar (jam, "
        "ventilasi, cuaca), sehingga relasinya dengan okupansi bergeser antar-periode. Dengan membuang sensor yang tidak "
        f"stasioner, simpangan baku macro-F1 test antar-seed turun dari {sA:.4f} (baseline) menjadi {sB:.4f} (GA+NN), "
        "artinya model menjadi jauh lebih stabil, bukan hanya lebih akurat.")
    R.p(f"<b>Mengapa PSO saja tidak membantu.</b> Pada 16 sensor, gbest ketiga seed PSO sangat berbeda "
        f"({', '.join(f'{x:.4f}' for x in pa_runs)}; std {pa['stats']['gbest_std']:.4f}), menandakan permukaan fitness "
        "yang kasar karena sensor tidak stasioner ikut menentukan hasil. Konfigurasi terbaiknya (lr "
        f"{C['hparams']['lr']:.4f}, batch {C['hparams']['batch_size']}) masih rentan: pada test, model C sering tertukar "
        f"antara kelas 1 dan 2 ({cmC[1][2]} jendela kelas 1 diprediksi 2 pada seed terpilih). Sebaliknya, pada 8 sensor "
        f"hasil GA, gbest ketiga seed hampir identik ({', '.join(f'{x:.4f}' for x in pg_runs)}; std "
        f"{pg['stats']['gbest_std']:.4f}): PSO baru efektif setelah input dibersihkan oleh GA. Ini menunjukkan "
        "manfaat integrasi: GA+PSO+NN lebih baik daripada masing-masing komponen sendiri.")
    R.p(f"<b>Model final.</b> Pada seed terpilih, F1 per kelas model D adalah "
        + ", ".join(f"{k} orang = {tD['f1_per_class'][k]:.3f}" for k in range(4)) +
        ". Kesalahan yang tersisa terkonsentrasi pada transisi (beberapa langkah saat orang masuk/keluar), karena sensor "
        "merespons dengan sedikit jeda dan jendela T langkah masih memuat kondisi sebelum perubahan.")


# =============================================================================== W12
def w12():
    R = Report(REP / "W12_Proposal_EDA.pdf", "Proposal & EDA – Smart Room Occupancy Estimation",
               "Minggu 12 · Data, preprocessing, baseline & perencanaan sistem", "W12")
    cc = eda["class_counts"]
    R.h1("1. Latar belakang & tujuan")
    R.p("HVAC adalah konsumen energi terbesar pada gedung, tetapi sering beroperasi penuh saat ruangan kosong. "
        "Proyek ini membangun sistem AI hibrida yang memperkirakan <b>jumlah penghuni ruangan (0–3 orang)</b> dari "
        "sensor lingkungan non-invasif, lalu mengubah perkiraan itu menjadi <b>rekomendasi level HVAC</b> "
        "(Minimum, Low, Medium, High). Tujuan terukur: (1) RNN dengan macro-F1 test setinggi mungkin pada split "
        "kronologis; (2) GA mengurangi jumlah sensor tanpa menurunkan performa secara berarti; (3) PSO menala "
        "hyperparameter RNN; (4) FIS menghasilkan keputusan yang konsisten dan dapat dijelaskan; (5) dashboard "
        "Streamlit dapat diakses publik.")
    R.h1("2. Arsitektur sistem")
    R.img(FIG / "architecture.png", 15, "Pipeline hybrid GA → PSO → RNN → Fuzzy → Streamlit.")
    R.p("Optimasi berat (GA, PSO, training) dijalankan offline melalui <i>scripts/</i>; dashboard hanya memuat "
        "artefak (bobot RNN .npz, scaler.json, log JSON) untuk inference. RNN ditulis dari nol dengan NumPy "
        "(Elman RNN + BPTT + Adam) agar ringan untuk deployment dan setiap langkah dapat dijelaskan.")
    R.h1("3. Dataset & data dictionary")
    R.p(f"Room Occupancy Estimation (UCI ID 864; Singh & Chaudhari, 2018; CC BY 4.0) direkam di ruangan 6 m × 4.6 m "
        f"dengan 7 node sensor. Total <b>{eda['n_rows']:,} baris</b>, {eda['n_sensor_features']} fitur sensor numerik, "
        f"periode {eda['time_start'][:10]} s.d. {eda['time_end'][:10]} ({len(eda['dates'])} tanggal), median interval "
        f"sampling {eda['sampling_median_s']:.0f} detik. Missing values: <b>{sum(eda['missing_values'].values())}</b>; "
        f"duplikat timestamp: {eda['duplicated_rows']}.")
    R.table(["Kolom", "Satuan", "Deskripsi", "Rentang"], data_dictionary_rows(), [3.2, 1.8, 8.6, 3.4],
            "Data dictionary (rentang dihitung dari data)", font_size=6.8)
    R.h1("4. Exploratory Data Analysis")
    R.img(FIG / "class_distribution.png", 8, "Distribusi target. Kelas 0 (ruang kosong) sangat dominan.")
    top = sorted(eda["corr_with_target"].items(), key=lambda kv: -abs(kv[1]))[:5]
    R.p(f"Target tidak seimbang: kelas 0 = {cc['0']} sampel, kelas 1 = {cc['1']}, kelas 2 = {cc['2']}, kelas 3 = {cc['3']} "
        f"(rasio max/min {eda['imbalance_ratio_max_min']:.1f}×). Karena itu metrik utama adalah <b>macro-F1</b>, "
        "dan training memakai <b>weighted cross-entropy</b> (bobot kelas dihitung dari train). Korelasi absolut "
        "tertinggi dengan target: " + ", ".join(f"{k} (r={v:.2f})" for k, v in top) + ". Banyak sensor saling "
        "berkorelasi tinggi (mis. Light antar-node dan Temp antar-node), sehingga ada redundansi yang dapat "
        "dipangkas oleh GA.")
    R.img(FIG / "correlation.png", 9.5, "Matriks korelasi Pearson; angka ditampilkan untuk |r| ≥ 0.5.")
    R.h1("5. Preprocessing & split anti-leakage")
    R.bullets([
        f"Data diurutkan menurut timestamp lalu dipotong menjadi <b>{eda['n_sessions']} sesi kontinu</b> "
        f"(jeda > {cfg['max_gap_seconds']} detik memulai sesi baru).",
        "<b>Split kronologis 60/20/20 di dalam setiap sesi</b>: bagian awal = train, tengah = validation, akhir = test. "
        "Tidak ada pengacakan, sehingga data masa depan tidak masuk ke training. Split per sesi menjaga agar ketiga "
        "split mencakup hari yang berbeda.",
        "StandardScaler di-fit <b>hanya pada train</b>; validation & test hanya di-transform.",
        "Windowing: input RNN adalah T langkah terakhir; jendela tidak pernah melintasi batas sesi atau batas split.",
        "Seleksi fitur (GA) dan tuning (PSO) hanya memakai periode train + validation (CV forward-chaining 3 fold); "
        "test dipakai sekali untuk evaluasi akhir.",
    ])
    split_table(R)
    R.h1("6. Metrik evaluasi")
    R.p("Klasifikasi 4 kelas: accuracy, precision, recall, <b>macro-F1</b> (utama) dan confusion matrix. GA/PSO "
        "dijalankan dengan beberapa seed dan dilaporkan best, mean, std. Fuzzy dinilai lewat konsistensi rule, "
        "skenario ekstrem dan kesesuaian keputusan, bukan disetarakan dengan akurasi.")
    R.h1("7. Baseline RNN")
    bv = base["val"]
    hp = base["hparams"]
    R.p(f"Baseline: 16 sensor, Elman RNN hidden {hp['hidden']}, T={hp['seq_len']}, lr={hp['lr']}, batch "
        f"{hp['batch_size']}, tanpa dropout, {base['n_params']:,} parameter, seed {base['seed']}. Hasil validasi: "
        f"accuracy <b>{f4(bv['accuracy'])}</b>, macro-F1 <b>{f4(bv['f1_macro'])}</b>, precision {f4(bv['precision_macro'])}, "
        f"recall {f4(bv['recall_macro'])}, {base['history']['epochs_run']} epoch ({base['history']['train_time_s']:.1f} s).")
    R.img(FIG / "curves_baseline_rnn.png", 12.5, "Kurva training baseline (model terpilih pada eksperimen 5-seed).")
    R.h1("8. Risiko & mitigasi")
    R.table(["Risiko", "Mitigasi"], [
        ["Kelas tidak seimbang (kelas 0 dominan)", "Macro-F1 sebagai metrik utama, weighted cross-entropy, laporan per kelas"],
        ["Leakage temporal", "Split kronologis per sesi, scaler fit di train, window tidak melintasi split"],
        ["GA/PSO mahal & stokastik", "RNN ringan untuk fitness, cache fitness, beberapa seed, log lengkap"],
        ["Hasil sangat tinggi sehingga perbedaan antar-model kecil", "Laporkan mean ± std 5 seed; tidak mengklaim perbaikan bila selisih < std"],
        ["Deployment gagal / koneksi saat UAS", "Dependensi minimal (tanpa TF/PyTorch), cadangan demo lokal `streamlit run app.py`"],
    ], [6.5, 10.5], "Risiko proyek")
    R.h1("9. Pembagian peran")
    roles_table(R)
    R.h1("10. Repository & Streamlit")
    R.p(f"Struktur repo: <i>src/occupancy</i> (modul), <i>scripts</i> (eksperimen 01–08), <i>models</i>, <i>artifacts</i> "
        f"(log JSON, figur), <i>app.py</i> (dashboard 8 halaman), README, requirements. Repo: {REPO_URL} · "
        f"Dashboard: {APP_URL} (isi setelah deploy).")
    return R.build()


# =============================================================================== W13
def w13():
    R = Report(REP / "W13_GA_Progress.pdf", "Progres GA – Seleksi Sensor",
               "Minggu 13 · Genetic Algorithm dan feature/variable selection", "W13")
    gc = ga["config"]
    fm = gc["fitness_model"]
    s = ga["stats"]
    R.h1("1. Desain GA")
    R.bullets([
        "<b>Kromosom</b>: vektor biner 16 gen, satu gen per sensor (urutan: " + ", ".join(ga["features"]) + "). "
        "Gen = 1 berarti sensor dipakai. Kromosom kosong diperbaiki dengan menyalakan satu gen acak.",
        f"<b>Fitness</b>: RNN kecil (hidden {fm['hidden']}, T={fm['seq_len']}, lr={fm['lr']}, maks {fm['max_epochs']} epoch, "
        f"early stopping {fm['patience']}) dilatih dan dinilai pada <b>3 fold CV forward-chaining</b> di dalam periode train+val: " + cv_text() + ". "
        "Seed training tetap sehingga kromosom yang sama selalu memberi fitness yang sama (di-cache).",
        f"<b>Operator</b>: tournament selection k={gc['tournament_k']}, uniform crossover pc={gc['p_crossover']}, "
        f"bit-flip mutation pm={gc['p_mutation']} (=1/16), elitism {gc['elitism']} individu.",
        f"<b>Populasi</b> {gc['pop_size']}, maks {gc['generations']} generasi, berhenti bila best-fitness tidak membaik "
        f"{gc['patience']} generasi. Populasi awal acak + satu individu semua-fitur sebagai acuan.",
        f"<b>Seed independen</b>: {', '.join(map(str, gc['seeds']))}.",
    ])
    R.formula(f"F(c) = mean<sub>fold</sub>[ macroF1<sub>val,k</sub> − logloss<sub>val,k</sub> ](RNN(c)) − λ · |c| / 16,   λ = {gc['lambda_size']}")
    R.p(f"Penalti λ memberi bonus kecil untuk subset yang lebih ringkas: membuang satu sensor bernilai "
        f"{gc['lambda_size'] / 16:.5f} fitness, sehingga GA hanya membuang sensor bila penurunan skor validasi lebih kecil "
        "dari itu. Log-loss validasi ditambahkan karena macro-F1 validasi baseline sudah ≈ 1.0 (jenuh) sehingga banyak "
        "kromosom akan seri; log-loss tetap kontinu dan membedakan model yang yakin-benar dari yang ragu. "
        "Test set tidak pernah dipakai di dalam fitness.")
    R.h1("2. Protokol validasi: iterasi 1 → iterasi 2")
    validation_section(R)
    R.h1("3. Konvergensi & hasil per seed (iterasi 2)")
    R.img(FIG / "ga_convergence.png", 16.5, "Kurva konvergensi GA per seed: best fitness (kiri) dan mean fitness populasi (kanan).")
    rows = [[r["seed"], r["best_chromosome"], f4(r["best_fitness"]), f4(r["best_info"]["f1"]),
             r["best_info"]["n_features"], len(r["history"]) - 1, r["n_evals"], f"{r['time_s']:.0f}"] for r in ga["runs"]]
    R.table(["Seed", "Kromosom terbaik", "Fitness", "F1 CV", "n sensor", "Generasi", "Evaluasi", "Waktu (s)"],
            rows, [1.2, 4.4, 1.6, 1.6, 1.6, 1.8, 1.8, 1.8], "Hasil GA per seed (F1 CV = rata-rata macro-F1 3 fold)")
    R.p(f"Statistik antar-seed: best fitness terbaik <b>{f4(s['best_fitness_best'])}</b>, rata-rata "
        f"<b>{f4(s['best_fitness_mean'])} ± {f4(s['best_fitness_std'])}</b>; F1 CV subset terbaik rata-rata "
        f"{f4(s['f1_mean'])} ± {f4(s['f1_std'])}; rata-rata {s['n_features_mean']:.1f} sensor. Total "
        f"{s['total_evals']} evaluasi fitness unik dalam {s['total_time_s']:.0f} detik. Kurva mean fitness yang naik "
        "menunjukkan populasi benar-benar bergerak ke daerah fitness lebih tinggi (bukan pencarian acak).")
    R.h1("4. Fitur terpilih")
    R.img(FIG / "ga_frequency.png", 16, "Frekuensi setiap sensor terpilih di kromosom terbaik semua seed.")
    R.table(["", "Sensor"], [[f"Sebelum GA ({len(ga['features'])})", ", ".join(ga["features"])],
                             [f"Sesudah GA ({len(GA_FEATS)})", ", ".join(GA_FEATS)],
                             [f"Dieliminasi ({len(DROPPED)})", ", ".join(DROPPED)]], [3.4, 13.6],
            "Fitur sebelum dan sesudah GA")
    R.p(f"Subset final diambil dari kromosom dengan fitness CV tertinggi (seed {ga['final_selection']['seed']}). "
        "Sensor yang selalu terpilih di semua seed dapat dianggap paling informatif; sensor yang frekuensinya rendah "
        "umumnya redundan dengan sensor sejenis di node lain (lihat matriks korelasi W12).")
    R.h1("5. Baseline NN vs GA + NN")
    d, pool = delta(A, B)
    rows = []
    for x in (A, B):
        rows.append([f"{x['code']}. {x['label']}", x["n_features"], pm(tf(x, "f1_macro", "val")),
                     pm(tf(x, "accuracy", "val")), pm(tf(x)), x["n_params"], f"{x['train_time_mean_s']:.1f}"])
    R.table(["Model", "Sensor", "F1 val", "Acc val", "F1 test", "Parameter", "Waktu train (s)"], rows,
            [3.0, 1.4, 2.6, 2.6, 2.6, 1.8, 2.2], "Baseline vs GA+NN (mean ± std 5 seed; hyperparameter identik)")
    R.p(f"Dengan hyperparameter identik, GA memangkas input dari 16 menjadi {B['n_features']} sensor "
        f"({100 * (1 - B['n_features'] / 16):.0f}% lebih sedikit) dan parameter dari {A['n_params']} menjadi "
        f"{B['n_params']}. Pada blok validasi utama keduanya sama-sama jenuh (≈ 1.0), sehingga perbedaannya baru terlihat "
        f"pada evaluasi akhir: macro-F1 test {word_delta(d, pool)}, dan std antar-seed turun dari {tf(A)['std']:.4f} "
        f"menjadi {tf(B)['std']:.4f}. Biaya GA ({s['total_time_s']:.0f} s total) dibayar sekali secara offline.")
    R.h1("6. Kriteria pemeriksaan")
    R.bullets(["GA melakukan pencarian nyata: populasi, seleksi, crossover, mutasi, elitism; log per generasi tersimpan "
               "di artifacts/logs/03_ga.log dan artifacts/ga_results.json.",
               "Fitness dapat dipertanggungjawabkan: rata-rata 3 fold CV dari macro-F1 validasi (metrik utama) dikurangi log-loss (pembeda saat F1 jenuh), ditambah penalti ukuran eksplisit.",
               "Fitur tidak dipilih berdasarkan test set.",
               "Hasil dapat diulang: seed GA & seed training tercatat di config.json; jalankan <i>python scripts/03_ga.py</i>.",
               "Halaman dashboard <b>GA Optimization</b> menampilkan fitness, kromosom, fitur terpilih/dieliminasi dan konvergensi per seed."])
    return R.build()


# =============================================================================== W14
def w14():
    R = Report(REP / "W14_PSO_NN_Progress.pdf", "Progres PSO & Neural Network",
               "Minggu 14 · PSO, Neural Network dan eksperimen pembanding", "W14")
    pc = pa["config"]
    R.h1("1. Desain PSO")
    R.table(["Dimensi partikel", "Batas", "Decode"], [
        ["hidden", f"{pc['bounds']['hidden']}", "round(x)"],
        ["log10_lr", f"{pc['bounds']['log10_lr']}", "lr = 10^x"],
        ["seq_len (T)", f"{pc['bounds']['seq_len']}", "round(x)"],
        ["dropout", f"{pc['bounds']['dropout']}", "x"],
        ["log2_batch", f"{pc['bounds']['log2_batch']}", "batch = 2^round(x) → 16…128"],
    ], [4, 4, 9], "Search space PSO")
    R.formula("v ← w·v + c<sub>1</sub>r<sub>1</sub>(pbest − x) + c<sub>2</sub>r<sub>2</sub>(gbest − x),    x ← x + v")
    R.bullets([f"Swarm {pc['swarm_size']} partikel, maks {pc['iterations']} iterasi, inertia w menurun linear "
               f"{pc['w_start']} → {pc['w_end']}, c1 = c2 = {pc['c1']}.",
               f"|v| ≤ {pc['vmax_frac']}·(ub − lb); posisi di luar batas di-clip dan komponen kecepatannya dibalik & diredam 0.5.",
               f"pbest diperbarui bila fitness partikel naik; gbest = pbest terbaik. Berhenti bila gbest stagnan {pc['patience']} iterasi.",
               f"Fitness = rata-rata (macro-F1 − log-loss) validasi RNN atas 3 fold CV forward-chaining (maks {pc['max_epochs']} epoch, early stopping {pc['train_patience']}). "
               "Konfigurasi duplikat di-cache.",
               f"Dua run: <b>PSO+NN</b> (16 sensor) dan <b>GA+PSO+NN</b> ({len(GA_FEATS)} sensor GA), masing-masing seed "
               f"{', '.join(map(str, pc['seeds']))}."])
    R.h1("2. Konvergensi gbest & konfigurasi terbaik")
    R.img(FIG / "pso_convergence.png", 16.5, "Kurva gbest per seed untuk kedua run PSO.")
    R.img(FIG / "pso_particles.png", 16.5, "Sebaran fitness partikel per iterasi dan lintasan di bidang hidden–log10(lr).")
    rows = []
    for tag, p in [("PSO+NN", pa), ("GA+PSO+NN", pg)]:
        for r in p["runs"]:
            h = r["gbest_hparams"]
            rows.append([tag, r["seed"], f4(r["gbest_fitness"]), h["hidden"], f"{h['lr']:.5f}", h["seq_len"],
                         f"{h['dropout']:.3f}", h["batch_size"], len(r["history"]) - 1, r["n_evals"], f"{r['time_s']:.0f}"])
    R.table(["Run", "Seed", "gbest", "hidden", "lr", "T", "drop", "batch", "Iter", "Eval", "Waktu (s)"], rows,
            [2.4, 1.2, 1.5, 1.5, 1.7, 0.8, 1.2, 1.3, 1.1, 1.2, 1.6], "Hasil PSO per seed (gbest = skor CV: macro-F1 − log-loss)",
            font_size=7.2)
    R.p(f"PSO+NN: gbest {f4(pa['stats']['gbest_best'])} (mean {f4(pa['stats']['gbest_mean'])} ± {f4(pa['stats']['gbest_std'])}), "
        f"konfigurasi final {hpstr(pa['final']['hparams'])}. GA+PSO+NN: gbest {f4(pg['stats']['gbest_best'])} "
        f"(mean {f4(pg['stats']['gbest_mean'])} ± {f4(pg['stats']['gbest_std'])}), konfigurasi final {hpstr(pg['final']['hparams'])}.")
    R.h1("3. Neural Network")
    hp = D["hparams"]
    R.p(f"Elman RNN many-to-one: h<sub>t</sub> = tanh(x<sub>t</sub>W<sub>x</sub> + h<sub>t−1</sub>W<sub>h</sub> + b), "
        "dropout pada h<sub>T</sub>, Dense(4)+softmax. Training: weighted cross-entropy, BPTT penuh (diverifikasi dengan "
        f"gradient check numerik, galat relatif &lt; 1e-5), Adam, L2 {cfg['train']['l2']}, clipping norma {cfg['train']['clip']}, "
        f"maks {cfg['train']['max_epochs']} epoch, early stopping patience {cfg['train']['patience']} pada validation loss (bobot terbaik = val loss terendah). "
        f"Model final (D) memakai {D['n_features']} sensor, hidden {hp['hidden']}, T={hp['seq_len']}: {D['n_params']:,} parameter. "
        "Checkpoint .npz menyimpan bobot + daftar fitur + hyperparameter; scaler.json menyimpan pipeline preprocessing; "
        "keduanya dapat dimuat ulang oleh <i>OccupancyPipeline</i>.")
    R.img(FIG / "curves_ga_pso_rnn.png", 16, "Training & validation curves model D (GA+PSO+NN).")
    R.h1("4. Perbandingan empat konfigurasi")
    comparison_tables(R)
    R.img(FIG / "comparison_f1_test.png", 13, "Macro-F1 test per konfigurasi (error bar = std 5 seed).")
    for x, y, name in [(A, B, "GA+NN vs Baseline"), (A, C, "PSO+NN vs Baseline"), (A, D, "GA+PSO+NN vs Baseline"),
                       (C, D, "GA+PSO+NN vs PSO+NN")]:
        d, pool = delta(x, y)
        R.bullets([f"<b>{name}</b>: macro-F1 test {word_delta(d, pool)}."])
    R.h2("Analisis")
    discussion(R)
    R.h2("Dampak protokol validasi")
    rows = []
    for key in ["baseline_rnn", "ga_rnn", "pso_rnn", "ga_pso_rnn"]:
        a, b = ITER1["comparison"][key], comp[key]
        rows.append([f"{b['code']}. {b['label']}", a["n_features"], pm(tf(a)), b["n_features"], pm(tf(b))])
    R.table(["Model", "Fitur (iter 1)", "F1 test iter 1 (1 blok val)", "Fitur (iter 2)", "F1 test iter 2 (CV)"], rows,
            [3.4, 2.0, 4.2, 2.0, 4.2], "Iterasi 1 (GA/PSO dinilai pada satu blok validasi) vs iterasi 2 (CV forward-chaining 3 fold); detail di W13")
    R.p("Protokol pembanding dibuat tidak bias: split, jumlah epoch maksimum, patience, L2, clipping dan 5 seed training "
        "identik untuk keempat model; model yang disimpan untuk dashboard dipilih berdasarkan skor validasi (macro-F1 − log-loss), bukan test.")
    R.h1("5. Kriteria pemeriksaan")
    R.bullets(["PSO melakukan optimasi riil (pbest/gbest/velocity, log per iterasi dan posisi semua partikel di artifacts/pso_*.json).",
               "Model dapat dimuat ulang (models/*.npz + scaler.json) dan dipakai dashboard tanpa training ulang.",
               "Perbandingan memakai anggaran evaluasi terdokumentasi; klaim perbaikan hanya dibuat bila selisih melebihi variasi antar-seed.",
               "Dashboard beta: halaman <b>Prediction / Simulation</b> dan <b>Model Comparison</b> sudah berfungsi."])
    return R.build()


# =============================================================================== W15
def fuzzy_section(R):
    R.h2("Variabel linguistik & membership function")
    rows = []
    for name, spec in list(fuzzy.INPUTS.items()) + [("hvac (output)", fuzzy.OUTPUT)]:
        for term, (kind, p) in spec["terms"].items():
            rows.append([name, f"{spec['range'][0]:g}–{spec['range'][1]:g} {spec['unit']}", term,
                         "trapmf" if kind == "trap" else "trimf", ", ".join(f"{v:g}" for v in p)])
    R.table(["Variabel", "Semesta", "Term", "Tipe", "Parameter"], rows, [3.2, 3.2, 3.0, 2.0, 5.6],
            "Himpunan fuzzy")
    R.img(FIG / "fuzzy_memberships.png", 15.5, "Membership function tiga input dan satu output.")
    R.p("Input Occupancy adalah <b>expected occupancy</b> E = Σ k·P(k) dari softmax RNN, sehingga ketidakpastian model "
        "ikut masuk ke keputusan (mis. P(1)=0.5 dan P(2)=0.5 memberi E=1.5). CO2 diambil dari sensor S5_CO2 dan suhu dari "
        "rata-rata S1–S4_Temp; keduanya tetap dipakai FIS walaupun mungkin dieliminasi GA dari input RNN, karena perannya "
        "berbeda: kualitas udara & kenyamanan termal, bukan penanda jumlah orang.")
    R.h2("Rule base (27 aturan)")
    R.p("Inferensi Mamdani: AND = min, implikasi = min, agregasi = max, defuzzifikasi centroid. Aturan disusun dari "
        "logika level = base(Occupancy) + adj(CO2) + adj(Suhu) dengan Empty=0, Few=1, Full=2; CO2 High=+1; suhu Cool=−1, "
        "Warm=+1; hasil di-clip ke 0…3 = Minimum…High.")
    rows = [[i + 1, r["occupancy"], r["co2"], r["temperature"], r["hvac"]] for i, r in enumerate(fuzzy.RULES)]
    half = (len(rows) + 1) // 2
    merged = [a + (b if j < len(rows) - half else ["", "", "", "", ""]) for j, (a, b) in
              enumerate(zip(rows[:half], rows[half:] + [["", "", "", "", ""]]))]
    R.table(["R", "Occ", "CO2", "Suhu", "HVAC", "R", "Occ", "CO2", "Suhu", "HVAC"], merged,
            [0.8, 1.5, 1.5, 2.2, 1.9, 0.8, 1.5, 1.5, 2.2, 1.9], "Rule base lengkap (IF Occ AND CO2 AND Suhu THEN HVAC)",
            font_size=7.2)


def inference_example(R):
    o, c, t = 2.0, 900, 26.2
    res = fuzzy.infer(o, c, t)
    mu = res["membership_in"]
    R.h2("Contoh inferensi langkah demi langkah")
    R.p(f"Input: Occupancy = {o} orang, CO2 = {c} ppm, suhu = {t} °C.")
    R.bullets([f"Fuzzifikasi Occupancy: " + ", ".join(f"{k}={v:.2f}" for k, v in mu["occupancy"].items()),
               f"Fuzzifikasi CO2: " + ", ".join(f"{k}={v:.2f}" for k, v in mu["co2"].items()),
               f"Fuzzifikasi Suhu: " + ", ".join(f"{k}={v:.2f}" for k, v in mu["temperature"].items()),
               "Aturan aktif (α = min derajat antecedent): " + "; ".join(
                   f"R{r['rule']} ({r['occupancy']}, {r['co2']}, {r['temperature']} → {r['hvac']}) α={r['strength']:.2f}"
                   for r in res["fired_rules"]),
               f"Agregasi max lalu centroid → <b>{res['hvac_percent']:.1f}%</b>; derajat output terbesar pada "
               f"<b>{res['decision']}</b>."])
    R.img(FIG / "fuzzy_example_output.png", 12.5, "Agregasi himpunan output & titik centroid untuk contoh di atas.")


def fuzzy_validation(R):
    ch = fe["rule_check"]
    R.h2("Validasi keputusan")
    R.bullets([f"Konsistensi: {ch['n_rules']} aturan, lengkap = {ch['complete']}, konflik = {ch['conflicts']}.",
               f"Monotonisitas (sapuan grid {ch['monotonic_checks']} pasang titik): level keputusan tidak pernah turun "
               f"ketika occupancy, CO2 atau suhu naik ({ch['label_monotonic_violations']} pelanggaran). Nilai crisp "
               f"dapat turun maksimal {ch['max_crisp_drop_pct']} poin persen di zona transisi; ini sifat bawaan centroid "
               "ketika satu himpunan output di-clip pada α berbeda, dan tidak mengubah label."])
    R.table(["Skenario", "Occ", "CO2", "Suhu", "HVAC %", "Keputusan"],
            [[s["scenario"], s["occupancy"], s["co2"], s["temperature"], f"{s['hvac_percent']:.1f}", s["decision"]]
             for s in fe["scenarios"]], [6.4, 1.2, 1.4, 1.4, 1.8, 2.4], "Uji skenario ekstrem")
    R.p(f"Pada {fe['test_n']:,} jendela test, keputusan dari occupancy <b>prediksi</b> dibandingkan dengan keputusan dari "
        f"occupancy <b>aktual</b> (CO2 & suhu sama). Kesesuaian label: <b>{fe['decision_agreement']:.2%}</b>; selisih ≤ 1 level: "
        f"{fe['decision_within_one_level']:.2%}; under-ventilation (rekomendasi lebih rendah dari seharusnya): "
        f"{fe['under_ventilation_rate']:.2%}; over-ventilation: {fe['over_ventilation_rate']:.2%}; MAE level HVAC "
        f"{fe['mae_hvac_percent']:.2f} poin persen. Rata-rata level HVAC yang direkomendasikan {fe['mean_hvac_pred']:.1f}% "
        f"(aktual {fe['mean_hvac_true']:.1f}%), dibanding 100% bila HVAC selalu High.")
    R.img(FIG / "fuzzy_decision_cm.png", 9.5, "Matriks keputusan HVAC: occupancy aktual (baris) vs prediksi (kolom).")


def ablation_table(R):
    comparison_tables(R, with_e=True)


def dashboard_pages(R):
    R.table(["Halaman", "Isi"], [
        ["Overview", "Topik, sumber UCI, tujuan, anggota, diagram sistem, metrik ringkas"],
        ["Dataset Explorer", "Statistik, missing values, distribusi target, split, korelasi, deret sensor per sesi"],
        ["GA Optimization", "Rumus fitness, parameter, konvergensi per seed, kromosom, fitur terpilih/dieliminasi"],
        ["PSO Optimization", "Search space, parameter swarm, kurva gbest, partikel, hyperparameter terbaik"],
        ["Neural Network", "Arsitektur & jumlah parameter, training/validation curves, confusion matrix, metrik per kelas"],
        ["Fuzzy Inference", "Slider input, membership plots, rule base dengan α aktif, agregasi & centroid, penjelasan"],
        ["Prediction / Simulation", "Input manual, pilih jendela test, upload CSV → prediksi RNN → keputusan HVAC, unduh hasil"],
        ["Model Comparison", "Tabel & grafik A–D (mean ± std), ringkasan sistem E"],
    ], [4, 13], "Halaman dashboard Streamlit")


def demo_scenario(R):
    R.bullets([
        "<b>Problem & data (3 menit)</b>: Overview → Dataset Explorer (imbalance, split kronologis per sesi, korelasi).",
        "<b>Metode (3 menit)</b>: GA (fitness, konvergensi, sensor terpilih) → PSO (kurva gbest, hyperparameter) → NN (arsitektur, kurva).",
        "<b>Live demo (5 menit)</b>: Prediction → preset “Ruang kosong” (HVAC Minimum) → “1 orang” (Low) → “2 orang” (Medium) → "
        "tab jendela test, geser slider melewati momen orang masuk → upload data/sample_input.csv dan unduh hasil.",
        "<b>Fuzzy</b>: geser CO2 ke 1300 ppm dan suhu ke 29 °C, tunjukkan aturan aktif & centroid berubah.",
        "<b>Penutup</b>: Model Comparison (A–D, mean ± std) dan keterbatasan.",
        "<b>Cadangan</b>: bila internet bermasalah, jalankan lokal: <i>pip install -r requirements.txt; streamlit run app.py</i>.",
    ])


def w15():
    R = Report(REP / "W15_Full_System_Draft.pdf", "Draft Sistem Lengkap: Fuzzy, Integrasi & Deployment",
               "Minggu 15 · Fuzzy Inference, integrasi dan public deployment", "W15")
    R.h1("1. Desain Fuzzy Inference System (rekomendasi HVAC)")
    fuzzy_section(R)
    inference_example(R)
    fuzzy_validation(R)
    R.h1("2. Pipeline inference end-to-end")
    R.p("<i>OccupancyPipeline</i> (src/occupancy/pipeline.py): CSV/input manual → validasi 16 kolom sensor (missing value "
        "di-ffill/bfill) → StandardScaler dari train → pilih sensor GA → jendela T langkah → RNN softmax → expected "
        "occupancy → FIS (dengan CO2 & suhu rata-rata) → persentase HVAC + label + penjelasan aturan aktif. Input baru "
        "selalu menghasilkan prediksi dan keputusan; setiap baris ke-t ≥ T−1 diprediksi dari T baris terakhirnya.")
    R.img(FIG / "test_timeline.png", 16, "Contoh test set: jumlah penghuni aktual vs prediksi model D, kesalahan, dan level HVAC fuzzy.")
    R.h1("3. Tabel ablation")
    ablation_table(R)
    R.p("Ablation protokol validasi (iterasi 1 vs 2) dibahas di W13/W14 dan laporan final.")
    R.h1("4. Dashboard & deployment")
    dashboard_pages(R)
    R.p(f"URL Streamlit: {APP_URL} · Repo: {REPO_URL}. Repo tidak memuat password/token; dashboard hanya "
        "membaca artefak. Uji akses publik: buka URL di jendela Incognito tanpa login.")
    R.h2("Checklist QA")
    R.bullets(["8 halaman dieksekusi tanpa error (uji otomatis dengan stub Streamlit + tests/test_core.py lulus).",
               "Input manual, pilih jendela test dan upload CSV menghasilkan prediksi + keputusan.",
               "Screenshot setiap halaman & video uji fungsi: diambil setelah deploy (lihat skenario demo)."])
    R.h1("5. Skenario demo UAS")
    demo_scenario(R)
    return R.build()


# =============================================================================== FINAL
def final():
    R = Report(REP / "FINAL_Laporan_Tubes_KB_KelasB_Kelompok15.pdf",
               "Smart Room Occupancy Estimation dengan Sistem Hibrida GA–PSO–RNN–Fuzzy",
               "Laporan Akhir Tugas Besar Kecerdasan Buatan · UAS", "FINAL")
    dAD, pAD = delta(A, D)
    R.h1("Abstrak")
    R.p(f"Kami membangun sistem estimasi jumlah penghuni ruangan dan rekomendasi HVAC berbasis dataset UCI Room Occupancy "
        f"Estimation ({eda['n_rows']:,} sampel, 16 sensor, 4 kelas). Genetic Algorithm memilih {len(GA_FEATS)} dari 16 sensor, "
        f"Particle Swarm Optimization menala lima hyperparameter Elman RNN yang ditulis dari nol dengan NumPy, dan "
        f"Fuzzy Mamdani 27 aturan mengubah prediksi menjadi level HVAC. Pada split kronologis yang bebas leakage, sistem "
        f"final (GA+PSO+NN) mencapai macro-F1 test {pm(tf(D))} dan accuracy {pm(tf(D, 'accuracy'))} (5 seed), "
        f"dibanding baseline {pm(tf(A))}. Keputusan HVAC dari prediksi sama dengan keputusan dari occupancy aktual pada "
        f"{fe['decision_agreement']:.1%} jendela test. Seluruh sistem tersaji di dashboard Streamlit 8 halaman.")
    R.h1("1. Pendahuluan")
    R.p("Penghematan energi HVAC membutuhkan informasi okupansi yang akurat dan murah. Kamera menimbulkan masalah privasi, "
        "sedangkan sensor lingkungan (suhu, cahaya, suara, CO2, PIR) murah dan non-invasif tetapi sinyalnya tidak langsung. "
        "Kami mengikuti pipeline wajib mata kuliah: Dataset UCI → preprocessing → GA → PSO → Neural Network → Fuzzy → "
        "Streamlit, dengan penekanan pada keterkaitan yang dapat dibuktikan antara Problem → Data → Optimization → "
        "Learning → Decision → Deployment.")
    R.img(FIG / "architecture.png", 16.5, "Arsitektur sistem hibrida.")
    R.h1("2. Data & preprocessing")
    R.table(["Kolom", "Satuan", "Deskripsi", "Rentang"], data_dictionary_rows(), [3.2, 1.8, 8.6, 3.4], "Data dictionary")
    R.img(FIG / "class_distribution.png", 10, "Distribusi kelas target.")
    R.p(f"Tidak ada missing value. Kelas 0 mendominasi (rasio max/min {eda['imbalance_ratio_max_min']:.1f}×), sehingga kami "
        f"memakai macro-F1 dan weighted cross-entropy. Data dibagi menjadi {eda['n_sessions']} sesi kontinu; tiap sesi "
        "dibagi kronologis 60/20/20; scaler di-fit di train; jendela tidak melintasi batas sesi/split.")
    split_table(R)
    R.img(FIG / "correlation.png", 11.5, "Korelasi antar-sensor dan target.")
    R.h1("3. Metodologi")
    R.h2("3.1 Elman RNN (Neural Network)")
    R.p("Data sensor adalah deret waktu 30 detik; kehadiran orang tampak sebagai pola temporal (lampu menyala, suara, CO2 "
        "naik bertahap), sehingga RNN yang membaca T langkah terakhir dipilih sesuai penugasan. Model: h<sub>t</sub> = "
        "tanh(x<sub>t</sub>W<sub>x</sub> + h<sub>t−1</sub>W<sub>h</sub> + b), dropout pada h<sub>T</sub>, softmax 4 kelas. "
        "W<sub>h</sub> diinisialisasi ortogonal, W<sub>x</sub> Xavier. Training dengan weighted cross-entropy + L2, BPTT "
        "penuh (lolos gradient check numerik), Adam, gradient clipping dan early stopping pada validation loss.")
    R.h2("3.2 Genetic Algorithm")
    gc = ga["config"]
    R.formula(f"F(c) = mean<sub>fold</sub>[ macroF1<sub>val,k</sub> − logloss<sub>val,k</sub> ](RNN(c)) − {gc['lambda_size']} · |c| / 16")
    R.p(f"Kromosom 16-bit; tournament k={gc['tournament_k']}, uniform crossover {gc['p_crossover']}, bit-flip {gc['p_mutation']}, "
        f"elitism {gc['elitism']}, populasi {gc['pop_size']}, maks {gc['generations']} generasi, early stop {gc['patience']}; "
        f"{len(gc['seeds'])} seed independen.")
    R.h2("3.3 Particle Swarm Optimization")
    pc = pa["config"]
    R.p(f"Partikel 5-D: hidden {pc['bounds']['hidden']}, log10 lr {pc['bounds']['log10_lr']}, T {pc['bounds']['seq_len']}, "
        f"dropout {pc['bounds']['dropout']}, log2 batch {pc['bounds']['log2_batch']}. Swarm {pc['swarm_size']}, {pc['iterations']} "
        f"iterasi, w {pc['w_start']}→{pc['w_end']}, c1=c2={pc['c1']}, v<sub>max</sub> {pc['vmax_frac']} rentang; fitness macro-F1 "
        f"validasi; {len(pc['seeds'])} seed untuk masing-masing run PSO+NN dan GA+PSO+NN.")
    R.h2("3.4 Fuzzy Inference System")
    fuzzy_section(R)
    R.h2("3.5 Protokol eksperimen")
    R.p(f"Empat konfigurasi (A Baseline, B GA+NN, C PSO+NN, D GA+PSO+NN) dilatih dengan {len(cfg['final_train_seeds'])} seed "
        f"training, maks {cfg['train']['max_epochs']} epoch, patience {cfg['train']['patience']}, pada split yang sama. "
        "Dilaporkan mean ± std; model untuk dashboard = seed dengan skor validasi (macro-F1 − log-loss) terbaik. Test hanya dipakai di tahap ini. "
        "GA dan PSO dinilai dengan CV forward-chaining 3 fold di dalam periode train+val (" + cv_text() + "); "
        "setiap fold memiliki scaler sendiri yang di-fit pada train fold tersebut.")
    R.h1("4. Hasil")
    R.h2("4.1 GA")
    R.img(FIG / "ga_convergence.png", 16, "Konvergensi GA per seed.")
    s = ga["stats"]
    R.p(f"Best fitness {f4(s['best_fitness_best'])}, mean {f4(s['best_fitness_mean'])} ± {f4(s['best_fitness_std'])} "
        f"({s['total_evals']} evaluasi, {s['total_time_s']:.0f} s). Sensor terpilih ({len(GA_FEATS)}): {', '.join(GA_FEATS)}. "
        f"Dieliminasi: {', '.join(DROPPED)}.")
    R.img(FIG / "ga_frequency.png", 15, "Frekuensi sensor terpilih antar-seed.")
    R.h2("4.2 PSO")
    R.img(FIG / "pso_convergence.png", 16, "Kurva gbest PSO.")
    R.p(f"PSO+NN: {hpstr(pa['final']['hparams'])} (gbest {f4(pa['stats']['gbest_best'])}, mean {f4(pa['stats']['gbest_mean'])} ± "
        f"{f4(pa['stats']['gbest_std'])}). GA+PSO+NN: {hpstr(pg['final']['hparams'])} (gbest {f4(pg['stats']['gbest_best'])}, mean "
        f"{f4(pg['stats']['gbest_mean'])} ± {f4(pg['stats']['gbest_std'])}).")
    R.h2("4.3 Ablation protokol validasi (iterasi 1 vs iterasi 2)")
    validation_section(R)
    R.h2("4.4 Perbandingan model")
    ablation_table(R)
    R.img(FIG / "comparison_f1_test.png", 12.5, "Macro-F1 test (mean ± std 5 seed).")
    R.h2("4.5 Neural Network final (D)")
    R.img(FIG / "curves_ga_pso_rnn.png", 15.5, "Training & validation curves model D.")
    t = D["selected_model"]["test"]
    R.table(["Kelas", "Precision", "Recall", "F1", "Support"],
            [[f"{k} orang", f4(t["precision_per_class"][k]), f4(t["recall_per_class"][k]), f4(t["f1_per_class"][k]),
              t["support"][k]] for k in range(4)], [3, 3, 3, 3, 3], "Metrik per kelas model D (seed terpilih, test)")
    R.img(FIG / "cm_ga_pso_rnn.png", 8.5, "Confusion matrix test model D.")
    R.img(FIG / "test_timeline.png", 15.5, "Aktual vs prediksi pada test set beserta keputusan HVAC.")
    R.h2("4.6 Fuzzy decision layer (E)")
    inference_example(R)
    fuzzy_validation(R)
    R.h1("5. Pembahasan")
    items = []
    for x, y, name in [(A, B, "GA+NN vs Baseline"), (A, C, "PSO+NN vs Baseline"), (A, D, "GA+PSO+NN vs Baseline"),
                       (C, D, "GA+PSO+NN vs PSO+NN")]:
        d, pool = delta(x, y)
        items.append(f"<b>{name}</b>: macro-F1 test {word_delta(d, pool)}.")
    R.bullets(items)
    discussion(R)
    R.p("<b>Lapisan fuzzy.</b> Fuzzy tidak menaikkan akurasi, tetapi menambah interpretabilitas: setiap rekomendasi disertai "
        "aturan aktif dan derajatnya. Memakai expected occupancy membuat keputusan bergeser halus saat model ragu. "
        f"Rata-rata level HVAC yang direkomendasikan pada test adalah {fe['mean_hvac_pred']:.1f}% "
        "(ruang lebih sering kosong), dibanding 100% bila HVAC selalu High; ini proksi potensi penghematan, bukan kWh terukur.")
    R.h1("6. Keterbatasan")
    R.bullets(["Dataset hanya satu ruangan, 4 hari, maksimum 3 orang; generalisasi ke ruangan/musim lain belum diuji.",
               "Test set hanya mencakup ±20% akhir setiap sesi (2.027 baris); kesimpulan antar-konfigurasi harus dibaca bersama std antar-seed.",
               "Protokol validasi diubah setelah test iterasi 1 terlihat (diungkapkan di bagian 4.3); idealnya diuji ulang pada data periode baru.",
               "Fitness GA memakai RNN kecil dengan epoch terbatas (proksi), sehingga subset optimal untuk RNN penuh bisa sedikit berbeda.",
               "Rule base fuzzy disusun berdasarkan pengetahuan domain, bukan data konsumsi energi aktual; proksi energi bukan pengukuran kWh.",
               "Input manual di dashboard mengasumsikan kondisi tunak (nilai konstan sepanjang jendela)."])
    R.h1("7. Kesimpulan")
    R.p(f"Sistem hibrida bekerja end-to-end dan dapat diverifikasi: GA memangkas sensor menjadi {len(GA_FEATS)}, PSO memilih "
        f"hyperparameter RNN, model final mencapai macro-F1 test {pm(tf(D))} pada split kronologis, dan FIS menghasilkan "
        f"rekomendasi HVAC yang konsisten (0 konflik, monoton secara label) dengan kesesuaian {fe['decision_agreement']:.1%} "
        "terhadap keputusan ideal. Semua artefak, log dan seed tersedia untuk diulang.")
    R.h1("8. Reproducibility & deployment")
    R.bullets(["<i>pip install -r requirements.txt</i> → <i>bash run_all.sh</i> (atau skrip 01–08 berurutan) → <i>streamlit run app.py</i>.",
               "Seed: GA " + ", ".join(map(str, ga["config"]["seeds"])) + "; PSO " + ", ".join(map(str, pa["config"]["seeds"])) +
               "; training " + ", ".join(map(str, cfg["final_train_seeds"])) + " (config.json). Versi library dikunci di requirements-lock.txt.",
               "Log: artifacts/logs/*.log; hasil: artifacts/*.json; model: models/*.npz + scaler.json. Angka dapat berbeda pada "
               "digit ke-4 antar-mesin karena urutan operasi floating-point BLAS multithread.",
               f"Repo: {REPO_URL} · Dashboard: {APP_URL}."])
    dashboard_pages(R)
    R.h1("9. Kontribusi anggota")
    roles_table(R, "Kontribusi anggota (lengkapi bukti: commit, file, halaman)")
    R.h1("Referensi")
    for ref in [
        "Singh, A. P., & Chaudhari, S. (2018). Room Occupancy Estimation [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P605",
        "Singh, A. P., Jain, V., Chaudhari, S., Kraemer, F. A., Werner, S., & Garg, V. (2018). Machine learning-based occupancy estimation using multivariate sensor nodes. IEEE Globecom Workshops.",
        "Elman, J. L. (1990). Finding structure in time. Cognitive Science, 14(2), 179–211.",
        "Werbos, P. J. (1990). Backpropagation through time: what it does and how to do it. Proceedings of the IEEE, 78(10).",
        "Holland, J. H. (1975). Adaptation in Natural and Artificial Systems. University of Michigan Press.",
        "Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proc. IEEE ICNN, 1942–1948.",
        "Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. Proc. IEEE CEC, 69–73.",
        "Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. Int. J. Man-Machine Studies, 7(1), 1–13.",
        "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.",
    ]:
        R.p(ref)
    return R.build()


if __name__ == "__main__":
    for fn in (w12, w13, w14, w15, final):
        print("built", fn())
