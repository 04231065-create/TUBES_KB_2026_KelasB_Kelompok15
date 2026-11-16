# Smart Room Occupancy Estimation - Hybrid GA · PSO · RNN · Fuzzy · Streamlit

Tugas Besar **Kecerdasan Buatan (TE2514031)**, Gasal 2026/2027, **Kelas B, Kelompok 15**.

| NIM | Nama |
|---|---|
| 04231011 | Arief Rahmansyah Firdaus |
| 04231018 | Davin Oentario Adytiawarman |
| 04231065 | Musta'in Yusuf |
| 04231063 | Muhammad Rozul Faza |

- **Dataset:** [UCI Room Occupancy Estimation (ID 864)](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation). Singh & Chaudhari (2018), CC BY 4.0, DOI 10.24432/C5P605.
- **Target:** `Room_Occupancy_Count` (0–3 orang), klasifikasi 4 kelas.
- **GA:** seleksi sensor temperature/light/sound/CO2/PIR (kromosom 16-bit).
- **PSO:** hyperparameter RNN (hidden units, learning rate, sequence length, dropout, batch size).
- **NN:** Elman RNN many-to-one, ditulis dari nol dengan NumPy (BPTT + Adam).
- **Fuzzy:** Mamdani 27 aturan → rekomendasi HVAC **Minimum / Low / Medium / High**.
- **Dashboard:** Streamlit 8 halaman → `https://<nama-app>.streamlit.app` *(isi setelah deploy)*.

## Hasil utama (test set kronologis, mean ± std 5 seed training)

| Model | Sensor | Hyperparameter | Macro-F1 test | Accuracy test |
|---|---|---|---|---|
| A. Baseline NN | 16 | hidden 32, lr 0.0050, T 10, dropout 0.00, batch 64 | 0.9242 ± 0.0568 | 0.9714 ± 0.0203 |
| B. GA + NN | 8 | hidden 32, lr 0.0050, T 10, dropout 0.00, batch 64 | 0.9733 ± 0.0060 | 0.9885 ± 0.0027 |
| C. PSO + NN | 16 | hidden 87, lr 0.0211, T 10, dropout 0.10, batch 128 | 0.8945 ± 0.0621 | 0.9609 ± 0.0217 |
| D. GA + PSO + NN | 8 | hidden 36, lr 0.0046, T 16, dropout 0.24, batch 32 | 0.9851 ± 0.0087 | 0.9924 ± 0.0040 |
| E. D + Fuzzy | 8 | FIS Mamdani 27 aturan | keputusan HVAC sama dengan keputusan dari occupancy aktual pada 99.5% jendela test | |

- **Sensor terpilih GA:** S1_Light, S2_Light, S3_Light, S4_Light, S1_Sound, S4_Sound, S6_PIR, S7_PIR. Keempat sensor suhu tidak pernah terpilih di 5 seed.
- **Protokol validasi:** GA dan PSO dinilai dengan CV *forward-chaining* 3 fold di dalam periode train+val. Iterasi pertama memakai satu blok validasi; hasilnya disimpan di `artifacts/iter1_single_val/` dan dibahas di laporan final (bagian 4.3).


## Struktur repository

```
├── app.py                    # dashboard Streamlit (8 halaman, inference saja)
├── config.json               # semua seed, split, anggaran GA/PSO, hyperparameter baseline
├── requirements.txt          # dependensi deploy (streamlit, numpy, pandas, matplotlib)
├── requirements-lock.txt     # versi persis saat eksperimen offline
├── run_all.sh                # menjalankan seluruh eksperimen 01 → 07 + tes
├── data/
│   ├── Occupancy_Estimation.csv   # dataset UCI (≈0,9 MB, CC BY 4.0)
│   └── sample_input.csv           # contoh CSV untuk demo upload
├── src/occupancy/
│   ├── data.py       # load, sesi, split kronologis, scaler (fit di train), windowing
│   ├── rnn.py        # Elman RNN NumPy: forward, BPTT, Adam, early stopping, save/load, gradient check
│   ├── ga.py         # Genetic Algorithm (tournament, uniform crossover, bit-flip, elitism)
│   ├── pso.py        # Particle Swarm Optimization (inertia linear, vmax, pbest/gbest)
│   ├── fuzzy.py      # Mamdani FIS: membership, 27 rule, centroid, uji konsistensi
│   ├── trainer.py    # train/eval satu konfigurasi (test hanya bila diminta)
│   ├── pipeline.py   # inference end-to-end: CSV/manual → scaler → RNN → fuzzy
│   ├── metrics.py    # accuracy, precision, recall, F1, confusion matrix
│   └── viz.py        # plot bersama untuk dashboard & laporan
├── scripts/
│   ├── 01_eda.py  02_baseline.py  03_ga.py  04_pso.py  05_compare.py
│   ├── 06_fuzzy_eval.py  07_figures.py  08_reports.py  09_validation_probe.py
├── models/           # *.npz (bobot + fitur + hyperparameter) dan scaler.json
├── artifacts/        # *.json hasil, logs/*.log, figures/*.png, iter1_single_val/ (ablation)
├── reports/          # W12–W15, FINAL (PDF), slide, skenario demo, form kontribusi
└── tests/test_core.py
```

## Cara menjalankan

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Dashboard (memakai model & log yang sudah ada di repo)
streamlit run app.py

# 2) Mengulang seluruh eksperimen (±1 jam, 2 core)
bash run_all.sh            # atau jalankan scripts/01_eda.py … 07_figures.py satu per satu
python tests/test_core.py  # gradient check BPTT, windowing, GA, PSO, fuzzy
```

Bila `data/Occupancy_Estimation.csv` tidak ada, unduh dari halaman UCI di atas (file zip berisi
`Occupancy_Estimation.csv`) lalu letakkan di folder `data/`.

## Desain eksperimen (anti-leakage)

1. Data diurutkan menurut waktu dan dipotong menjadi **sesi kontinu** (jeda > 120 detik memulai sesi baru).
2. Setiap sesi dibagi **kronologis 60/20/20** (train/val/test). Tidak ada random split.
3. `StandardScaler` di-fit **hanya pada train**. Jendela RNN tidak melintasi batas sesi atau split.
4. GA dan PSO hanya memakai **train + validation**. Test dipakai sekali di `05_compare.py`.
5. Empat konfigurasi (A Baseline, B GA+NN, C PSO+NN, D GA+PSO+NN) dilatih dengan **5 seed** dengan protokol identik, lalu dilaporkan sebagai mean ± std.

## Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub (publik). Pastikan tidak ada token/password di repo.
2. Buka https://share.streamlit.io → **Create app** → pilih repo, branch `main`, file `app.py`.
3. Di *Advanced settings* pilih Python 3.11 atau 3.12 → **Deploy**.
4. Uji URL di jendela Incognito, lalu isi URL di README dan laporan.

Cadangan demo lokal: `streamlit run app.py`, lalu buka `http://localhost:8501`.

## Catatan

- RNN ditulis dengan NumPy agar ringan di Streamlit Cloud dan agar setiap langkah (forward, BPTT, Adam) bisa dijelaskan saat UAS. Kebenaran gradien diuji dengan *gradient check* numerik (`tests/test_core.py`).
- Keputusan fuzzy adalah lapisan pendukung keputusan, bukan pengganti akurasi NN. Fuzzy divalidasi lewat konsistensi rule, skenario ekstrem, dan kesesuaian keputusan pada test set.
