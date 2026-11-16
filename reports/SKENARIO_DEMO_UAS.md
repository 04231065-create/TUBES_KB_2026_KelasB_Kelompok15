# Skenario Demo UAS & Storyboard Video — Kelas B Kelompok 15

Durasi slot UAS ±15 menit: **3 menit problem & data → 3 menit metode & eksperimen → 5 menit live dashboard → 4 menit tanya-jawab**.

## Persiapan (H-1)

- [ ] Buka URL Streamlit di jendela **Incognito**. Pastikan bisa diakses tanpa login dan app tidak tertidur. Kalau tertidur, klik "Wake up" 5 menit sebelum giliran.
- [ ] Siapkan cadangan lokal: `pip install -r requirements.txt` lalu `streamlit run app.py` (tes tanpa internet).
- [ ] Siapkan file `data/sample_input.csv` di desktop untuk demo upload.
- [ ] Buka slide, dashboard, dan repo GitHub di tab terpisah.
- [ ] Setiap anggota sudah mencentang tabel "Kesiapan tanya-jawab" di `FORM_Kontribusi_Anggota.md`.

## Alur presentasi (pembicara usulan)

| Menit | Bagian | Pembicara | Yang ditunjukkan |
|---|---|---|---|
| 0:00–3:00 | Problem & data | Arief | Slide masalah HVAC → dashboard **Overview** (diagram sistem) → **Dataset Explorer**: imbalance 17.9×, 8 sesi, split kronologis per sesi, korelasi Light/CO2 |
| 3:00–4:30 | GA | Davin | **GA Optimization**: rumus fitness (F1 − log-loss − penalti ukuran), kurva konvergensi 5 seed, sensor terpilih vs dieliminasi |
| 4:30–6:00 | PSO & RNN | Musta'in | **PSO Optimization**: kurva gbest, hyperparameter terbaik → **Neural Network**: arsitektur, kurva train/val, confusion matrix |
| 6:00–11:00 | Live demo | Rozul | **Prediction / Simulation** (lihat langkah di bawah) → **Fuzzy Inference** → **Model Comparison** |
| 11:00–15:00 | Tanya-jawab | Semua | Dosen dapat menunjuk anggota secara acak |

## Langkah live demo (5 menit)

1. **Prediction → Input manual**: pilih preset *Ruang kosong* → prediksi 0 orang, HVAC **Minimum** (10%). Ganti ke *1 orang* → **Low** (±45%), lalu *2 orang* → **Medium** (±68%). Jelaskan bahwa nilai sensor dianggap konstan selama T langkah. Jangan memakai preset *3 orang* sebagai contoh utama: median yang ditahan konstan menghilangkan fluktuasi suara/PIR sehingga terbaca 2 orang. Kalau ditanya, jelaskan alasan ini.
2. **Tab "Pilih jendela dari test set"**: pilih sesi dengan perpindahan orang, lalu geser slider melewati momen orang masuk. Tunjukkan prediksi vs aktual dan HVAC yang ikut berubah. Tunjukkan juga plot kesalahan: error terkumpul di transisi.
3. **Tab "Upload CSV"**: unggah `sample_input.csv` → tabel prediksi per baris → unduh hasil CSV.
4. **Fuzzy Inference**: geser CO₂ ke 1300 ppm dan suhu ke 29 °C. Tunjukkan derajat keanggotaan, aturan aktif (α), area agregasi, dan centroid.
5. **Model Comparison**: tabel A–D (mean ± std 5 seed) dan ringkasan sistem E. Akui terus terang selisih yang masih di dalam std antar-seed.

## Pertanyaan yang mungkin muncul (siapkan jawaban)

- Mengapa split kronologis per sesi, bukan acak? → mencegah kebocoran data masa depan; setiap split tetap memuat semua kelas.
- Mengapa fitness memakai log-loss? → macro-F1 validasi sudah ≈ 1.0 (jenuh) sehingga tidak bisa membedakan kandidat.
- Mengapa F1 test lebih rendah dari validasi? → bagian test sesi 4 mengalami pergeseran suhu (S2_Temp sampai 29 °C, di luar rentang train).
- Bagaimana BPTT dihitung? → tunjukkan `rnn.py::_loss_grad` dan gradient check di `tests/test_core.py`.
- Bagaimana PSO menangani batas? → posisi di-clip, komponen kecepatan dibalik & diredam 0.5, |v| ≤ 0,2·rentang.
- Mengapa fuzzy memakai *expected occupancy*? → ketidakpastian softmax ikut masuk sehingga keputusan bergeser halus.
- Apakah fuzzy menaikkan akurasi? → tidak; fuzzy adalah lapisan keputusan dan divalidasi secara terpisah.

## Storyboard video demo (3–5 menit)

| Waktu | Visual | Narasi singkat |
|---|---|---|
| 0:00–0:20 | Judul + anggota (slide 1) | Topik, dataset UCI 864, tujuan hemat energi HVAC |
| 0:20–0:50 | Overview → diagram sistem | Pipeline GA → PSO → RNN → Fuzzy → Streamlit |
| 0:50–1:20 | Dataset Explorer | Imbalance, split kronologis, sensor penting |
| 1:20–1:50 | GA page | Fitness, konvergensi, sensor terpilih |
| 1:50–2:20 | PSO + NN page | gbest, hyperparameter, kurva training, confusion matrix |
| 2:20–3:40 | Prediction (manual → jendela test → upload CSV) | Input baru → prediksi → keputusan HVAC |
| 3:40–4:20 | Fuzzy page (geser slider) | Membership, rule aktif, centroid |
| 4:20–4:50 | Model Comparison | Hasil A–D, sistem E, keterbatasan |

Rekam dengan OBS atau Xbox Game Bar (Win+Alt+R), resolusi 1080p, lalu unggah ke Drive/YouTube (unlisted, *Anyone with the link*).
