---
title: Student Academic Risk Prediction
emoji: 🎓
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.14.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Prediksi risiko akademik siswa dengan MLP (Deep Learning)
---

# 🎓 Student Academic Risk Prediction — Proyek UTS Pembelajaran Mendalam (Kelompok AyapBesi)

Sistem peringatan dini yang memprediksi apakah seorang siswa berisiko **tidak lulus** (nilai akhir G3 < 10)
menggunakan **Deep Neural Network jenis Multilayer Perceptron (MLP)**, lengkap dengan penjelasan faktor penyebab,
deteksi input di luar domain, probabilitas terkalibrasi, dan rekomendasi intervensi.

| | |
|---|---|
| **Dataset** | [Student Performance Data Set (Kaggle)](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set) — `student-por.csv`, 649 siswa, 33 kolom (UCI, Cortez & Silva, 2008) |
| **Target** | `at_risk = 1` jika G3 < 10 (100 siswa = 15,41%) |
| **Model** | MLP 45 → 64 → 32 → 1 (BatchNorm, ReLU, Dropout 0,3, L2 0,001), 5.441 parameter, TensorFlow/Keras |
| **Hasil test** | Accuracy 0,8615 · Precision 0,5357 · Recall 0,7500 · F1 0,6250 · ROC-AUC 0,9182 · PR-AUC 0,7026 |
| **Aplikasi** | Gradio — di-deploy ke Hugging Face Spaces |

## Fitur unik

| # | Fitur | Bukti dampak (data testing kecuali disebut lain) |
|---|---|---|
| 1 | **Focal loss** untuk data tidak seimbang | F1 validation 0,654 ± 0,040 vs BCE 0,633 dan weighted BCE 0,555 (5 seed) |
| 2 | **Integrated Gradients** (explainable AI) | completeness error 0,0006; hapus 3 fitur teratas mengubah P 5,3× lebih besar dari 3 fitur acak |
| 3 | **Deteksi input di luar domain** (aturan + Mahalanobis Ledoit-Wolf) — *baru di UTS* | false alarm data asli 0,8%; 100% input janggal terdeteksi pada 4 dari 5 skenario uji |
| 4 | **Probabilitas terkalibrasi + confidence threshold** (temperature scaling, τ = 0,75) — *baru di UTS* | ECE 0,195 → 0,088; Brier 0,118 → 0,095; prediksi "yakin" (80%) akurasinya 93,3% vs 86,2% keseluruhan |

Pengembangan lain di UTS: **engine inferensi NumPy** (`risk_engine.py`) — bobot MLP diekspor dari Keras sehingga
aplikasi tidak perlu TensorFlow. Paritas diuji: metrik test, confusion matrix, dan atribusi IG identik dengan Keras.

## Struktur repositori

```
app.py                     aplikasi web Gradio (prediksi individu, batch CSV, tentang model)
risk_engine.py             inferensi MLP (NumPy), Integrated Gradients, MC Dropout, detektor OOD
data_utils.py              memuat data & split stratified 60/20/20 (seed 42) — identik dengan notebook
uts_analysis.py            membangun & mengevaluasi fitur unik #3 dan #4 → results/uts_metrics.json, figures/16-19
train.py                   melatih ulang model utama (TensorFlow) — versi skrip dari notebook
tools/export_light_model.py  ekspor .keras + preprocessor → mlp_weights.npz + preprocess_params.json
tests/test_parity.py       uji paritas engine NumPy vs hasil Keras
notebook/
  Student_Academic_Risk_MLP.ipynb   EDA → preprocessing → training → evaluasi → IG (TensorFlow, Colab)
  UTS_Pengembangan_Fitur.ipynb      fitur unik #3 & #4 + paritas engine (dengan output)
models/   mlp_student_risk.keras, preprocessor.joblib, metadata.json,
          mlp_weights.npz, preprocess_params.json, ood_detector.npz, calibration.json
results/  metrics.json, uts_metrics.json, preprocessing.json, eda_summary.json, scenario_runs.csv, history_main.csv
figures/  01-15 grafik EDA & evaluasi, 16-19 grafik fitur UTS & arsitektur deployment
data/     student-por.csv, kelas_uji.csv (130 siswa test), contoh_demo.csv (contoh untuk video demo)
```

## Menjalankan secara lokal

```bash
git clone https://github.com/alfath537/ayapbesi_proyek_deeplearning.git
cd ayapbesi_proyek_deeplearning
pip install -r requirements.txt     # numpy, pandas, matplotlib, gradio (tanpa TensorFlow)
python app.py                       # buka http://localhost:7860
python tests/test_parity.py         # uji paritas → "semua uji paritas LULUS"
```

Reproduksi penuh (melatih ulang dan membangun artefak):

```bash
pip install -r requirements-train.txt
jupyter notebook notebook/Student_Academic_Risk_MLP.ipynb    # atau buka di Google Colab
python tools/export_light_model.py      # .keras → mlp_weights.npz + preprocess_params.json
python uts_analysis.py                  # detektor OOD, kalibrasi, figures/16-19, results/uts_metrics.json
```

## Deployment ke Hugging Face Spaces

1. **Simpan model** — notebook menyimpan `models/mlp_student_risk.keras` dan `models/preprocessor.joblib`;
   `tools/export_light_model.py` mengekspornya ke `mlp_weights.npz` + `preprocess_params.json`.
2. **Skrip inferensi** — `risk_engine.py` (forward pass, IG, OOD) dipanggil oleh `app.py`.
3. **Antarmuka** — `app.py` membangun UI Gradio (3 tab). Gradio otomatis menyediakan API (`/predict_one`, `/predict_batch`).
4. **Dependensi** — `requirements.txt` (ringan, untuk aplikasi); `requirements-train.txt` (pelatihan).
5. **Unggah** — di <https://huggingface.co/new-space>: *Space name* `student-academic-risk`, *SDK* **Gradio**,
   *Hardware* **CPU basic (free)**, *Public*. Lalu unggah isi repo lewat salah satu cara:
   - **Web**: tab *Files* → *Add file* → *Upload files* → seret semua berkas/folder → *Commit*.
   - **CLI**: `pip install huggingface_hub` → `hf auth login` → `hf upload <username>/student-academic-risk . . --repo-type space`
   Cara web/CLI dianjurkan karena berkas biner (`.npz`, `.keras`, `.png`) otomatis ditangani Hugging Face (Xet/LFS).
6. **Uji** — tunggu status *Running*, lalu coba contoh di tab *Prediksi Individu* dan unggah `data/kelas_uji.csv`
   serta `data/contoh_demo.csv` di tab *Prediksi Batch*.

Header YAML di bagian atas README ini adalah konfigurasi Space (SDK, versi Gradio, versi Python, berkas aplikasi).
Jika versi Gradio tidak tersedia di Spaces, ubah `sdk_version` ke versi 6.x yang ditawarkan saat membuat Space.

## Tim AyapBesi — Universitas Mikroskil

| NIM | Nama | Peran |
|---|---|---|
| 231110321 | Richi Putra Lin | Ketua, data engineer, deployment |
| 231112458 | Ahmad Danu | Model engineer, app developer |
| 231112897 | Asyifa Alfath | Evaluasi, XAI, dokumentasi |
