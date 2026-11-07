# Student Academic Risk Prediction
Sistem prediksi dan analisis risiko akademik siswa menggunakan **Multilayer Perceptron (Deep Neural Network)**.

- Dataset: [Student Performance Data Set (Kaggle)](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set) — `student-por.csv`, 649 siswa, 33 atribut (asal: UCI, Cortez & Silva, 2008)
- Target: `at_risk = 1` jika nilai akhir G3 < 10 (15.41% data)
- Model: MLP 64-32 (BatchNorm, ReLU, Dropout 0.3, L2) + **Focal Loss** · Explainability: **Integrated Gradients**
- Hasil test: Accuracy 0.8615 · Precision 0.5357 · Recall 0.7500 · F1 0.6250 · ROC-AUC 0.9182

## Struktur
```
notebook/   Student_Academic_Risk_MLP.ipynb   (EDA → preprocessing → training → evaluasi)
app.py      aplikasi web Gradio (prediksi individu, batch CSV, penjelasan IG)
train.py    skrip pelatihan ulang model utama
models/     mlp_student_risk.keras, preprocessor.joblib, metadata.json
results/    metrics.json, eda_summary.json, scenario_runs.csv, history_main.csv
figures/    grafik EDA & evaluasi
data/       student-por.csv
```

## Menjalankan
```bash
pip install -r requirements.txt
python app.py          # buka http://localhost:7860
python train.py        # (opsional) latih ulang model
```
Deploy ke Hugging Face Spaces: buat Space bertipe Gradio, unggah seluruh isi repo (app.py di root).
