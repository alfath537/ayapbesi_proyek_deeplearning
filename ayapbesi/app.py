"""Student Academic Risk Prediction — aplikasi web Gradio (model MLP).
Jalankan lokal: python app.py   |   Colab: import app; app.build_demo().launch(share=True)
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import gradio as gr

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = keras.models.load_model(os.path.join(BASE, "models", "mlp_student_risk.keras"), compile=False)
PREP = joblib.load(os.path.join(BASE, "models", "preprocessor.joblib"))
META = json.load(open(os.path.join(BASE, "models", "metadata.json")))
METRICS = json.load(open(os.path.join(BASE, "results", "metrics.json")))
BASELINE = np.array(META["baseline"], dtype="float32")
GROUPS = META["groups"]
GROUP_IDX = {g: [i for i, x in enumerate(GROUPS) if x == g] for g in dict.fromkeys(GROUPS)}

LABEL_ID = {
    "school": "Sekolah", "sex": "Jenis kelamin", "age": "Usia", "address": "Tempat tinggal", "famsize": "Ukuran keluarga",
    "Pstatus": "Status orang tua", "Medu": "Pendidikan ibu (0-4)", "Fedu": "Pendidikan ayah (0-4)", "Mjob": "Pekerjaan ibu",
    "Fjob": "Pekerjaan ayah", "reason": "Alasan memilih sekolah", "guardian": "Wali", "traveltime": "Waktu tempuh (1-4)",
    "studytime": "Waktu belajar/minggu (1-4)", "failures": "Jumlah gagal kelas sebelumnya", "schoolsup": "Bimbingan sekolah",
    "famsup": "Dukungan belajar keluarga", "paid": "Les berbayar", "activities": "Ekstrakurikuler", "nursery": "Pernah PAUD",
    "higher": "Ingin kuliah", "internet": "Internet di rumah", "romantic": "Berpacaran", "famrel": "Hubungan keluarga (1-5)",
    "freetime": "Waktu luang (1-5)", "goout": "Frekuensi keluar rumah (1-5)", "Dalc": "Alkohol hari kerja (1-5)",
    "Walc": "Alkohol akhir pekan (1-5)", "health": "Kesehatan (1-5)", "absences": "Jumlah absen", "G1": "Nilai periode 1 (0-20)",
    "G2": "Nilai periode 2 (0-20)",
}
ADVICE = {
    "G2": "Nilai periode 2 rendah: jadwalkan remedial dan pendampingan intensif sebelum ujian akhir.",
    "G1": "Nilai periode 1 rendah: lakukan diagnosis materi yang tertinggal sejak awal semester.",
    "failures": "Riwayat gagal kelas: tetapkan wali/mentor akademik dan pantau kemajuan tiap bulan.",
    "absences": "Absensi tinggi: koordinasi dengan orang tua dan pantau kehadiran mingguan.",
    "studytime": "Waktu belajar kurang: susun jadwal belajar terstruktur atau kelompok belajar.",
    "higher": "Motivasi studi lanjut rendah: konseling karier untuk menumbuhkan tujuan akademik.",
    "Dalc": "Konsumsi alkohol hari kerja: rujuk ke konselor sekolah.",
    "Walc": "Konsumsi alkohol akhir pekan: edukasi dan konseling gaya hidup sehat.",
    "goout": "Sering keluar rumah: bantu manajemen waktu antara sosial dan belajar.",
    "schoolsup": "Sudah/perlu bimbingan sekolah: evaluasi efektivitas program bimbingan yang diikuti.",
    "school": "Faktor sekolah asal: bandingkan dukungan akademik antar sekolah.",
}

# rekomendasi hanya diberikan bila nilai fiturnya memang tergolong kurang baik
NEEDS = {"G2": lambda v: v < 12, "G1": lambda v: v < 12, "failures": lambda v: v >= 1, "absences": lambda v: v >= 8,
         "studytime": lambda v: v <= 1, "higher": lambda v: v == "no", "Dalc": lambda v: v >= 3, "Walc": lambda v: v >= 3,
         "goout": lambda v: v >= 4}

def risk_level(p):
    for th, nm in META["risk_levels"]:
        if p < th:
            return nm
    return META["risk_levels"][-1][1]

def integrated_gradients(x, steps=64):
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps + 1), (-1, 1))
    b = tf.constant(BASELINE[None, :]); x = tf.constant(x[None, :].astype("float32"))
    path = b + alphas * (x - b)
    with tf.GradientTape() as tape:
        tape.watch(path); pr = MODEL(path, training=False)[:, 0]
    g = tape.gradient(pr, path)
    attr = ((x - b)[0] * tf.reduce_mean((g[:-1] + g[1:]) / 2.0, axis=0)).numpy()
    return pd.Series({grp: float(attr[idx].sum()) for grp, idx in GROUP_IDX.items()})

def predict_frame(df):
    Xt = PREP.transform(df[META["features"]]).astype("float32")
    return MODEL.predict(Xt, verbose=0).ravel(), Xt

def explain_plot(contrib, p):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([LABEL_ID.get(k, k) for k in top.index], top.values, color=["#E4572E" if v > 0 else "#2E86AB" for v in top.values])
    ax.axvline(0, color="k", lw=.8)
    ax.set_title(f"Faktor penentu (Integrated Gradients) — P(berisiko) = {p:.2f}")
    ax.set_xlabel("merah = menaikkan risiko, biru = menurunkan risiko")
    fig.tight_layout()
    return fig

def predict_one(*values):
    row = pd.DataFrame([dict(zip(META["features"], values))])
    for c in META["numeric"]:
        row[c] = pd.to_numeric(row[c])
    p, Xt = predict_frame(row); p = float(p[0])
    contrib = integrated_gradients(Xt[0])
    lvl = risk_level(p)
    up = contrib[contrib > 0].sort_values(ascending=False)
    vals = row.iloc[0]
    recs = [ADVICE[k] for k in up.index if k in ADVICE and NEEDS.get(k, lambda v: True)(vals[k])][:3] or ["Pertahankan pola belajar saat ini dan lakukan pemantauan rutin."]
    md = (f"### Tingkat risiko: **{lvl}**\n"
          f"Probabilitas tidak lulus (G3 < {META['pass_mark']}): **{p*100:.1f}%** — ambang keputusan {META['threshold']}\n\n"
          f"**Faktor yang paling menaikkan risiko:** " + (", ".join(LABEL_ID.get(k, k) for k in up.index[:3]) or "-") + "\n\n"
          "**Rekomendasi intervensi:**\n" + "\n".join(f"- {r}" for r in recs))
    return {"Berisiko": p, "Tidak Berisiko": 1 - p}, md, explain_plot(contrib, p)

def predict_batch(file):
    if file is None:
        return None, None, None
    df = pd.read_csv(file, sep=None, engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]
    missing = [c for c in META["features"] if c not in df.columns]
    if missing:
        raise gr.Error(f"Kolom tidak ditemukan: {missing}")
    for c in META["numeric"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    p, _ = predict_frame(df)
    out = df.copy(); out["prob_berisiko"] = p.round(4); out["tingkat_risiko"] = [risk_level(v) for v in p]
    out = out.sort_values("prob_berisiko", ascending=False)
    path = os.path.join(BASE, "hasil_prediksi.csv"); out.to_csv(path, index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    vc = out["tingkat_risiko"].value_counts().reindex(["Rendah", "Sedang", "Tinggi"]).fillna(0)
    ax.bar(vc.index, vc.values, color=["#2E86AB", "#F4A261", "#E4572E"]); ax.set_title(f"Sebaran tingkat risiko ({len(out)} siswa)")
    fig.tight_layout()
    show = ["prob_berisiko", "tingkat_risiko"] + [c for c in ["school", "sex", "age", "failures", "absences", "G1", "G2"] if c in out]
    return out[show].head(100), fig, path

def build_inputs():
    comps = {}
    rng, dflt, ch = META["valid_ranges"], META["defaults"], META["choices"]
    sections = [("Demografi", ["school", "sex", "age", "address"]),
                ("Keluarga", ["famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "guardian", "famrel", "famsup"]),
                ("Sekolah & Belajar", ["reason", "traveltime", "studytime", "failures", "schoolsup", "paid", "activities", "nursery", "higher", "internet", "absences"]),
                ("Sosial & Gaya Hidup", ["romantic", "freetime", "goout", "Dalc", "Walc", "health"])]
    with gr.Row():
        comps["G1"] = gr.Slider(0, 20, value=dflt["G1"], step=1, label=LABEL_ID["G1"])
        comps["G2"] = gr.Slider(0, 20, value=dflt["G2"], step=1, label=LABEL_ID["G2"])
    for title, cols in sections:
        with gr.Accordion(title, open=(title == "Sekolah & Belajar")):
            with gr.Row():
                for c in cols:
                    if c in META["numeric"]:
                        lo, hi = rng[c]
                        comps[c] = gr.Slider(lo, hi, value=dflt[c], step=1, label=LABEL_ID.get(c, c), min_width=180)
                    else:
                        comps[c] = gr.Dropdown(ch[c], value=dflt[c], label=LABEL_ID.get(c, c), min_width=180)
    return [comps[c] for c in META["features"]]

def build_demo():
    m = METRICS["main_metrics"]["test"]
    with gr.Blocks(title="Student Academic Risk Prediction") as demo:
        gr.Markdown("# 🎓 Student Academic Risk Prediction\nSistem prediksi & analisis risiko akademik siswa berbasis **Multilayer Perceptron** "
                    "(dataset Student Performance — mata pelajaran Bahasa Portugis). Siswa dikategorikan *berisiko* bila diprediksi G3 < 10.")
        with gr.Tab("Prediksi Individu"):
            with gr.Row():
                with gr.Column(scale=3):
                    inputs = build_inputs()
                    btn = gr.Button("Analisis Risiko", variant="primary")
                with gr.Column(scale=2):
                    out_label = gr.Label(label="Probabilitas", num_top_classes=2)
                    out_md = gr.Markdown()
                    out_plot = gr.Plot(label="Penjelasan model")
            btn.click(predict_one, inputs, [out_label, out_md, out_plot])
        with gr.Tab("Prediksi Batch (CSV)"):
            gr.Markdown("Unggah CSV dengan kolom yang sama seperti dataset (G3 tidak wajib). Hasil diurutkan dari risiko tertinggi.")
            f_in = gr.File(label="File CSV", file_types=[".csv"], type="filepath")
            b_btn = gr.Button("Prediksi", variant="primary")
            b_tab = gr.Dataframe(label="Hasil (100 teratas)")
            b_plot = gr.Plot(); b_file = gr.File(label="Unduh hasil lengkap")
            b_btn.click(predict_batch, f_in, [b_tab, b_plot, b_file])
        with gr.Tab("Tentang Model"):
            gr.Markdown(f"""**Arsitektur:** MLP {'-'.join(map(str, META['config']['units']))} + BatchNorm + ReLU + Dropout {META['config']['dropout']}, output sigmoid
**Loss:** Binary Focal Crossentropy (alpha={META['config']['focal_alpha']}, gamma={META['config']['gamma']}) · **Optimizer:** Adam lr={META['config']['lr']}
**Hasil data testing:** Accuracy {m['accuracy']:.4f} · Precision {m['precision']:.4f} · Recall {m['recall']:.4f} · F1 {m['f1']:.4f} · ROC-AUC {m['roc_auc']:.4f}
**Explainability:** Integrated Gradients (baseline = rata-rata data training)
**Tingkat risiko:** Rendah < 0.30 ≤ Sedang < 0.50 ≤ Tinggi""")
    return demo

if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
