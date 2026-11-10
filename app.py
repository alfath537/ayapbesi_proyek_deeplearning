"""Student Academic Risk Prediction — aplikasi web Gradio (versi UTS).

Model: MLP 64-32 (dilatih dengan TensorFlow/Keras + focal loss), dijalankan lewat risk_engine.py
(NumPy) sehingga aplikasi tidak perlu memuat TensorFlow.

Alur satu prediksi:
  1) validasi aturan + deteksi input di luar domain (fitur unik #3)
  2) MLP → probabilitas mentah
  3) kalibrasi temperature scaling + confidence threshold (fitur unik #4)
  4) Integrated Gradients → faktor penyebab & rekomendasi (fitur unik #2)

Jalankan lokal : python app.py            (http://localhost:7860)
Google Colab   : import app; app.build_demo().launch(share=True)
Hugging Face   : Space bertipe Gradio, app_file = app.py (lihat README)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr

from risk_engine import OODDetector, RiskEngine

BASE = os.path.dirname(os.path.abspath(__file__))
ENGINE = RiskEngine()
OOD = OODDetector(ENGINE)
META = ENGINE.meta
METRICS = json.load(open(os.path.join(BASE, "results", "metrics.json")))
UTS = json.load(open(os.path.join(BASE, "results", "uts_metrics.json")))
CAL = json.load(open(os.path.join(BASE, "models", "calibration.json")))
TEMP, TAU = float(CAL["temperature"]), float(CAL["confidence_threshold"])
RISK_LEVELS = CAL["risk_levels_calibrated"]  # setara dengan batas 0.30 / 0.50 pada probabilitas mentah

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


# ---------------------------------------------------------------- inti prediksi
def calibrate(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / TEMP))


def risk_level(pc):
    for th, nm in RISK_LEVELS:
        if pc < th:
            return nm
    return RISK_LEVELS[-1][1]


def confidence_status(pc):
    conf = max(pc, 1 - pc)
    return conf, ("Yakin" if conf >= TAU else "Perlu verifikasi")


def clean_frame(df):
    df = df.copy()
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    for c in META["binary"] + META["nominal"]:
        if c in df:
            df[c] = df[c].astype(str).str.strip().str.strip('"')
    for c in META["numeric"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def explain_plot(contrib, pc):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([LABEL_ID.get(k, k) for k in top.index], top.values,
            color=["#e34948" if v > 0 else "#2a78d6" for v in top.values])
    ax.axvline(0, color="k", lw=.8)
    ax.set_title(f"Faktor penentu (Integrated Gradients) — P(berisiko) terkalibrasi = {pc:.2f}")
    ax.set_xlabel("merah = menaikkan risiko, biru = menurunkan risiko")
    fig.tight_layout()
    return fig


def predict_one(*values):
    row = clean_frame(pd.DataFrame([dict(zip(META["features"], values))]))
    issues = OOD.rule_check(row.iloc[0])
    if issues:  # lapis 1 fitur unik #3: input tidak sah → tidak diprediksi
        md = "### ⛔ Input ditolak\nData berikut tidak sesuai rentang/kategori yang dikenal model:\n" + \
             "\n".join(f"- {i}" for i in issues)
        return None, md, None
    p, X = ENGINE.predict_df(row[META["features"]])
    p = float(p[0]); pc = float(calibrate(p))
    score = float(OOD.score(X)[0])
    conf, status = confidence_status(pc)
    lvl = risk_level(pc)
    contrib = ENGINE.group_attributions(ENGINE.integrated_gradients(X[0]))
    up = contrib[contrib > 0].sort_values(ascending=False)
    vals = row.iloc[0]
    recs = [ADVICE[k] for k in up.index if k in ADVICE and NEEDS.get(k, lambda v: True)(vals[k])][:3] \
        or ["Pertahankan pola belajar saat ini dan lakukan pemantauan rutin."]

    parts = []
    if score > OOD.threshold:  # lapis 2 fitur unik #3
        parts.append(f"> ⚠️ **Profil di luar domain data latih** — jarak Mahalanobis {score:.1f} > ambang {OOD.threshold:.1f} "
                     f"(lebih jauh dari {OOD.percentile(score):.1f}% siswa training). Prediksi kurang dapat dipercaya; "
                     "periksa kembali data yang dimasukkan.\n")
    parts.append(f"### Tingkat risiko: **{lvl}**")
    parts.append(f"Probabilitas tidak lulus (G3 < {META['pass_mark']}), terkalibrasi: **{pc * 100:.1f}%** "
                 f"(skor mentah model {p:.3f}; ambang keputusan 0.5)")
    icon = "✅" if status == "Yakin" else "🔎"
    parts.append(f"\n**Keyakinan prediksi:** {icon} {status} — {conf * 100:.1f}% (confidence threshold τ = {TAU})" +
                 ("" if status == "Yakin" else "\n\n_Prediksi berada di zona ragu: sebaiknya diverifikasi guru/wali kelas sebelum ditindaklanjuti._"))
    parts.append(f"\n**Status input:** {'OK' if score <= OOD.threshold else 'DI LUAR DOMAIN'} (skor OOD {score:.1f})")
    parts.append("\n**Faktor yang paling menaikkan risiko:** " + (", ".join(LABEL_ID.get(k, k) for k in up.index[:3]) or "-"))
    parts.append("\n**Rekomendasi intervensi:**\n" + "\n".join(f"- {r}" for r in recs))
    return {"Berisiko": pc, "Tidak Berisiko": 1 - pc}, "\n".join(parts), explain_plot(contrib, pc)


def predict_batch(file):
    if file is None:
        return None, None, None
    df = clean_frame(pd.read_csv(file, sep=None, engine="python"))
    missing = [c for c in META["features"] if c not in df.columns]
    if missing:
        raise gr.Error(f"Kolom tidak ditemukan: {missing}")
    chk = OOD.check_frame(df[META["features"]], ENGINE.preprocess(df[META["features"]]))
    ok = chk.status_input != "DITOLAK"
    out = df.copy()
    pc = np.full(len(df), np.nan)
    if ok.any():
        p, _ = ENGINE.predict_df(df.loc[ok, META["features"]])
        pc[ok.values] = calibrate(p)
    out["prob_berisiko"] = np.round(pc, 4)
    out["tingkat_risiko"] = [risk_level(v) if ok_ else "-" for v, ok_ in zip(pc, ok)]
    out["keyakinan"] = [confidence_status(v)[1] if ok_ else "-" for v, ok_ in zip(pc, ok)]
    out["status_input"] = chk.status_input.values
    out["skor_ood"] = chk.ood_score.round(1).values
    out["masalah_input"] = chk.masalah.values
    out = out.sort_values("prob_berisiko", ascending=False, na_position="first")
    path = os.path.join(BASE, "hasil_prediksi.csv")
    out.to_csv(path, index=False)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    vc = out["tingkat_risiko"].value_counts().reindex(["Rendah", "Sedang", "Tinggi"]).fillna(0)
    ax.bar(vc.index, vc.values, color=["#2a78d6", "#eda100", "#e34948"])
    n_ver = int((out["keyakinan"] == "Perlu verifikasi").sum()); n_ood = int((out["status_input"] != "OK").sum())
    ax.set_title(f"Sebaran tingkat risiko ({len(out)} siswa) · perlu verifikasi: {n_ver} · input bermasalah: {n_ood}", fontsize=9)
    fig.tight_layout()
    show = ["prob_berisiko", "tingkat_risiko", "keyakinan", "status_input"] + \
           [c for c in ["keterangan", "school", "sex", "age", "failures", "absences", "G1", "G2", "G3"] if c in out]
    return out[show].head(100), fig, path


# ---------------------------------------------------------------- antarmuka
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


def demo_examples():
    path = os.path.join(BASE, "data", "contoh_demo.csv")
    if not os.path.exists(path):
        return [], []
    d = pd.read_csv(path)
    ok = [OOD.rule_check(r) == [] for _, r in d[META["features"]].iterrows()]
    d = d[ok]
    return d[META["features"]].values.tolist(), d["keterangan"].tolist()


def build_demo():
    m = METRICS["main_metrics"]["test"]
    c, o = UTS["calibration"], UTS["ood"]
    with gr.Blocks(title="Student Academic Risk Prediction") as demo:
        gr.Markdown("# 🎓 Student Academic Risk Prediction\nSistem prediksi & analisis risiko akademik siswa berbasis **Multilayer Perceptron** "
                    "(dataset Student Performance — mata pelajaran Bahasa Portugis). Siswa dikategorikan *berisiko* bila diprediksi G3 < 10. "
                    "Setiap prediksi dilengkapi **deteksi input di luar domain**, **probabilitas terkalibrasi + status keyakinan**, "
                    "dan **penjelasan Integrated Gradients**.")
        with gr.Tab("Prediksi Individu"):
            with gr.Row():
                with gr.Column(scale=3):
                    inputs = build_inputs()
                    btn = gr.Button("Analisis Risiko", variant="primary")
                with gr.Column(scale=2):
                    out_label = gr.Label(label="Probabilitas (terkalibrasi)", num_top_classes=2)
                    out_md = gr.Markdown()
                    out_plot = gr.Plot(label="Penjelasan model")
            ex, ex_lab = demo_examples()
            if ex:
                gr.Markdown("**Contoh siswa dari data testing** (klik untuk mengisi formulir): " +
                            " · ".join(f"{i + 1}) {t}" for i, t in enumerate(ex_lab)))
                gr.Examples(examples=ex, inputs=inputs)
            btn.click(predict_one, inputs, [out_label, out_md, out_plot])
        with gr.Tab("Prediksi Batch (CSV)"):
            gr.Markdown("Unggah CSV dengan kolom yang sama seperti dataset (G3 tidak wajib). Hasil diurutkan dari risiko tertinggi. "
                        "Baris dengan nilai di luar rentang/kategori tidak dikenal **DITOLAK** dan tidak diprediksi; baris dengan profil "
                        "tidak wajar ditandai **DI LUAR DOMAIN**. Contoh berkas: `data/kelas_uji.csv` dan `data/contoh_demo.csv`.")
            f_in = gr.File(label="File CSV", file_types=[".csv"], type="filepath")
            b_btn = gr.Button("Prediksi", variant="primary")
            b_tab = gr.Dataframe(label="Hasil (100 teratas)")
            b_plot = gr.Plot()
            b_file = gr.File(label="Unduh hasil lengkap")
            b_btn.click(predict_batch, f_in, [b_tab, b_plot, b_file])
        with gr.Tab("Tentang Model"):
            gr.Markdown(f"""**Arsitektur:** MLP {'-'.join(map(str, META['config']['units']))} + BatchNorm + ReLU + Dropout {META['config']['dropout']}, output sigmoid (5.441 parameter)
**Loss:** Binary Focal Crossentropy (alpha={META['config']['focal_alpha']}, gamma={META['config']['gamma']}) · **Optimizer:** Adam lr={META['config']['lr']}
**Hasil data testing (ambang 0.5):** Accuracy {m['accuracy']:.4f} · Precision {m['precision']:.4f} · Recall {m['recall']:.4f} · F1 {m['f1']:.4f} · ROC-AUC {m['roc_auc']:.4f}

**Fitur unik**
1. *Focal loss* untuk data tidak seimbang (15,4% siswa berisiko).
2. *Integrated Gradients* — faktor penyebab per siswa (baseline = rata-rata data training).
3. *Deteksi input di luar domain* — validasi aturan + jarak Mahalanobis (Ledoit-Wolf), ambang {o['threshold']} (persentil 99 out-of-fold training). False alarm data testing {o['scenarios']['Testing (asli)']['flag_rate'] * 100:.1f}%.
4. *Probabilitas terkalibrasi + confidence threshold* — temperature scaling T = {c['temperature']} (di-fit pada validation), ECE test {c['ece_test_before']} → {c['ece_test_after']}; prediksi dengan keyakinan < {TAU} ditandai **Perlu verifikasi**.

**Tingkat risiko** (probabilitas terkalibrasi): Rendah < {RISK_LEVELS[0][0]:.3f} ≤ Sedang < 0.5 ≤ Tinggi — setara batas 0.30/0.50 pada skor mentah.
**Inferensi:** NumPy (bobot diekspor dari Keras; paritas metrik 100% identik), tanpa TensorFlow saat deployment.""")
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
