"""risk_engine.py — mesin inferensi ringan (NumPy) untuk model MLP Student Academic Risk.

Model tetap MLP 64-32 yang dilatih dengan TensorFlow/Keras (models/mlp_student_risk.keras).
Bobotnya diekspor ke models/mlp_weights.npz dan parameter preprocessing ke
models/preprocess_params.json (lihat tools/export_light_model.py), sehingga aplikasi web
tidak perlu memuat TensorFlow (~600 MB). Paritas dengan Keras diuji di tests/test_parity.py.

Isi modul:
  * preprocess()            — replikasi ColumnTransformer (StandardScaler + OneHotEncoder)
  * predict_proba()         — forward pass MLP (Dense → BatchNorm → ReLU → Dropout) × 2 → sigmoid
  * integrated_gradients()  — fitur unik #2 (gradien dihitung analitik, tanpa autodiff)
  * OODDetector             — fitur unik #3: deteksi input di luar domain (validasi aturan + Mahalanobis)
  * mc_dropout()            — fitur unik #4: estimasi ketidakpastian dengan Monte Carlo Dropout
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")


class RiskEngine:
    def __init__(self, models_dir: str = MODELS):
        w = np.load(os.path.join(models_dir, "mlp_weights.npz"))
        self.W = {k: w[k].astype("float32") for k in w.files}
        self.P = json.load(open(os.path.join(models_dir, "preprocess_params.json")))
        self.meta = json.load(open(os.path.join(models_dir, "metadata.json")))
        self.eps = float(self.P.get("bn_epsilon", 1e-3))
        self.dropout = float(self.meta["config"]["dropout"])
        self.baseline = np.array(self.meta["baseline"], dtype="float32")
        groups = self.meta["groups"]
        self.group_idx = {g: [i for i, x in enumerate(groups) if x == g] for g in dict.fromkeys(groups)}
        # BatchNorm (mode inferensi) dilipat menjadi skala & geser: n = a * s + t
        self._bn = []
        for i in (1, 2):
            g, b, m, v = (self.W[f"bn{i}_{k}"] for k in ("gamma", "beta", "mean", "var"))
            s = g / np.sqrt(v + self.eps)
            self._bn.append((s.astype("float32"), (b - m * s).astype("float32")))

    # ------------------------------------------------------------------ preprocessing
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        P = self.P
        cols = []
        num = df[P["numeric"]].apply(pd.to_numeric, errors="coerce").to_numpy("float64")
        cols.append((num - np.array(P["scaler_mean"])) / np.array(P["scaler_scale"]))
        for c in P["binary"]:  # OneHotEncoder(drop="if_binary"): 1 jika kategori kedua
            pos = P["binary_categories"][c][1]
            cols.append((df[c].astype(str).str.strip().to_numpy() == pos).astype("float64")[:, None])
        for c in P["nominal"]:  # one-hot penuh; kategori tak dikenal → vektor nol
            v = df[c].astype(str).str.strip().to_numpy()
            cols.append(np.stack([(v == k) for k in P["nominal_categories"][c]], axis=1).astype("float64"))
        return np.concatenate(cols, axis=1).astype("float32")

    # ------------------------------------------------------------------ forward / backward
    def _forward(self, X, masks=None):
        W = self.W
        h = X
        cache = []
        for i, (s, t) in enumerate(self._bn, 1):
            n = (h @ W[f"dense{i}_kernel"] + W[f"dense{i}_bias"]) * s + t
            r = np.maximum(n, 0.0)
            if masks is not None:
                r = r * masks[i - 1]
            cache.append(n)
            h = r
        z = h @ W["out_kernel"] + W["out_bias"]
        return 1.0 / (1.0 + np.exp(-z[:, 0])), cache

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.atleast_2d(X).astype("float32"))[0]

    def predict_df(self, df: pd.DataFrame):
        X = self.preprocess(df)
        return self.predict_proba(X), X

    def input_gradient(self, X: np.ndarray) -> np.ndarray:
        """dP/dx untuk setiap baris X (backprop analitik melalui MLP mode inferensi)."""
        W = self.W
        p, (n1, n2) = self._forward(X)
        d = (p * (1 - p))[:, None] * W["out_kernel"][:, 0][None, :]      # dP/dr2
        d = d * (n2 > 0) * self._bn[1][0]                                  # → dP/da2
        d = (d @ W["dense2_kernel"].T) * (n1 > 0) * self._bn[0][0]         # → dP/da1
        return d @ W["dense1_kernel"].T                                    # → dP/dx

    # ------------------------------------------------------------------ fitur unik #2
    def integrated_gradients(self, x: np.ndarray, steps: int = 64) -> np.ndarray:
        """Integrated Gradients (Sundararajan dkk., 2017), aturan trapesium, baseline = rata-rata training."""
        b = self.baseline
        alphas = np.linspace(0.0, 1.0, steps + 1, dtype="float32")[:, None]
        g = self.input_gradient(b + alphas * (x - b))
        return (x - b) * ((g[:-1] + g[1:]) / 2.0).mean(axis=0)

    def group_attributions(self, attr: np.ndarray) -> pd.Series:
        """Menjumlahkan atribusi kolom one-hot kembali ke 32 fitur asal."""
        return pd.Series({g: float(attr[idx].sum()) for g, idx in self.group_idx.items()})

    # ------------------------------------------------------------------ fitur unik #4
    def mc_dropout(self, X: np.ndarray, T: int = 100, seed: int | None = 0):
        """Monte Carlo Dropout (Gal & Ghahramani, 2016): T forward pass dengan dropout aktif
        (inverted dropout seperti Keras, BatchNorm tetap mode inferensi).
        Mengembalikan (rata-rata, simpangan baku, semua sampel [T, n])."""
        X = np.atleast_2d(X).astype("float32")
        rng = np.random.default_rng(seed)
        keep = 1.0 - self.dropout
        n = X.shape[0]
        samples = np.empty((T, n), dtype="float32")
        for t in range(T):
            masks = [(rng.random((n, k)) < keep).astype("float32") / keep for k in (64, 32)]
            samples[t] = self._forward(X, masks)[0]
        return samples.mean(0), samples.std(0), samples


class OODDetector:
    """Fitur unik #3 — deteksi input di luar domain (out-of-distribution).

    Lapis 1 (aturan): nilai di luar rentang dokumentasi, bukan bilangan bulat, kosong,
                      atau kategori yang tidak dikenal → input DITOLAK.
    Lapis 2 (statistik): jarak Mahalanobis kuadrat pada ruang fitur 45 dimensi hasil
                      preprocessing, dengan kovarians Ledoit-Wolf yang di-fit pada data training.
                      Jarak > ambang (persentil ke-99 data training) → PERINGATAN.
    """

    def __init__(self, engine: RiskEngine, models_dir: str = MODELS):
        d = np.load(os.path.join(models_dir, "ood_detector.npz"))
        self.mu, self.prec = d["mean"], d["precision"]
        self.threshold = float(d["threshold"])
        self.train_scores = d["train_scores"]
        self.engine = engine
        self.meta = engine.meta

    def score(self, X: np.ndarray) -> np.ndarray:
        D = np.atleast_2d(X).astype("float64") - self.mu
        return np.einsum("ij,jk,ik->i", D, self.prec, D)

    def percentile(self, s: float) -> float:
        """Persentase data training yang jaraknya lebih kecil dari s."""
        return float((self.train_scores < s).mean() * 100)

    def rule_check(self, row: pd.Series) -> list[str]:
        issues = []
        rng, ch = self.meta["valid_ranges"], self.meta["choices"]
        for c in self.meta["numeric"]:
            v = pd.to_numeric(row.get(c), errors="coerce")
            lo, hi = rng[c]
            if pd.isna(v):
                issues.append(f"{c} kosong/bukan angka")
            elif v < lo or v > hi:
                issues.append(f"{c}={v:g} di luar rentang {lo}-{hi}")
            elif float(v) != int(v):
                issues.append(f"{c}={v:g} harus bilangan bulat")
        for c in self.meta["binary"] + self.meta["nominal"]:
            v = str(row.get(c)).strip()
            if v not in ch[c]:
                issues.append(f"{c}='{v}' bukan kategori yang dikenal {ch[c]}")
        return issues

    def check_frame(self, df: pd.DataFrame, X: np.ndarray | None = None) -> pd.DataFrame:
        rules = [self.rule_check(r) for _, r in df.iterrows()]
        if X is None:
            X = self.engine.preprocess(df)
        s = self.score(np.nan_to_num(X, nan=0.0))
        status = ["DITOLAK" if r else ("DI LUAR DOMAIN" if v > self.threshold else "OK") for r, v in zip(rules, s)]
        return pd.DataFrame({"ood_score": s, "status_input": status, "masalah": ["; ".join(r) for r in rules]}, index=df.index)
