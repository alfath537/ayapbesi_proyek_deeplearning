"""Ekspor model Keras + preprocessor ke format ringan untuk deployment (tanpa TensorFlow).

    python tools/export_light_model.py

Menghasilkan:
  models/mlp_weights.npz          — bobot Dense & BatchNorm (float32)
  models/preprocess_params.json   — mean/scale StandardScaler + kategori OneHotEncoder

Bobot dibaca langsung dari arsip .keras (berkas model.weights.h5) dengan h5py, sehingga
skrip ini tidak membutuhkan TensorFlow. Urutan variabel mengikuti Keras 3:
Dense = [kernel, bias], BatchNormalization = [gamma, beta, moving_mean, moving_variance].
"""
import io
import json
import os
import zipfile

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "models")


def read_keras_weights(path):
    import h5py  # pip install h5py

    with zipfile.ZipFile(path) as z:
        cfg = json.loads(z.read("config.json"))
        h5 = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")
    eps = [l["config"]["epsilon"] for l in cfg["config"]["layers"] if l["class_name"] == "BatchNormalization"][0]
    v = lambda layer, i: np.array(h5[f"layers/{layer}/vars/{i}"], dtype="float32")
    W = {
        "dense1_kernel": v("dense", 0), "dense1_bias": v("dense", 1),
        "dense2_kernel": v("dense_1", 0), "dense2_bias": v("dense_1", 1),
        "out_kernel": v("dense_2", 0), "out_bias": v("dense_2", 1),
    }
    for i, layer in ((1, "batch_normalization"), (2, "batch_normalization_1")):
        for j, k in enumerate(("gamma", "beta", "mean", "var")):
            W[f"bn{i}_{k}"] = v(layer, j)
    return W, eps


def preprocess_params(prep, eps):
    num, b, n = (prep.named_transformers_[k] for k in ("num", "bin", "nom"))
    cols = {name: list(c) for name, _, c in prep.transformers_ if name in ("num", "bin", "nom")}
    return {
        "numeric": cols["num"], "binary": cols["bin"], "nominal": cols["nom"],
        "scaler_mean": num.mean_.tolist(), "scaler_scale": num.scale_.tolist(),
        "binary_categories": {c: [str(x) for x in cats] for c, cats in zip(cols["bin"], b.categories_)},
        "nominal_categories": {c: [str(x) for x in cats] for c, cats in zip(cols["nom"], n.categories_)},
        "feature_out": list(prep.get_feature_names_out()),
        "bn_epsilon": eps,
    }


if __name__ == "__main__":
    W, eps = read_keras_weights(os.path.join(M, "mlp_student_risk.keras"))
    np.savez(os.path.join(M, "mlp_weights.npz"), **W)
    params = preprocess_params(joblib.load(os.path.join(M, "preprocessor.joblib")), eps)
    json.dump(params, open(os.path.join(M, "preprocess_params.json"), "w"), indent=1)
    print("OK:", {k: v.shape for k, v in W.items()})
