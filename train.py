"""Melatih ulang model utama (MLP 64-32, lr 1e-2, focal loss) dari CSV — versi skrip dari notebook.
Pemakaian: python train.py   (membaca data/student-por.csv, menulis models/ dan results/train_metrics.json)
"""
import os, json, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, pandas as pd, joblib, tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

SEED, PASS_MARK = 42, 10
BINARY_COLS = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
NOMINAL_COLS = ["Mjob", "Fjob", "reason", "guardian"]
NUMERIC_COLS = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]
FEATURES = NUMERIC_COLS + BINARY_COLS + NOMINAL_COLS
CFG = dict(units=(64, 32), dropout=0.3, l2=1e-3, lr=1e-2, batch_size=32, epochs=200, gamma=2.0, patience=25)

def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s); random.seed(s); np.random.seed(s); keras.utils.set_random_seed(s)

def main(path="data/student-por.csv"):
    tf.config.experimental.enable_op_determinism()
    df = pd.read_csv(path, sep=None, engine="python"); df.columns = [c.strip().strip('"') for c in df.columns]
    for c in ["G1", "G2", "G3"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates().dropna(subset=FEATURES + ["G3"]).reset_index(drop=True)
    df["at_risk"] = (df["G3"] < PASS_MARK).astype(int)
    X, y = df[FEATURES], df["at_risk"].values
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC_COLS),
                              ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), BINARY_COLS),
                              ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL_COLS)], sparse_threshold=0)
    A, B, C = (prep.fit_transform(X_tr).astype("float32"), prep.transform(X_va).astype("float32"), prep.transform(X_te).astype("float32"))
    alpha = round(1 - float(y_tr.mean()), 4)
    set_seed(SEED)
    inp = keras.Input(shape=(A.shape[1],), name="fitur_siswa"); x = inp
    for i, u in enumerate(CFG["units"], 1):
        x = keras.layers.Dense(u, kernel_regularizer=keras.regularizers.l2(CFG["l2"]), name=f"dense_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"batchnorm_{i}")(x)
        x = keras.layers.Activation("relu", name=f"relu_{i}")(x)
        x = keras.layers.Dropout(CFG["dropout"], name=f"dropout_{i}")(x)
    model = keras.Model(inp, keras.layers.Dense(1, activation="sigmoid", name="prob_berisiko")(x), name="student_risk_mlp")
    model.compile(optimizer=keras.optimizers.Adam(CFG["lr"]),
                  loss=keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=alpha, gamma=CFG["gamma"]),
                  metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy"),
                           keras.metrics.Recall(name="recall"), keras.metrics.Precision(name="precision")])
    model.fit(A, y_tr, validation_data=(B, y_va), epochs=CFG["epochs"], batch_size=CFG["batch_size"], verbose=2, shuffle=True,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=CFG["patience"], restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5)])
    p = model.predict(C, verbose=0).ravel(); yp = (p >= 0.5).astype(int)
    res = {"accuracy": accuracy_score(y_te, yp), "precision": precision_score(y_te, yp), "recall": recall_score(y_te, yp),
           "f1": f1_score(y_te, yp), "roc_auc": roc_auc_score(y_te, p), "pr_auc": average_precision_score(y_te, p)}
    res = {k: round(float(v), 4) for k, v in res.items()}
    os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
    model.save("models/mlp_student_risk.keras"); joblib.dump(prep, "models/preprocessor.joblib")
    json.dump(res, open("results/train_metrics.json", "w"), indent=2)
    print("Test metrics:", res)

if __name__ == "__main__":
    main()
