"""Memuat dataset dan membagi train/validation/test persis seperti notebook (stratified 60/20/20, seed 42)."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED, PASS_MARK = 42, 10


def load_data(path=os.path.join(ROOT, "data", "student-por.csv")):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]
    for c in ["G1", "G2", "G3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates().reset_index(drop=True)


def load_split():
    """Mengembalikan ((train, y_train), (val, y_val), (test, y_test)) berupa DataFrame lengkap + label at_risk."""
    df = load_data()
    y = (df["G3"] < PASS_MARK).astype(int).values
    a, te, ya, yte = train_test_split(df, y, test_size=0.20, stratify=y, random_state=SEED)
    tr, va, ytr, yva = train_test_split(a, ya, test_size=0.25, stratify=ya, random_state=SEED)
    return (tr, ytr), (va, yva), (te, yte)
