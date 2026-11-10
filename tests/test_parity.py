"""Uji paritas engine NumPy vs model Keras (tanpa TensorFlow).

    python -m pytest tests/      atau      python tests/test_parity.py

Angka acuan berasal dari results/metrics.json yang dihasilkan notebook TensorFlow/Keras.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from risk_engine import OODDetector, RiskEngine  # noqa: E402
from data_utils import load_split  # noqa: E402


def test_metrics_identical_to_keras():
    E = RiskEngine()
    ref = json.load(open("results/metrics.json"))
    (_, _), (_, _), (te, yte) = load_split()
    p = E.predict_proba(E.preprocess(te[E.meta["features"]]))
    yp = (p >= 0.5).astype(int)
    tn, fp, fn, tp = (int(((yp == a) & (yte == b)).sum()) for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)))
    assert {"tn": tn, "fp": fp, "fn": fn, "tp": tp} == ref["confusion_matrix"]
    assert abs(float(E.predict_proba(E.baseline[None])[0]) - ref["ig"]["p_baseline"]) < 1e-4


def test_ig_completeness():
    E = RiskEngine()
    (_, _), (_, _), (te, _) = load_split()
    X = E.preprocess(te[E.meta["features"]].head(10))
    pb = E.predict_proba(E.baseline[None])[0]
    for x in X:
        assert abs(E.integrated_gradients(x).sum() - (E.predict_proba(x[None])[0] - pb)) < 5e-3


def test_ood_rules():
    E = RiskEngine(); D = OODDetector(E)
    row = pd.Series({**E.meta["defaults"], "G1": 85})
    assert any("G1" in i for i in D.rule_check(row))
    assert D.rule_check(pd.Series(E.meta["defaults"])) == []


if __name__ == "__main__":
    test_metrics_identical_to_keras(); test_ig_completeness(); test_ood_rules()
    print("semua uji paritas LULUS")
