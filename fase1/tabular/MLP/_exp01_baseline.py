"""Baseline: quanto dá pra tirar desses dados? (referência para o MLP)"""
import os, time
os.environ.setdefault("KERAS_BACKEND", "torch")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

RS = 42
d = pd.read_csv("treino.csv")
X = d.drop(columns=["risco_evasao"]).values.astype("float32")
y = d["risco_evasao"].values

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=RS, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)

for name, m, scaled in [
    ("LogReg bal", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RS), True),
    ("RF d4 bal", RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced",
                                        random_state=RS, n_jobs=-1), False),
    ("RF d12 bal", RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=4,
                                         class_weight="balanced", random_state=RS, n_jobs=-1), False),
    ("HGB", HistGradientBoostingClassifier(random_state=RS), False),
]:
    t = time.time()
    m.fit(Xtr_s if scaled else Xtr, ytr)
    p = m.predict(Xva_s if scaled else Xva)
    print(f"{name:12s} F1-macro={f1_score(yva, p, average='macro'):.4f}  ({time.time()-t:.1f}s)")

# --- MLP keras ---
try:
    from tensorflow import keras
except Exception:
    import keras

def build(n_in, units=(64, 32), drop=0.2, lr=1e-3, l2=0.0):
    reg = keras.regularizers.l2(l2) if l2 else None
    layers = [keras.layers.Input(shape=(n_in,))]
    for u in units:
        layers += [keras.layers.Dense(u, activation="relu",
                                      kernel_initializer="glorot_uniform",
                                      kernel_regularizer=reg),
                   keras.layers.Dropout(drop)]
    layers += [keras.layers.Dense(3, activation="softmax")]
    mdl = keras.Sequential(layers)
    mdl.compile(optimizer=keras.optimizers.Adam(lr),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return mdl

cw = {c: len(ytr) / (3 * (ytr == c).sum()) for c in range(3)}
print("class_weight:", {k: round(v, 3) for k, v in cw.items()})

t = time.time()
keras.utils.set_random_seed(RS)
mdl = build(Xtr_s.shape[1])
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
h = mdl.fit(Xtr_s, ytr, validation_data=(Xva_s, yva), epochs=400, batch_size=32,
            class_weight=cw, verbose=0, callbacks=[es])
pv = mdl.predict(Xva_s, verbose=0).argmax(1)
print(f"MLP 64-32    F1-macro={f1_score(yva, pv, average='macro'):.4f}  "
      f"epochs={len(h.history['loss'])}  ({time.time()-t:.1f}s)")
print(classification_report(yva, pv, target_names=["Baixo", "Medio", "Alto"], digits=3))
