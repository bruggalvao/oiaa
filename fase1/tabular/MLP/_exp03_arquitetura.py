"""Etapa 2: varredura ampla de arquiteturas x funções de ativação."""
import time, itertools, csv, sys
import numpy as np
from _lab import carregar, avaliar_cv

Xdf, y = carregar()

ARQUITETURAS = [
    (16,), (32,), (64,), (128,), (256,),
    (32, 16), (64, 32), (128, 64), (256, 128), (64, 64), (128, 128), (256, 256),
    (64, 32, 16), (128, 64, 32), (256, 128, 64), (128, 128, 64), (256, 256, 128),
    (256, 128, 64, 32),
]
ATIVS = ["relu", "tanh", "elu", "selu", "gelu", "swish", "leaky_relu", "mish",
         "softplus", "sigmoid"]

saida = open("_res03_arquitetura.csv", "w", newline="", encoding="utf-8")
w = csv.writer(saida)
w.writerow(["units", "ativacao", "f1", "std", "f1_oof", "epocas", "seg"])

res = []
total = len(ARQUITETURAS) * len(ATIVS)
i = 0
t_ini = time.time()
for units, ativ in itertools.product(ARQUITETURAS, ATIVS):
    i += 1
    t = time.time()
    r = avaliar_cv(Xdf, y, {"units": units, "ativacao": ativ}, n_splits=4)
    dt = time.time() - t
    linha = ["-".join(map(str, units)), ativ, round(r["f1"], 4), round(r["std"], 4),
             round(r["f1_oof"], 4), round(r["epocas"], 1), round(dt, 1)]
    w.writerow(linha); saida.flush()
    res.append((r["f1"], r["std"], units, ativ, r["epocas"]))
    print(f"[{i:3d}/{total}] {str(units):20s} {ativ:11s} f1={r['f1']:.4f} "
          f"(+-{r['std']:.4f}) ep={r['epocas']:.0f}  {dt:.0f}s  "
          f"| decorrido {(time.time()-t_ini)/60:.1f}min", flush=True)
saida.close()

print("\n=== TOP 20 ===")
for f1, sd, u, a, ep in sorted(res, reverse=True)[:20]:
    print(f"  f1={f1:.4f} (+-{sd:.4f})  {str(u):20s} {a:11s} ep={ep:.0f}")

print("\n=== MELHOR POR ATIVACAO ===")
for a in ATIVS:
    sub = [x for x in res if x[3] == a]
    b = max(sub)
    print(f"  {a:11s} melhor f1={b[0]:.4f}  {str(b[2]):20s} | "
          f"media={np.mean([x[0] for x in sub]):.4f}")

print("\n=== MELHOR POR ARQUITETURA ===")
for u in ARQUITETURAS:
    sub = [x for x in res if x[2] == u]
    b = max(sub)
    print(f"  {str(u):20s} melhor f1={b[0]:.4f} ({b[3]:11s}) | "
          f"media={np.mean([x[0] for x in sub]):.4f}")
