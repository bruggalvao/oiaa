"""Etapa 1: features + escalonamento (arquitetura fixa 64-32 relu)."""
import time
import numpy as np
from _lab import carregar, avaliar_cv

Xdf, y = carregar()

t0 = time.time()
r = avaliar_cv(Xdf, y, {}, n_splits=4)
print(f"[cronometro] 1 config x 4 folds = {time.time()-t0:.1f}s -> {r}")

print("\n=== FEATURES ===")
for v in ["v0", "v1", "v2"]:
    t = time.time()
    r = avaliar_cv(Xdf, y, {"features": v}, n_splits=4)
    print(f"  {v:4s} f1={r['f1']:.4f} (+-{r['std']:.4f}) oof={r['f1_oof']:.4f} "
          f"ep={r['epocas']:.0f} {r['folds']}  {time.time()-t:.0f}s")

print("\n=== SCALER (features=v2) ===")
for s in ["standard", "minmax", "robust", "quantile", "power"]:
    t = time.time()
    r = avaliar_cv(Xdf, y, {"scaler": s}, n_splits=4)
    print(f"  {s:9s} f1={r['f1']:.4f} (+-{r['std']:.4f}) oof={r['f1_oof']:.4f} "
          f"ep={r['epocas']:.0f}  {time.time()-t:.0f}s")
