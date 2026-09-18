"""
Etapa 5: features v3.

Avaliado com o protocolo honesto (CV 5-fold, 2 sementes de partição, ensemble
de 3 redes por fold) nas duas melhores configurações, para não confundir ganho
de feature com sorte de arquitetura.
"""
import json, time
import numpy as np
from _lab import carregar, engenharia
from _lab2 import avaliar

Xdf, y = carregar()

VARIANTES = ["v2", "v2_sem_onehot", "v3", "v3_sem_onehot"]
for v in VARIANTES:
    print(f"  {v:16s} -> {engenharia(Xdf, v).shape[1]} colunas")

CFGS = {
    "128-64-32/selu": {"units": (128, 64, 32), "ativacao": "selu", "batch_size": 16,
                       "otimizador": "adamw", "weight_decay": 1e-4},
    "64-32/elu": {"units": (64, 32), "ativacao": "elu", "otimizador": "nadam"},
}
CV_SEEDS = [42, 7]

print("\n" + "=" * 84)
print("FEATURES v3  (CV 5-fold, 2 cv-seeds, ensemble de 3 redes por fold)")
print("=" * 84)

res = []
for nome, cfg in CFGS.items():
    print(f"\n### {nome}")
    for v in VARIANTES:
        t = time.time()
        f1s = []
        for cs in CV_SEEDS:
            r, _ = avaliar(Xdf, y, {**cfg, "features": v}, n_splits=5, cv_seed=cs,
                           seeds=(0, 1, 2), ajustar=False)
            f1s.append(r["f1"])
        m = float(np.mean(f1s))
        print(f"  {v:16s} f1={m:.4f}  (por cv-seed: {[round(x,4) for x in f1s]})"
              f"  {time.time()-t:.0f}s", flush=True)
        res.append(dict(cfg=nome, features=v, f1=m))

with open("_res06_features.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

print("\n=== RANKING ===")
for r in sorted(res, key=lambda x: -x["f1"]):
    print(f"  {r['f1']:.4f}  {r['features']:16s} {r['cfg']}")
