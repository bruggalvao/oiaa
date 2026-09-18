"""
Etapa 4: ensemble de seeds + regra de decisão para F1-macro.

Os multiplicadores de classe são ajustados na fatia interna de early stopping
de cada fold e aplicados no fold de avaliação, que o modelo nunca viu. Duas
sementes de CV para não confundir ganho real com sorte de partição.
"""
import json, time
import numpy as np
from _lab import carregar
from _lab2 import avaliar, avaliar_holdout

Xdf, y = carregar()
cfgs = json.load(open("_res04_melhores.json", encoding="utf-8"))
for c in cfgs:
    c["cfg"]["units"] = tuple(c["cfg"]["units"])

CV_SEEDS = [42, 7]

print("=" * 84)
print("EFEITO DO ENSEMBLE DE SEMENTES E DO AJUSTE DA REGRA DE DECISAO")
print("=" * 84)

resumo = []
for item in cfgs:
    cfg = item["cfg"]
    nome = f"{'-'.join(map(str,cfg['units']))}/{cfg['ativacao']}"
    print(f"\n### {nome}  cfg={cfg}")
    for n_seeds in (1, 3, 5):
        seeds = tuple(range(n_seeds))
        f1s, f1s_aj, pesos = [], [], []
        t = time.time()
        for cs in CV_SEEDS:
            r, _ = avaliar(Xdf, y, cfg, n_splits=5, cv_seed=cs, seeds=seeds,
                           ajustar=True)
            f1s.append(r["f1"]); f1s_aj.append(r["f1_aj"]); pesos.append(r["pesos"])
        print(f"  {n_seeds} seed(s): argmax={np.mean(f1s):.4f}  "
              f"ajustado={np.mean(f1s_aj):.4f}  "
              f"(por cv-seed: {[round(x,4) for x in f1s_aj]})  "
              f"w~{np.round(np.mean(pesos,axis=0),2)}  {time.time()-t:.0f}s", flush=True)
        resumo.append(dict(nome=nome, cfg=cfg, n_seeds=n_seeds,
                           f1=float(np.mean(f1s)), f1_aj=float(np.mean(f1s_aj))))

print("\n" + "=" * 84)
print("HOLDOUT 25% random_state=42 (mesmo protocolo do caderno)")
print("=" * 84)
for item in cfgs:
    cfg = item["cfg"]
    nome = f"{'-'.join(map(str,cfg['units']))}/{cfg['ativacao']}"
    r, _, _ = avaliar_holdout(Xdf, y, cfg, seeds=tuple(range(5)))
    print(f"  {nome:22s} argmax={r['f1']:.4f}  ajustado={r['f1_aj']:.4f}  w={r['pesos']}")

with open("_res05_ensemble.json", "w", encoding="utf-8") as f:
    json.dump(resumo, f, indent=2, ensure_ascii=False, default=str)

print("\n=== MELHORES (por f1 ajustado, media de 2 cv-seeds) ===")
for r in sorted(resumo, key=lambda x: -x["f1_aj"])[:6]:
    print(f"  {r['f1_aj']:.4f}  {r['nome']:22s} seeds={r['n_seeds']}")
