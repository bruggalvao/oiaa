"""
Etapa 3: hiperparâmetros de treino.

Busca por coordenadas (um fator por vez), partindo das melhores arquiteturas
encontradas na etapa 2. Ordem dos fatores escolhida do mais para o menos
influente. Cada rodada mantém o melhor valor encontrado até então.
"""
import time, csv, json
import numpy as np
import pandas as pd
from _lab import carregar, avaliar_cv, CFG_PADRAO

Xdf, y = carregar()

r3 = pd.read_csv("_res03_arquitetura.csv")
# critério de seleção: f1 alto e estável (penaliza desvio entre folds)
r3["score"] = r3["f1"] - 0.3 * r3["std"]
top = r3.sort_values("score", ascending=False).head(3)
print("Arquiteturas de partida:")
print(top.to_string(index=False), "\n")

FATORES = [
    ("monitor", ["val_loss", "val_f1"]),
    ("class_weight", [True, False]),
    ("batch_size", [16, 32, 64, 128, 256]),
    ("lr", [3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 1e-2]),
    ("otimizador", ["adam", "adamw", "nadam", "rmsprop", "sgd"]),
    ("dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    ("batchnorm", [False, True]),
    ("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
    ("label_smoothing", [0.0, 0.03, 0.05, 0.1]),
    ("patience", [15, 30, 60]),
]

log = open("_res04_hiper.csv", "w", newline="", encoding="utf-8")
w = csv.writer(log)
w.writerow(["arq", "fator", "valor", "f1", "std", "f1_oof", "epocas"])

resultados_finais = []
for _, linha in top.iterrows():
    units = tuple(int(x) for x in str(linha["units"]).split("-"))
    base = {"units": units, "ativacao": linha["ativacao"]}
    tag = f"{linha['units']}/{linha['ativacao']}"
    print("=" * 78)
    print(f"ARQUITETURA {tag}  (f1 inicial {linha['f1']:.4f})")
    print("=" * 78)

    cfg = dict(base)
    r = avaliar_cv(Xdf, y, cfg, n_splits=4)
    melhor_f1 = r["f1"] - 0.3 * r["std"]
    print(f"  partida: f1={r['f1']:.4f} (+-{r['std']:.4f})")

    for fator, valores in FATORES:
        atual = {**CFG_PADRAO, **cfg}[fator]
        linhas = []
        for v in valores:
            if v == atual and fator != "monitor":
                linhas.append((melhor_f1, v, None))
                continue
            t = time.time()
            r = avaliar_cv(Xdf, y, {**cfg, fator: v}, n_splits=4)
            sc = r["f1"] - 0.3 * r["std"]
            linhas.append((sc, v, r))
            w.writerow([tag, fator, str(v), round(r["f1"], 4), round(r["std"], 4),
                        round(r["f1_oof"], 4), round(r["epocas"], 1)]); log.flush()
            print(f"    {fator:16s}={str(v):10s} f1={r['f1']:.4f} "
                  f"(+-{r['std']:.4f}) ep={r['epocas']:.0f}  {time.time()-t:.0f}s",
                  flush=True)
        linhas.sort(key=lambda x: -x[0])
        if linhas[0][0] > melhor_f1 + 1e-5:
            melhor_f1 = linhas[0][0]
            cfg[fator] = linhas[0][1]
            print(f"  -> adota {fator}={linhas[0][1]}  (score {melhor_f1:.4f})")
        else:
            print(f"  -> mantem {fator}={atual}")

    r = avaliar_cv(Xdf, y, cfg, n_splits=4)
    print(f"\n  FINAL {tag}: f1={r['f1']:.4f} (+-{r['std']:.4f}) oof={r['f1_oof']:.4f}")
    print(f"  cfg = {cfg}\n", flush=True)
    resultados_finais.append({"cfg": {k: (list(v) if isinstance(v, tuple) else v)
                                      for k, v in cfg.items()},
                              "f1": r["f1"], "std": r["std"], "f1_oof": r["f1_oof"]})

log.close()
with open("_res04_melhores.json", "w", encoding="utf-8") as f:
    json.dump(resultados_finais, f, indent=2, ensure_ascii=False)

print("=" * 78)
for r in sorted(resultados_finais, key=lambda x: -x["f1"]):
    print(f"f1={r['f1']:.4f} (+-{r['std']:.4f})  {r['cfg']}")
