"""
Extensões do laboratório: regra de decisão para F1-macro e ensemble.

Por que existe: argmax da softmax maximiza acurácia, não F1-macro. Com a classe
"Médio" em 18% dos casos, deslocar o limiar em favor dela troca um pouco de
acurácia por bastante F1-macro. Os multiplicadores são ajustados SEMPRE numa
fatia interna (a mesma do early stopping), nunca no fold de avaliação.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

from _lab import (CFG_PADRAO, preparar, treinar, prever_proba, engenharia,
                  SCALERS, RS)


def ajustar_pesos(proba, y, n_rodadas=4, grade=None):
    """Busca por coordenadas nos multiplicadores por classe (w0=1 fixo)."""
    if grade is None:
        grade = np.concatenate([np.linspace(0.4, 2.5, 43), [3.0, 3.5, 4.0]])
    w = np.ones(3)
    melhor = f1_score(y, (proba * w).argmax(1), average="macro")
    for _ in range(n_rodadas):
        mudou = False
        for c in (1, 2):
            atual = w[c]
            for v in grade:
                w[c] = v
                s = f1_score(y, (proba * w).argmax(1), average="macro")
                if s > melhor + 1e-6:
                    melhor, atual, mudou = s, v, True
            w[c] = atual
        if not mudou:
            break
    return w, melhor


def avaliar(Xdf, y, cfg, n_splits=5, cv_seed=RS, seeds=(0,), es_frac=0.18,
            ajustar=True, verbose=False):
    """
    CV estratificada com:
      - ensemble de `seeds` redes por fold (média das probabilidades);
      - multiplicadores de classe ajustados na fatia interna de early stopping.
    Devolve F1-macro sem e com o ajuste, além das probabilidades out-of-fold.
    """
    cfg = {**CFG_PADRAO, **cfg}
    oof = np.zeros((len(y), 3))
    s_argmax, s_ajust, pesos = [], [], []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    for f, (itr, iva) in enumerate(skf.split(Xdf, y)):
        Xtr_s, Xva_s, _ = preparar(Xdf.iloc[itr], Xdf.iloc[iva], cfg)
        ytr, yva = y[itr], y[iva]
        itr2, ies = train_test_split(np.arange(len(ytr)), test_size=es_frac,
                                     random_state=RS, stratify=ytr)

        p_va = np.zeros((len(iva), 3))
        p_es = np.zeros((len(ies), 3))
        for sd in seeds:
            mdl, _, _ = treinar(Xtr_s[itr2], ytr[itr2], Xtr_s[ies], ytr[ies],
                                cfg, seed=sd)
            p_va += prever_proba(mdl, Xva_s)
            p_es += prever_proba(mdl, Xtr_s[ies])
        p_va /= len(seeds); p_es /= len(seeds)

        oof[iva] = p_va
        s_argmax.append(f1_score(yva, p_va.argmax(1), average="macro"))
        if ajustar:
            w, _ = ajustar_pesos(p_es, ytr[ies])          # ajuste na fatia interna
            pesos.append(w)
            s_ajust.append(f1_score(yva, (p_va * w).argmax(1), average="macro"))
        if verbose:
            extra = f" ajust={s_ajust[-1]:.4f} w={np.round(w,2)}" if ajustar else ""
            print(f"    fold {f}: argmax={s_argmax[-1]:.4f}{extra}", flush=True)

    out = dict(f1=float(np.mean(s_argmax)), std=float(np.std(s_argmax)),
               folds=[round(s, 4) for s in s_argmax])
    if ajustar:
        out.update(f1_aj=float(np.mean(s_ajust)), std_aj=float(np.std(s_ajust)),
                   folds_aj=[round(s, 4) for s in s_ajust],
                   pesos=np.round(np.mean(pesos, axis=0), 3).tolist())
    return out, oof


def avaliar_holdout(Xdf, y, cfg, seeds=(0,), test_size=0.25, split_seed=RS,
                    es_frac=0.18, ajustar=True):
    """Mesmo protocolo do caderno do aluno: holdout de 25%, stratify=y."""
    cfg = {**CFG_PADRAO, **cfg}
    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=test_size, random_state=split_seed,
                                stratify=y)
    Xtr_s, Xva_s, _ = preparar(Xdf.iloc[itr], Xdf.iloc[iva], cfg)
    ytr, yva = y[itr], y[iva]
    itr2, ies = train_test_split(np.arange(len(ytr)), test_size=es_frac,
                                 random_state=RS, stratify=ytr)
    p_va = np.zeros((len(iva), 3)); p_es = np.zeros((len(ies), 3))
    for sd in seeds:
        mdl, _, _ = treinar(Xtr_s[itr2], ytr[itr2], Xtr_s[ies], ytr[ies], cfg, seed=sd)
        p_va += prever_proba(mdl, Xva_s)
        p_es += prever_proba(mdl, Xtr_s[ies])
    p_va /= len(seeds); p_es /= len(seeds)
    r = {"f1": f1_score(yva, p_va.argmax(1), average="macro")}
    if ajustar:
        w, _ = ajustar_pesos(p_es, ytr[ies])
        r["f1_aj"] = f1_score(yva, (p_va * w).argmax(1), average="macro")
        r["pesos"] = np.round(w, 3).tolist()
    return r, p_va, yva
