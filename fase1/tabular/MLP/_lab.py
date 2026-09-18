"""
Laboratório de experimentos MLP para risco_evasao.

Harness em PyTorch puro (rápido) para varrer arquiteturas, ativações e
hiperparâmetros. A configuração vencedora é depois reescrita em Keras
(estilo das aulas) e verificada.
"""
import math, time, itertools, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   QuantileTransformer, PowerTransformer)
from sklearn.metrics import f1_score

torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
DEV = "cpu"
RS = 42

# ----------------------------------------------------------------------------
# Dados e engenharia de features
# ----------------------------------------------------------------------------
CURSOS = [c for c in pd.read_csv("treino.csv", nrows=1).columns if c.startswith("curso_")]


def carregar():
    d = pd.read_csv("treino.csv")
    return d.drop(columns=["risco_evasao"]), d["risco_evasao"].values.astype("int64")


def engenharia(df, variante="v2"):
    """variante 'v0' = cru; 'v1' = deltas/medias; 'v2' = v1 + flags e interações."""
    X = df.copy()
    if variante == "v0":
        return X
    if variante == "v2_sem_onehot":
        return engenharia(df, "v2").drop(columns=CURSOS)

    n1, n2 = X["uc_1sem_nota_media"], X["uc_2sem_nota_media"]
    t1, t2 = X["taxa_aprovacao_1sem"], X["taxa_aprovacao_2sem"]

    X["delta_nota"] = n2 - n1
    X["delta_taxa"] = t2 - t1
    X["media_nota"] = (n1 + n2) / 2
    X["media_taxa"] = (t1 + t2) / 2
    X["min_taxa"] = np.minimum(t1, t2)
    X["min_nota"] = np.minimum(n1, n2)

    if variante == "v1":
        return X

    # 0.0 em nota/taxa é ausência estrutural de avaliação, não desempenho ruim
    X["sem_aval_1"] = (n1 == 0).astype("int64")
    X["sem_aval_2"] = (n2 == 0).astype("int64")
    X["sem_aval_ambos"] = ((n1 == 0) & (n2 == 0)).astype("int64")
    X["desempenho_1"] = n1 * t1
    X["desempenho_2"] = n2 * t2
    X["media_desempenho"] = (X["desempenho_1"] + X["desempenho_2"]) / 2
    X["instabilidade_nota"] = (n2 - n1).abs()
    X["instabilidade_taxa"] = (t2 - t1).abs()
    X["financeiro_ok"] = X["mensalidades_em_dia"] * (1 - X["devedor"])
    X["risco_financeiro"] = X["devedor"] + (1 - X["mensalidades_em_dia"])
    X["escolaridade_pais"] = X["escolaridade_mae"] + X["escolaridade_pai"]
    X["idade_log"] = np.log1p(X["idade_na_matricula"])
    X["ingresso_tardio"] = (X["idade_na_matricula"] > 22).astype("int64")
    X["curso_idx"] = X[CURSOS].values.argmax(1) * X[CURSOS].values.max(1)
    if variante == "v2":
        return X

    # ---------------- v3: mais sinais sobre trajetória e contexto ----------------
    X["reprovou_1"] = (t1 < 1).astype("int64")
    X["reprovou_2"] = (t2 < 1).astype("int64")
    X["n_semestres_perfeitos"] = (t1 == 1).astype("int64") + (t2 == 1).astype("int64")
    X["nunca_aprovou"] = ((t1 == 0) & (t2 == 0)).astype("int64")
    X["piorou"] = (t2 < t1).astype("int64")
    X["melhorou"] = (t2 > t1).astype("int64")
    X["taxa_x_nota_delta"] = X["delta_nota"] * X["delta_taxa"]
    X["nota_sobre_taxa_1"] = n1 / (t1 + 0.1)
    X["nota_sobre_taxa_2"] = n2 / (t2 + 0.1)
    # contexto socioeconômico agregado
    X["ocupacao_pais"] = X["ocupacao_mae"] + X["ocupacao_pai"]
    X["pais_sem_escolaridade"] = ((X["escolaridade_mae"] == 0) &
                                  (X["escolaridade_pai"] == 0)).astype("int64")
    X["apoio_financeiro"] = X["bolsista"] - X["devedor"]
    X["vulnerabilidade"] = (X["devedor"] + (1 - X["mensalidades_em_dia"])
                            + (1 - X["bolsista"]) + X["deslocado"])
    # desalinhamento com a escolha do curso
    X["nao_primeira_opcao_deslocado"] = ((1 - X["primeira_opcao"]) * X["deslocado"])
    X["idade_x_media_taxa"] = X["idade_na_matricula"] * X["media_taxa"]
    X["qualif_x_nota"] = X["qualificacao_anterior"] * X["media_nota"]

    if variante == "v3":
        return X
    if variante == "v3_sem_onehot":
        # 17 colunas one-hot muito esparsas podem só adicionar ruído para a MLP
        return X.drop(columns=CURSOS)
    raise ValueError(variante)


SCALERS = {
    "standard": lambda: StandardScaler(),
    "minmax": lambda: MinMaxScaler(),
    "robust": lambda: RobustScaler(),
    "quantile": lambda: QuantileTransformer(output_distribution="normal",
                                           n_quantiles=500, random_state=RS),
    "power": lambda: PowerTransformer(),
}

# ----------------------------------------------------------------------------
# Rede
# ----------------------------------------------------------------------------
ATIVACOES = {
    "relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "selu": nn.SELU,
    "gelu": nn.GELU, "swish": nn.SiLU, "leaky_relu": nn.LeakyReLU,
    "softplus": nn.Softplus, "mish": nn.Mish, "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    def __init__(self, n_in, units, ativacao="relu", dropout=0.2, batchnorm=False,
                 n_out=3):
        super().__init__()
        Act = ATIVACOES[ativacao]
        camadas, prev = [], n_in
        for u in units:
            camadas.append(nn.Linear(prev, u))
            if batchnorm:
                camadas.append(nn.BatchNorm1d(u))
            camadas.append(Act())
            if dropout > 0:
                camadas.append(nn.AlphaDropout(dropout) if ativacao == "selu"
                               else nn.Dropout(dropout))
            prev = u
        camadas.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*camadas)
        self._init(ativacao)

    def _init(self, ativacao):
        for m in self.net:
            if isinstance(m, nn.Linear):
                if ativacao == "selu":
                    nn.init.normal_(m.weight, 0, math.sqrt(1 / m.in_features))
                else:
                    nn.init.xavier_uniform_(m.weight)  # GlorotUniform (aula 2)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


CFG_PADRAO = dict(
    units=(64, 32), ativacao="relu", dropout=0.2, batchnorm=False,
    otimizador="adam", lr=1e-3, weight_decay=0.0, batch_size=64,
    epochs=400, patience=30, class_weight=True, label_smoothing=0.0,
    scaler="standard", features="v2", monitor="val_loss",
)


def _otimizador(nome, params, lr, wd):
    if nome == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if nome == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd or 1e-2)
    if nome == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    if nome == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    if nome == "nadam":
        return torch.optim.NAdam(params, lr=lr, weight_decay=wd)
    raise ValueError(nome)


def treinar(Xtr, ytr, Xes, yes, cfg, seed=RS):
    """Treina com early stopping em (Xes, yes). Devolve modelo e nº de épocas."""
    cfg = {**CFG_PADRAO, **cfg}
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr)
    Xes_t = torch.tensor(Xes, dtype=torch.float32)
    yes_t = torch.tensor(yes)

    if cfg["class_weight"]:
        cnt = np.bincount(ytr, minlength=3)
        w = torch.tensor(len(ytr) / (3 * np.maximum(cnt, 1)), dtype=torch.float32)
    else:
        w = None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])
    crit_val = nn.CrossEntropyLoss(weight=w)

    mdl = MLP(Xtr.shape[1], cfg["units"], cfg["ativacao"], cfg["dropout"],
              cfg["batchnorm"]).to(DEV)
    opt = _otimizador(cfg["otimizador"], mdl.parameters(), cfg["lr"], cfg["weight_decay"])

    n, bs = len(Xtr_t), cfg["batch_size"]
    melhor, melhor_ep, melhor_st, espera = -np.inf, 0, None, 0
    g = torch.Generator().manual_seed(seed)

    for ep in range(1, cfg["epochs"] + 1):
        mdl.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2 and cfg["batchnorm"]:
                continue
            opt.zero_grad()
            loss = crit(mdl(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

        mdl.eval()
        with torch.no_grad():
            logits = mdl(Xes_t)
            if cfg["monitor"] == "val_loss":
                score = -crit_val(logits, yes_t).item()
            else:  # val_f1
                score = f1_score(yes, logits.argmax(1).numpy(), average="macro")

        if score > melhor + 1e-6:
            melhor, melhor_ep, espera = score, ep, 0
            melhor_st = copy.deepcopy(mdl.state_dict())
        else:
            espera += 1
            if espera >= cfg["patience"]:
                break

    if melhor_st is not None:
        mdl.load_state_dict(melhor_st)
    mdl.eval()
    return mdl, melhor_ep, ep


@torch.no_grad()
def prever_proba(mdl, X):
    return torch.softmax(mdl(torch.tensor(X, dtype=torch.float32)), 1).numpy()


# ----------------------------------------------------------------------------
# Avaliação
# ----------------------------------------------------------------------------
def preparar(Xdf_tr, Xdf_te, cfg):
    cfg = {**CFG_PADRAO, **cfg}
    A = engenharia(Xdf_tr, cfg["features"]).values.astype("float64")
    B = engenharia(Xdf_te, cfg["features"]).values.astype("float64")
    sc = SCALERS[cfg["scaler"]]().fit(A)
    return sc.transform(A).astype("float32"), sc.transform(B).astype("float32"), sc


def avaliar_cv(Xdf, y, cfg, n_splits=4, seeds=(42,), es_frac=0.18, retornar_oof=False):
    """CV estratificada. Dentro de cada fold, separa uma fatia para early stopping."""
    cfg = {**CFG_PADRAO, **cfg}
    oof = np.zeros((len(y), 3))
    scores, epocas = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RS)
    for f, (itr, iva) in enumerate(skf.split(Xdf, y)):
        Xtr_df, Xva_df = Xdf.iloc[itr], Xdf.iloc[iva]
        ytr, yva = y[itr], y[iva]
        Xtr_s, Xva_s, _ = preparar(Xtr_df, Xva_df, cfg)
        itr2, ies = train_test_split(np.arange(len(ytr)), test_size=es_frac,
                                     random_state=RS, stratify=ytr)
        proba = np.zeros((len(iva), 3))
        for sd in seeds:
            mdl, best_ep, tot = treinar(Xtr_s[itr2], ytr[itr2], Xtr_s[ies], ytr[ies],
                                        cfg, seed=sd)
            proba += prever_proba(mdl, Xva_s)
            epocas.append(best_ep)
        proba /= len(seeds)
        oof[iva] = proba
        scores.append(f1_score(yva, proba.argmax(1), average="macro"))
    res = dict(f1=float(np.mean(scores)), std=float(np.std(scores)),
               f1_oof=float(f1_score(y, oof.argmax(1), average="macro")),
               epocas=float(np.mean(epocas)), folds=[round(s, 4) for s in scores])
    return (res, oof) if retornar_oof else res
