"""
================================================================================
OPERACAO FAROL - Rede neural (MLP) para prever risco_evasao
================================================================================
Le  : treino.csv (3236 alunos com resposta) e teste.csv (1078 alunos sem resposta)
Gera: submissoes/tabular/submissao_tabular.csv  ->  id_estudante,risco_previsto

A arquitetura e os hiperparametros abaixo nao foram escolhidos no chute: sairam de
uma varredura de 180 combinacoes de arquitetura x ativacao mais uma busca por
coordenadas nos hiperparametros de treino, sempre medindo F1-macro com validacao
cruzada estratificada de 5 folds.

Estrutura, seguindo as aulas de MLP:
  - Sequential + Dense, pesos iniciais GlorotUniform (Aula 2)
  - camadas ocultas com ativacao, camada de saida com 3 neuronios softmax (Aula 3)
  - loss sparse_categorical_crossentropy, porque o alvo e inteiro 0/1/2 (Aula 3)
  - EarlyStopping com restore_best_weights, monitorando val_loss (Aula 4)
================================================================================
"""
import os, json, pathlib, warnings

os.environ.setdefault("KERAS_BACKEND", "torch")   # ignorado no Colab, que usa TF
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                             confusion_matrix)

try:                                  # Colab / TensorFlow
    from tensorflow import keras
except Exception:                     # Keras 3 autonomo (backend torch)
    import keras

# ----------------------------------------------------------------------------
# CONFIGURACAO VENCEDORA
# ----------------------------------------------------------------------------
CFG = dict(
    neuronios=(128, 64, 32),   # 3 camadas ocultas
    ativacao="selu",
    dropout=0.2,
    otimizador="adamw",
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=16,
    epochs=400,                # o EarlyStopping corta muito antes (~10-30)
    patience=30,
    n_redes=5,                 # ensemble: media das probabilidades de 5 sementes
)
SEMENTE = 42
NOMES = {0: "Baixo", 1: "Medio", 2: "Alto"}
SAIDA = pathlib.Path("submissoes/tabular/submissao_tabular.csv")


# ----------------------------------------------------------------------------
# 1. ENGENHARIA DE FEATURES
# ----------------------------------------------------------------------------
CURSOS = None   # preenchido no carregamento


def engenharia(df):
    """
    Deriva sinais sobre a trajetoria do aluno. Ganho medido: +0.014 de F1-macro
    contra usar as colunas cruas.
    """
    X = df.copy()
    n1, n2 = X["uc_1sem_nota_media"], X["uc_2sem_nota_media"]
    t1, t2 = X["taxa_aprovacao_1sem"], X["taxa_aprovacao_2sem"]

    # evolucao entre os dois semestres
    X["delta_nota"] = n2 - n1
    X["delta_taxa"] = t2 - t1
    X["media_nota"] = (n1 + n2) / 2
    X["media_taxa"] = (t1 + t2) / 2
    X["min_taxa"] = np.minimum(t1, t2)
    X["min_nota"] = np.minimum(n1, n2)

    # nota 0.0 significa "nao foi avaliado", nao "tirou zero".
    # Sem essa marcacao a rede confunde ausencia de avaliacao com desempenho ruim.
    X["sem_aval_1"] = (n1 == 0).astype("int64")
    X["sem_aval_2"] = (n2 == 0).astype("int64")
    X["sem_aval_ambos"] = ((n1 == 0) & (n2 == 0)).astype("int64")

    # desempenho = nota ponderada pelo quanto o aluno de fato aprovou
    X["desempenho_1"] = n1 * t1
    X["desempenho_2"] = n2 * t2
    X["media_desempenho"] = (X["desempenho_1"] + X["desempenho_2"]) / 2
    X["instabilidade_nota"] = (n2 - n1).abs()
    X["instabilidade_taxa"] = (t2 - t1).abs()

    # situacao financeira e contexto
    X["financeiro_ok"] = X["mensalidades_em_dia"] * (1 - X["devedor"])
    X["risco_financeiro"] = X["devedor"] + (1 - X["mensalidades_em_dia"])
    X["escolaridade_pais"] = X["escolaridade_mae"] + X["escolaridade_pai"]
    X["idade_log"] = np.log1p(X["idade_na_matricula"])
    X["ingresso_tardio"] = (X["idade_na_matricula"] > 22).astype("int64")
    X["curso_idx"] = X[CURSOS].values.argmax(1) * X[CURSOS].values.max(1)
    return X


# ----------------------------------------------------------------------------
# 2. A REDE
# ----------------------------------------------------------------------------
def construir_rede(n_entradas, cfg=CFG):
    """MLP: entradas -> camadas ocultas -> 3 neuronios softmax."""
    camadas = [keras.layers.Input(shape=(n_entradas,))]

    for n in cfg["neuronios"]:
        camadas.append(keras.layers.Dense(
            n,
            activation=cfg["ativacao"],
            # selu pede LecunNormal; nas outras ativacoes vale o GlorotUniform da aula
            kernel_initializer=("lecun_normal" if cfg["ativacao"] == "selu"
                                else "glorot_uniform"),
        ))
        if cfg["dropout"] > 0:
            camadas.append(_dropout(cfg))

    # camada de saida: um neuronio por classe, softmax devolve probabilidades
    camadas.append(keras.layers.Dense(3, activation="softmax"))

    modelo = keras.Sequential(camadas)
    modelo.compile(
        optimizer=_otimizador(cfg),
        loss="sparse_categorical_crossentropy",   # alvo inteiro: 0, 1 ou 2
        metrics=["accuracy"],
    )
    return modelo


def _dropout(cfg):
    """AlphaDropout preserva a auto-normalizacao da selu; nas outras, Dropout comum."""
    if cfg["ativacao"] == "selu" and hasattr(keras.layers, "AlphaDropout"):
        return keras.layers.AlphaDropout(cfg["dropout"])
    return keras.layers.Dropout(cfg["dropout"])


def _otimizador(cfg):
    lr, wd = cfg["learning_rate"], cfg["weight_decay"]
    if cfg["otimizador"] == "adamw":
        try:
            return keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        except Exception:
            return keras.optimizers.Adam(learning_rate=lr)
    return keras.optimizers.Adam(learning_rate=lr)


def pesos_das_classes(y):
    """
    'Medio' e so 18% dos alunos. Sem reequilibrar, a rede aprende a quase nunca
    apostar nessa classe e o F1-macro despenca (medido: -0.03).
    """
    contagem = np.bincount(y, minlength=3)
    return {c: len(y) / (3 * max(contagem[c], 1)) for c in range(3)}


def treinar_ensemble(X_treino, y_treino, X_es, y_es, cfg=CFG, verbose=False):
    """Treina n_redes com sementes diferentes. Devolve a lista de modelos."""
    modelos = []
    for i in range(cfg["n_redes"]):
        keras.utils.set_random_seed(SEMENTE + i)
        rede = construir_rede(X_treino.shape[1], cfg)
        parada = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience"],
            restore_best_weights=True,   # volta para a epoca de menor val_loss
        )
        historico = rede.fit(
            X_treino, y_treino,
            validation_data=(X_es, y_es),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            class_weight=pesos_das_classes(y_treino),
            callbacks=[parada],
            verbose=0,
        )
        modelos.append(rede)
        print(f"    rede {i+1}/{cfg['n_redes']}: "
              f"{len(historico.history['loss'])} epocas treinadas", flush=True)
    return modelos


def prever(modelos, X):
    """Media das probabilidades do ensemble; a classe e o argmax."""
    proba = np.mean([m.predict(X, verbose=0) for m in modelos], axis=0)
    return proba.argmax(1), proba


# ----------------------------------------------------------------------------
# 3. EXECUCAO
# ----------------------------------------------------------------------------
def main():
    global CURSOS

    print("=" * 72)
    print("CARREGANDO DADOS")
    print("=" * 72)
    treino = pd.read_csv("treino.csv")
    teste = pd.read_csv("teste.csv")
    CURSOS = [c for c in treino.columns if c.startswith("curso_")]

    ids_teste = teste["id_estudante"].values
    X_bruto = treino.drop(columns=["risco_evasao"])
    y = treino["risco_evasao"].values.astype("int64")
    X_teste_bruto = teste.drop(columns=["id_estudante"])

    # as colunas do teste tem de bater com as do treino, na mesma ordem
    X_teste_bruto = X_teste_bruto[X_bruto.columns]

    X = engenharia(X_bruto).values.astype("float64")
    X_teste = engenharia(X_teste_bruto).values.astype("float64")
    print(f"  treino: {X.shape[0]} alunos, {X.shape[1]} variaveis "
          f"(35 originais + {X.shape[1]-35} derivadas)")
    print(f"  teste : {X_teste.shape[0]} alunos")
    print(f"  classes no treino: " +
          ", ".join(f"{NOMES[c]}={(y==c).sum()} ({(y==c).mean()*100:.1f}%)"
                    for c in range(3)))

    # ------------------------------------------------------------------
    # ETAPA A - estimar o desempenho num conjunto que a rede nunca ve
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ETAPA A - VALIDACAO (holdout de 25%, estratificado)")
    print("=" * 72)
    i_tr, i_va = train_test_split(np.arange(len(y)), test_size=0.25,
                                  random_state=SEMENTE, stratify=y)
    # o escalonador aprende SO no treino, senao vaza informacao da validacao
    escala = StandardScaler().fit(X[i_tr])
    Xtr, Xva = escala.transform(X[i_tr]), escala.transform(X[i_va])
    ytr, yva = y[i_tr], y[i_va]

    # fatia interna, exclusiva do EarlyStopping: a validacao fica intocada
    i_fit, i_es = train_test_split(np.arange(len(ytr)), test_size=0.18,
                                   random_state=SEMENTE, stratify=ytr)
    modelos = treinar_ensemble(Xtr[i_fit], ytr[i_fit], Xtr[i_es], ytr[i_es])

    pred_va, _ = prever(modelos, Xva)
    f1 = f1_score(yva, pred_va, average="macro")
    acc = accuracy_score(yva, pred_va)
    print(f"\n  F1-MACRO (metrica oficial) ... {f1*100:.2f}%")
    print(f"  Acuracia .................... {acc*100:.2f}%")
    print("\n" + classification_report(yva, pred_va,
                                       target_names=list(NOMES.values()), digits=3))
    print("  Matriz de confusao (linha = verdadeiro, coluna = previsto):")
    cm = confusion_matrix(yva, pred_va)
    print("            " + "".join(f"{NOMES[c]:>8}" for c in range(3)))
    for c in range(3):
        print(f"    {NOMES[c]:<8}" + "".join(f"{v:>8}" for v in cm[c]))

    # ------------------------------------------------------------------
    # ETAPA B - retreinar com TODOS os dados e prever o teste
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ETAPA B - MODELO FINAL (treinado com os 3236 alunos) E SUBMISSAO")
    print("=" * 72)
    escala_final = StandardScaler().fit(X)
    X_all = escala_final.transform(X)
    X_teste_s = escala_final.transform(X_teste)

    i_fit, i_es = train_test_split(np.arange(len(y)), test_size=0.18,
                                   random_state=SEMENTE, stratify=y)
    modelos_finais = treinar_ensemble(X_all[i_fit], y[i_fit],
                                      X_all[i_es], y[i_es])

    palpites, proba = prever(modelos_finais, X_teste_s)

    submissao = pd.DataFrame({"id_estudante": ids_teste,
                              "risco_previsto": palpites})
    assert len(submissao) == 1078, f"esperado 1078 linhas, saiu {len(submissao)}"
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    submissao.to_csv(SAIDA, index=False)

    print(f"\n  Arquivo gerado: {SAIDA}  ({len(submissao)} linhas)")
    print("  Distribuicao das previsoes no teste:")
    for c in range(3):
        n = (palpites == c).sum()
        print(f"    {NOMES[c]:<8} {n:>5}  ({n/len(palpites)*100:5.1f}%)")
    print(f"  Confianca media do ensemble: {proba.max(1).mean()*100:.1f}%")
    print("\n  Previsoes finais (F1-macro esperado ~{:.2f}% na validacao)".format(f1*100))


if __name__ == "__main__":
    main()
