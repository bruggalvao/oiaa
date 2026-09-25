"""Corrige uma submissão de simulado contra o gabarito oculto, como a plataforma faria.

Uso (da raiz do repositório):
    .venv/Scripts/python simulados/corrigir.py <desafio> <caminho/da/submissao.csv>

Desafios: boletim, batimento, sentinela, florada, pos_venda, checagem.
Imprime F1-macro (métrica oficial), acurácia, a métrica de apoio de cada desafio e a matriz
de confusão. Com --registrar, anexa o resultado a simulados/_gabaritos/placar.csv.
"""
import argparse
import datetime
import pathlib
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

RAIZ = pathlib.Path(__file__).resolve().parent
GABARITOS = RAIZ / "_gabaritos"

# id da linha, coluna prevista, coluna do gabarito, classes em ordem, classe crítica.
DESAFIOS = {
    "boletim":   ("id_estudante", "risco_previsto", "risco_reprovacao", [0, 1, 2], 2),
    "batimento": ("id_exame", "estado_previsto", "estado_fetal", [0, 1, 2], 2),
    "sentinela": ("id_foto", "classe_prevista", "classe",
                  ["floresta", "pastagem", "lavoura", "campo", "rio"], None),
    "florada":   ("id_foto", "classe_prevista", "classe",
                  ["margarida", "dente_de_leao", "rosa", "girassol", "tulipa"], None),
    "pos_venda": ("id_mensagem", "rotulo_previsto", "rotulo", [0, 1], 0),
    "checagem":  ("id_mensagem", "rotulo_previsto", "rotulo", [0, 1], 0),
}


def corrigir(desafio, caminho):
    col_id, col_prev, col_gab, classes, critica = DESAFIOS[desafio]
    gab = pd.read_csv(GABARITOS / f"{desafio}.csv")
    sub = pd.read_csv(caminho)

    faltam = {col_id, col_prev} - set(sub.columns)
    if faltam:
        sys.exit(f"REPROVADA: faltam as colunas {sorted(faltam)}. Esperado: {col_id},{col_prev}")
    if len(sub) != len(gab):
        sys.exit(f"REPROVADA: a submissão tem {len(sub)} linhas; o teste tem {len(gab)}.")
    if sub[col_id].duplicated().any() or set(sub[col_id]) != set(gab[col_id]):
        sys.exit(f"REPROVADA: os valores de '{col_id}' não batem com os ids do teste.")

    par = gab.merge(sub[[col_id, col_prev]], on=col_id)
    real, prev = par[col_gab], par[col_prev]
    if pd.api.types.is_numeric_dtype(real):
        prev = pd.to_numeric(prev, errors="coerce")
    invalidas = sorted(set(prev.dropna()) - set(classes)) + (["<vazio>"] if prev.isna().any() else [])
    if invalidas:
        sys.exit(f"REPROVADA: classes desconhecidas na submissão: {invalidas}. Válidas: {classes}")

    f1 = f1_score(real, prev, average="macro", labels=classes)
    acc = accuracy_score(real, prev)
    print("=" * 56)
    print(f"  SIMULADO {desafio.upper()}  ·  {len(par)} linhas corrigidas")
    print("=" * 56)
    print(f"  ⭐ F1-MACRO (métrica oficial) ...... {f1:.4f}")
    print(f"     Acurácia ........................ {acc:.4f}")
    apoio = None
    if critica is not None:
        apoio = recall_score(real, prev, labels=[critica], average="macro")
        print(f"     Revocação da classe crítica ({critica}) .. {apoio:.4f}")
    print("\n  F1 por classe:")
    for c, v in zip(classes, f1_score(real, prev, average=None, labels=classes)):
        print(f"     {str(c):<14} {v:.4f}")
    cm = pd.DataFrame(confusion_matrix(real, prev, labels=classes),
                      index=pd.Index(classes, name="real"),
                      columns=pd.Index(classes, name="previsto"))
    print("\n  Matriz de confusão (linhas = real, colunas = previsto):\n")
    print(cm.to_string())
    return f1, acc, apoio


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("desafio", choices=DESAFIOS)
    ap.add_argument("submissao", type=pathlib.Path)
    ap.add_argument("--registrar", metavar="NOME", help="anexa o resultado ao placar local")
    args = ap.parse_args()
    f1, acc, apoio = corrigir(args.desafio, args.submissao)
    if args.registrar:
        linha = pd.DataFrame([{"quando": datetime.datetime.now().isoformat(timespec="seconds"),
                               "desafio": args.desafio, "quem": args.registrar,
                               "arquivo": str(args.submissao), "f1_macro": round(f1, 4),
                               "acuracia": round(acc, 4),
                               "apoio": None if apoio is None else round(apoio, 4)}])
        placar = GABARITOS / "placar.csv"
        linha.to_csv(placar, mode="a", header=not placar.exists(), index=False)
        print(f"\n  Registrado em {placar.relative_to(RAIZ.parent)}")
