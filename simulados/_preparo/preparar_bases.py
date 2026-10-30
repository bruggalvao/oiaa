"""Monta as bases dos seis simulados da Fase 2 a partir das fontes originais.

Uso (da raiz do repositório, com o .venv):
    .venv/Scripts/python simulados/_preparo/preparar_bases.py <pasta_com_os_brutos> [etapas...]

Fontes esperadas:
    fase1/tabular/MLP/treino.csv  e  fase1/nlp/treino.csv     (já estão no repositório)
    <brutos>/student/student-{mat,por}.csv                     UCI 320
    <brutos>/trashnet/dataset-resized/<classe>/*.jpg           github.com/garythung/trashnet
    <brutos>/flower_photos/<classe>/*.jpg                      TensorFlow flower_photos.tgz
    <brutos>/Fake.br-Corpus-master/size_normalized_texts/{fake,true}/*.txt

Tudo é determinístico (SEMENTE = 2026). O rótulo do teste nunca entra na pasta do desafio:
fica em simulados/_gabaritos/, que só o corretor lê. Nas bases da Fase 1, o teste oficial
não tem rótulo; por isso o teste oculto daqui é recortado das linhas rotuladas.
"""
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SEMENTE = 2026
RAIZ = pathlib.Path(__file__).resolve().parents[1]
REPO = RAIZ.parent
BRUTOS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "brutos")
GABARITOS = RAIZ / "_gabaritos"
GABARITOS.mkdir(exist_ok=True)
LADO_CURTO = 256  # as fotos ficam com 256 px no lado menor: o preparo dos pesos corta 224


def _sem_categoria_inedita(pub, tst, categoricas):
    """Linha de teste com categoria que o público não tem vai para o público.

    A rede com embeddings e o TabNet precisam saber, antes do treino, quantos valores cada
    categoria tem. Se o teste trouxesse um valor inédito, nenhum dos dois conseguiria prever.
    """
    for c in categoricas:
        ineditas = ~tst[c].isin(set(pub[c]))
        if ineditas.any():
            pub = pd.concat([pub, tst[ineditas]])
            tst = tst[~ineditas]
    return pub, tst


def _gravar_tabela(pasta, pub, tst, col_id, col_alvo, nome):
    pasta = RAIZ / pasta
    pasta.mkdir(parents=True, exist_ok=True)
    pub = pub.sample(frac=1, random_state=SEMENTE).reset_index(drop=True)
    tst = tst.reset_index(drop=True)
    tst.insert(0, col_id, np.arange(len(tst)))
    pub.to_csv(pasta / "treino.csv", index=False)
    tst.drop(columns=[col_alvo]).to_csv(pasta / "teste.csv", index=False)
    tst[[col_id, col_alvo]].to_csv(GABARITOS / f"{nome}.csv", index=False)
    print(f"{pasta.relative_to(RAIZ)}: público {len(pub)} | teste {len(tst)} | "
          f"{pub[col_alvo].value_counts(normalize=True).round(3).to_dict()}")


# --------------------------------------------------------------------------- TABULAR
def farol():
    """Dados oficiais da Operação Farol, com o curso de volta a uma coluna de categoria."""
    d = pd.read_csv(REPO / "fase1/tabular/MLP/treino.csv")
    cursos = [c for c in d.columns if c.startswith("curso_")]
    curso = d[cursos].idxmax(axis=1).str.removeprefix("curso_")
    d = d.drop(columns=cursos)
    d.insert(d.columns.get_loc("risco_evasao"), "curso", curso)
    categoricas = ["curso", "ocupacao_mae", "ocupacao_pai", "qualificacao_anterior"]
    pub, tst = train_test_split(d, test_size=0.2, stratify=d.risco_evasao, random_state=SEMENTE)
    pub, tst = _sem_categoria_inedita(pub, tst, categoricas)
    _gravar_tabela("tabular/farol", pub, tst, "id_estudante", "risco_evasao", "farol")


def boletim():
    """UCI Student Performance, com as categorias em texto (como chegam de uma secretaria)."""
    m = pd.read_csv(BRUTOS / "student/student-mat.csv", sep=";").assign(disc="matematica")
    p = pd.read_csv(BRUTOS / "student/student-por.csv", sep=";").assign(disc="portugues")
    d = pd.concat([m, p], ignore_index=True)
    chave = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
             "Mjob", "Fjob", "reason", "nursery", "internet"]
    grupo = d[chave].astype(str).agg("|".join, axis=1).factorize()[0]

    ocup = {"at_home": "em_casa", "services": "servicos", "health": "saude",
            "teacher": "professor", "other": "outra"}
    sim = lambda s: (s == "yes").astype(int)
    t = pd.DataFrame({
        "escola": d.school.map({"GP": "escola_a", "MS": "escola_b"}),
        "disciplina": d.disc,
        "genero": (d.sex == "M").astype(int),
        "idade": d.age,
        "mora_zona_urbana": (d.address == "U").astype(int),
        "familia_grande": (d.famsize == "GT3").astype(int),
        "pais_moram_juntos": (d.Pstatus == "T").astype(int),
        "escolaridade_mae": d.Medu,
        "escolaridade_pai": d.Fedu,
        "ocupacao_mae": d.Mjob.map(ocup),
        "ocupacao_pai": d.Fjob.map(ocup),
        "motivo_escolha": d.reason.map({"home": "perto_de_casa", "reputation": "reputacao",
                                        "course": "curso", "other": "outro"}),
        "responsavel": d.guardian.map({"mother": "mae", "father": "pai", "other": "outro"}),
        "tempo_deslocamento": d.traveltime,
        "horas_estudo": d.studytime,
        "reprovacoes_anteriores": d.failures,
        "reforco_escolar": sim(d.schoolsup),
        "apoio_familiar": sim(d.famsup),
        "aulas_particulares": sim(d.paid),
        "atividades_extra": sim(d.activities),
        "quer_ensino_superior": sim(d.higher),
        "internet_em_casa": sim(d.internet),
        "namorando": sim(d.romantic),
        "relacao_familiar": d.famrel,
        "tempo_livre": d.freetime,
        "sai_com_amigos": d.goout,
        "alcool_dias_semana": d.Dalc,
        "alcool_fim_de_semana": d.Walc,
        "saude": d.health,
        "faltas": d.absences,
        "nota_1bim": d.G1,
    })
    # G3 (nota final, 0-20) vira o risco; G2 sai porque quase determina G3.
    t["risco_reprovacao"] = pd.cut(d.G3, [-1, 9, 11, 20], labels=[2, 1, 0]).astype(int)
    sgk = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEMENTE)
    i_pub, i_tst = next(sgk.split(t, t.risco_reprovacao, grupo))
    pub, tst = _sem_categoria_inedita(t.iloc[i_pub], t.iloc[i_tst],
                                      ["escola", "disciplina", "ocupacao_mae", "ocupacao_pai",
                                       "motivo_escolha", "responsavel"])
    _gravar_tabela("tabular/boletim", pub, tst, "id_estudante", "risco_reprovacao", "boletim")


# ----------------------------------------------------------------------------- VISÃO
def _foto(caminho):
    im = Image.open(caminho).convert("RGB")
    escala = LADO_CURTO / min(im.size)
    if escala < 1:
        im = im.resize((round(im.width * escala), round(im.height * escala)), Image.LANCZOS)
    return im


def _gravar_fotos(pasta, arquivos, rotulos, n_teste, nome):
    """fotos.zip com treino/<classe>/NNNN.jpg e teste/NNNN.jpg (o id da foto é o número)."""
    pasta = RAIZ / pasta
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos, rotulos = np.array(arquivos), np.array(rotulos)
    i_pub, i_tst = train_test_split(np.arange(len(arquivos)), test_size=n_teste,
                                    stratify=rotulos, random_state=SEMENTE)
    rng = np.random.default_rng(SEMENTE)
    i_pub = rng.permutation(i_pub)  # a ordem dos nomes não pode entregar a classe
    with zipfile.ZipFile(pasta / "fotos.zip", "w", zipfile.ZIP_STORED) as z:
        for k, i in enumerate(i_pub):
            with z.open(f"treino/{rotulos[i]}/{k:04d}.jpg", "w") as f:
                _foto(arquivos[i]).save(f, "JPEG", quality=85)
        for k, i in enumerate(i_tst):
            with z.open(f"teste/{k:04d}.jpg", "w") as f:
                _foto(arquivos[i]).save(f, "JPEG", quality=85)
    pd.DataFrame({"id_foto": np.arange(len(i_tst)), "classe": rotulos[i_tst]}).to_csv(
        GABARITOS / f"{nome}.csv", index=False)
    u, c = np.unique(rotulos[i_pub], return_counts=True)
    tam = (pasta / "fotos.zip").stat().st_size / 2**20
    print(f"{pasta.relative_to(RAIZ)}: público {len(i_pub)} | teste {len(i_tst)} | "
          f"{dict(zip(u, c))} | {tam:.1f} MB")


def recomeco():
    """TrashNet sem a classe 'trash': as mesmas 2.390 fotos da Missão Recomeço, em resolução cheia."""
    nomes = {"paper": "papel", "cardboard": "papelao", "plastic": "plastico",
             "glass": "vidro", "metal": "metal"}
    arquivos, rotulos = [], []
    for pasta, nome in nomes.items():
        for f in sorted((BRUTOS / "trashnet/dataset-resized" / pasta).glob("*.jpg")):
            arquivos.append(f); rotulos.append(nome)
    # 240 no teste, como na Fase 1.
    _gravar_fotos("visao/recomeco", arquivos, rotulos, 240, "recomeco")


def florada():
    """TF Flowers (Flickr, CC BY): metade de cada espécie, em resolução cheia."""
    nomes = {"daisy": "margarida", "dandelion": "dente_de_leao", "roses": "rosa",
             "sunflowers": "girassol", "tulips": "tulipa"}
    rng = np.random.default_rng(SEMENTE)
    arquivos, rotulos = [], []
    for pasta, nome in nomes.items():
        todos = sorted((BRUTOS / "flower_photos" / pasta).glob("*.jpg"))
        for f in rng.choice(todos, round(len(todos) * 0.5), replace=False):
            arquivos.append(f); rotulos.append(nome)
    _gravar_fotos("visao/florada", arquivos, rotulos, 300, "florada")


# ------------------------------------------------------------------------------- NLP
def voz_do_cliente():
    """Acervo oficial da Operação Voz do Cliente; 900 linhas rotuladas viram o teste oculto."""
    d = pd.read_csv(REPO / "fase1/nlp/treino.csv")
    pub, tst = train_test_split(d, test_size=900, stratify=d.rotulo, random_state=SEMENTE)
    _gravar_tabela("nlp/voz_do_cliente", pub, tst, "id_mensagem", "rotulo", "voz_do_cliente")


def checagem():
    """Fake.br (tamanho normalizado). 0 = falsa (crítica, ~30%), 1 = verdadeira."""
    base = BRUTOS / "Fake.br-Corpus-master/size_normalized_texts"
    ler = lambda p: " ".join(p.read_text(encoding="utf-8", errors="replace").split())
    ordenar = lambda pasta: sorted(pasta.glob("*.txt"), key=lambda p: int(p.stem))
    falsas = [ler(p) for p in ordenar(base / "fake")]
    verdadeiras = [ler(p) for p in ordenar(base / "true")]
    rng = np.random.default_rng(SEMENTE)
    falsas = list(rng.choice(falsas, round(len(verdadeiras) * 0.30 / 0.70), replace=False))
    d = pd.DataFrame({"mensagem": falsas + verdadeiras,
                      "rotulo": [0] * len(falsas) + [1] * len(verdadeiras)})
    d = d.drop_duplicates("mensagem").sample(frac=1, random_state=SEMENTE)
    pub, tst = train_test_split(d, test_size=700, stratify=d.rotulo, random_state=SEMENTE)
    _gravar_tabela("nlp/checagem", pub, tst, "id_mensagem", "rotulo", "checagem")


if __name__ == "__main__":
    etapas = {"farol": farol, "boletim": boletim, "recomeco": recomeco, "florada": florada,
              "voz_do_cliente": voz_do_cliente, "checagem": checagem}
    for nome in (sys.argv[2:] or etapas):
        etapas[nome]()
