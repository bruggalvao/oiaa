"""Monta as bases dos seis simulados a partir das fontes públicas originais.

Uso (da raiz do repositório, com o .venv):
    .venv/Scripts/python simulados/_preparo/preparar_bases.py <pasta_com_os_brutos>

A pasta de brutos precisa conter (cada item vem do link indicado em simulados/README.md):
    student/student-mat.csv, student/student-por.csv      UCI 320
    ctg/CTG.xls                                            UCI 193
    eurosat/EuroSAT_RGB/<Classe>/*.jpg                     Zenodo 7711810
    flower_photos/<classe>/*.jpg                           TensorFlow flower_photos.tgz
    olist_reviews.csv                                      olist/work-at-olist-data
    Fake.br-Corpus-master/size_normalized_texts/{fake,true}/*.txt

Tudo é determinístico (SEMENTE = 2026). O rótulo do conjunto de teste NUNCA vai para a
pasta do desafio: ele é gravado em simulados/_gabaritos/, que só o corretor lê.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SEMENTE = 2026
BRUTOS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "brutos")
RAIZ = pathlib.Path(__file__).resolve().parents[1]
GABARITOS = RAIZ / "_gabaritos"
GABARITOS.mkdir(exist_ok=True)


def _gravar_tabela(pasta, publico, teste, col_id, col_alvo, nome_gabarito):
    pasta = RAIZ / pasta
    pasta.mkdir(parents=True, exist_ok=True)
    teste = teste.reset_index(drop=True)
    teste.insert(0, col_id, np.arange(len(teste)))
    publico.reset_index(drop=True).to_csv(pasta / "treino.csv", index=False)
    teste.drop(columns=[col_alvo]).to_csv(pasta / "teste.csv", index=False)
    teste[[col_id, col_alvo]].to_csv(GABARITOS / nome_gabarito, index=False)
    print(f"{pasta.relative_to(RAIZ)}: público {len(publico)} | teste {len(teste)} | "
          f"classes {publico[col_alvo].value_counts(normalize=True).round(3).to_dict()}")


# --------------------------------------------------------------------------- TABULAR
def boletim():
    """UCI Student Performance: risco de reprovação a partir do 1º bimestre."""
    m = pd.read_csv(BRUTOS / "student/student-mat.csv", sep=";").assign(disciplina_matematica=1)
    p = pd.read_csv(BRUTOS / "student/student-por.csv", sep=";").assign(disciplina_matematica=0)
    d = pd.concat([m, p], ignore_index=True)
    # 382 estudantes aparecem nas duas disciplinas. A chave abaixo é a do script de merge
    # dos autores; ela vira grupo para que o mesmo estudante não fique dos dois lados.
    chave = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
             "Mjob", "Fjob", "reason", "nursery", "internet"]
    grupo = d[chave].astype(str).agg("|".join, axis=1).factorize()[0]

    ocup = {"at_home": 0, "services": 1, "health": 2, "teacher": 3, "other": 4}
    sim = lambda s: (s == "yes").astype(int)
    t = pd.DataFrame({
        "escola_b": (d.school == "MS").astype(int),
        "genero": (d.sex == "M").astype(int),
        "idade": d.age,
        "mora_zona_urbana": (d.address == "U").astype(int),
        "familia_grande": (d.famsize == "GT3").astype(int),
        "pais_moram_juntos": (d.Pstatus == "T").astype(int),
        "escolaridade_mae": d.Medu,
        "escolaridade_pai": d.Fedu,
        "ocupacao_mae": d.Mjob.map(ocup),
        "ocupacao_pai": d.Fjob.map(ocup),
        "tempo_deslocamento": d.traveltime,
        "horas_estudo": d.studytime,
        "reprovacoes_anteriores": d.failures,
        "reforco_escolar": sim(d.schoolsup),
        "apoio_familiar": sim(d.famsup),
        "aulas_particulares": sim(d.paid),
        "atividades_extra": sim(d.activities),
        "fez_creche": sim(d.nursery),
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
        "disciplina_matematica": d.disciplina_matematica,
    })
    # G3 é a nota final (0-20; aprovação a partir de 10). G2 fica de fora: ela quase
    # determina G3 e transformaria o desafio em ler a resposta.
    t["risco_reprovacao"] = pd.cut(d.G3, [-1, 9, 11, 20], labels=[2, 1, 0]).astype(int)
    sgk = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEMENTE)
    i_pub, i_tst = next(sgk.split(t, t.risco_reprovacao, grupo))
    _gravar_tabela("tabular/boletim", t.iloc[i_pub], t.iloc[i_tst],
                   "id_estudante", "risco_reprovacao", "boletim.csv")


def batimento():
    """UCI Cardiotocography: estado fetal (normal / suspeito / patológico)."""
    d = pd.read_excel(BRUTOS / "ctg/CTG.xls", sheet_name="Raw Data").dropna(subset=["NSP"])
    dur_min = (d.e - d.b) / 60  # b e e marcam início e fim do trecho, em segundos
    por10 = lambda c: (d[c] / dur_min * 10).round(2)
    t = pd.DataFrame({
        "linha_de_base_bpm": d.LB,
        "aceleracoes_10min": por10("AC"),
        "movimentos_fetais_10min": por10("FM"),
        "contracoes_10min": por10("UC"),
        "desaceleracoes_leves_10min": por10("DL"),
        "desaceleracoes_severas_10min": por10("DS"),
        "desaceleracoes_prolongadas_10min": por10("DP"),
        "pct_variab_curta_anormal": d.ASTV,
        "media_variab_curta": d.MSTV,
        "pct_variab_longa_anormal": d.ALTV,
        "media_variab_longa": d.MLTV,
        "hist_largura": d.Width,
        "hist_minimo": d.Min,
        "hist_maximo": d.Max,
        "hist_n_picos": d.Nmax,
        "hist_n_zeros": d.Nzeros,
        "hist_moda": d.Mode,
        "hist_media": d.Mean,
        "hist_mediana": d.Median,
        "hist_variancia": d.Variance,
        "hist_tendencia": d.Tendency,
    })
    # As colunas A..SUSP e CLASS do arquivo são o laudo morfológico do especialista:
    # elas praticamente codificam a resposta, então não entram.
    t["estado_fetal"] = (d.NSP.astype(int) - 1).values
    t = t.drop_duplicates().reset_index(drop=True)
    pub, tst = train_test_split(t, test_size=0.2, stratify=t.estado_fetal, random_state=SEMENTE)
    _gravar_tabela("tabular/batimento", pub, tst, "id_exame", "estado_fetal", "batimento.csv")


# ----------------------------------------------------------------------------- VISÃO
def _quadrado_48(caminho):
    im = Image.open(caminho).convert("RGB")
    w, h = im.size
    lado = min(w, h)
    esq, topo = (w - lado) // 2, (h - lado) // 2
    im = im.crop((esq, topo, esq + lado, topo + lado))
    return np.asarray(im.resize((48, 48), Image.LANCZOS), dtype=np.uint8)


def _gravar_fotos(pasta, fotos, rotulos, n_teste, nome_gabarito):
    pasta = RAIZ / pasta
    pasta.mkdir(parents=True, exist_ok=True)
    i_pub, i_tst = train_test_split(np.arange(len(fotos)), test_size=n_teste,
                                    stratify=rotulos, random_state=SEMENTE)
    np.savez_compressed(pasta / "treino_fotos.npz", fotos=fotos[i_pub], rotulos=rotulos[i_pub])
    np.savez_compressed(pasta / "teste_fotos.npz", fotos=fotos[i_tst])
    pd.DataFrame({"id_foto": np.arange(len(i_tst)), "classe": rotulos[i_tst]}).to_csv(
        GABARITOS / nome_gabarito, index=False)
    u, c = np.unique(rotulos[i_pub], return_counts=True)
    print(f"{pasta.relative_to(RAIZ)}: público {len(i_pub)} | teste {len(i_tst)} | {dict(zip(u, c))}")


def sentinela():
    """EuroSAT RGB (Sentinel-2): uso do solo. Quantidades desiguais de propósito."""
    rng = np.random.default_rng(SEMENTE)
    escolha = {"Forest": ("floresta", 620), "Pasture": ("pastagem", 420),
               "AnnualCrop": ("lavoura", 560), "HerbaceousVegetation": ("campo", 380),
               "River": ("rio", 470)}
    fotos, rotulos = [], []
    for pasta, (nome, n) in escolha.items():
        arquivos = sorted((BRUTOS / "eurosat/EuroSAT_RGB" / pasta).glob("*.jpg"))
        for f in rng.choice(arquivos, n, replace=False):
            fotos.append(_quadrado_48(f)); rotulos.append(nome)
    _gravar_fotos("visao/sentinela", np.stack(fotos), np.array(rotulos), 250, "sentinela.csv")


def florada():
    """TF Flowers (Flickr, CC-BY): cinco espécies, amostra com as proporções originais."""
    nomes = {"daisy": "margarida", "dandelion": "dente_de_leao", "roses": "rosa",
             "sunflowers": "girassol", "tulips": "tulipa"}
    rng = np.random.default_rng(SEMENTE)
    fotos, rotulos = [], []
    for pasta, nome in nomes.items():
        arquivos = sorted((BRUTOS / "flower_photos" / pasta).glob("*.jpg"))
        # 70% de cada espécie: mantém o desequilíbrio original e o arquivo abaixo de 20 MB.
        for f in rng.choice(arquivos, round(len(arquivos) * 0.7), replace=False):
            fotos.append(_quadrado_48(f)); rotulos.append(nome)
    _gravar_fotos("visao/florada", np.stack(fotos), np.array(rotulos), 300, "florada.csv")


# ------------------------------------------------------------------------------- NLP
def _deduplicar(df):
    """Mesmo texto com rótulos diferentes é descartado; repetido com o mesmo rótulo fica uma vez."""
    df = df.assign(mensagem=df.mensagem.str.strip()).loc[lambda x: x.mensagem.str.len() > 0]
    rot = df.groupby("mensagem").rotulo.agg(["nunique", "first"])
    ok = rot[rot["nunique"] == 1]
    return pd.DataFrame({"mensagem": ok.index, "rotulo": ok["first"].values})


def pos_venda():
    """Olist: avaliações com comentário. Nota 1-2 = insatisfeito (0), 4-5 = satisfeito (1)."""
    d = pd.read_csv(BRUTOS / "olist_reviews.csv").dropna(subset=["review_comment_message"])
    # Nota 3 fica fora do treino: é a zona cinzenta. Uma amostra sem rótulo vai para o
    # caderno, que pergunta o que o modelo faz com ela.
    nota3 = d[d.review_score == 3].review_comment_message.str.replace(r"\s+", " ", regex=True)
    nota3 = nota3.str.strip().drop_duplicates().sample(400, random_state=SEMENTE)
    (RAIZ / "nlp/pos_venda").mkdir(parents=True, exist_ok=True)
    nota3.rename("mensagem").to_frame().to_csv(RAIZ / "nlp/pos_venda/nota3_amostra.csv", index=False)
    d = d[d.review_score != 3]
    d = pd.DataFrame({"mensagem": d.review_comment_message.str.replace(r"\s+", " ", regex=True),
                      "rotulo": (d.review_score >= 4).astype(int)})
    d = _deduplicar(d).sample(frac=1, random_state=SEMENTE)
    pub, tst = train_test_split(d, test_size=900, stratify=d.rotulo, random_state=SEMENTE)
    _gravar_tabela("nlp/pos_venda", pub, tst, "id_mensagem", "rotulo", "pos_venda.csv")


def checagem():
    """Fake.br (textos de tamanho normalizado). 0 = falsa (crítica, ~30%), 1 = verdadeira."""
    base = BRUTOS / "Fake.br-Corpus-master/size_normalized_texts"
    ler = lambda p: p.read_text(encoding="utf-8", errors="replace").strip()
    falsas = [ler(p) for p in sorted((base / "fake").glob("*.txt"), key=lambda p: int(p.stem))]
    verdadeiras = [ler(p) for p in sorted((base / "true").glob("*.txt"), key=lambda p: int(p.stem))]
    rng = np.random.default_rng(SEMENTE)
    n_falsas = round(len(verdadeiras) * 0.30 / 0.70)
    falsas = list(rng.choice(falsas, n_falsas, replace=False))
    d = pd.DataFrame({"mensagem": falsas + verdadeiras,
                      "rotulo": [0] * len(falsas) + [1] * len(verdadeiras)})
    d = _deduplicar(d.assign(mensagem=d.mensagem.str.replace(r"\s+", " ", regex=True)))
    d = d.sample(frac=1, random_state=SEMENTE)
    pub, tst = train_test_split(d, test_size=700, stratify=d.rotulo, random_state=SEMENTE)
    _gravar_tabela("nlp/checagem", pub, tst, "id_mensagem", "rotulo", "checagem.csv")


if __name__ == "__main__":
    etapas = {"boletim": boletim, "batimento": batimento, "sentinela": sentinela,
              "florada": florada, "pos_venda": pos_venda, "checagem": checagem}
    for nome in (sys.argv[2:] or etapas):
        etapas[nome]()
