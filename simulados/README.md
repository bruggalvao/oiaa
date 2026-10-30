# Simulados da Fase 2 — OIAA 2026

Seis desafios de treino, **dois por trilha**, no formato dos cadernos da Fase 2 (material dos
aulões em [`material-curso/fase2/`](../material-curso/fase2/)). Em cada trilha:

- **um simulado com a base da Fase 1**, agora resolvido com os modelos novos da trilha;
- **um simulado com uma base nova**, nunca usada nos desafios oficiais.

| Trilha | Modelos da Fase 2 | Base da Fase 1 | Base nova |
|---|---|---|---|
| Tabular | rede densa · rede com embeddings · TabNet | 🔦 **Farol** (`tabular/farol/`) — evasão, 3 classes | 📒 **Boletim** (`tabular/boletim/`) — risco de reprovação, categorias em texto, base pequena |
| Visão | EfficientNet-B0 · ResNet-50 · ViT-B/16 | ♻️ **Recomeço** (`visao/recomeco/`) — as mesmas fotos do TrashNet, agora em resolução cheia | 🌼 **Florada** (`visao/florada/`) — 5 espécies de flor, fotos amadoras com fundo |
| NLP | DistilBERT multilíngue · BERTimbau · XLM-RoBERTa | 👂 **Voz do Cliente** (`nlp/voz_do_cliente/`) — as 110 mil avaliações oficiais | 🔎 **Checagem** (`nlp/checagem/`) — notícias falsas × verdadeiras, textos longos |

Todos seguem a forma dos aulões: **Bloco 1** (piso e o que se mede antes de treinar), **Bloco
2** (as fichas dos três modelos e os botões), **Bloco 3** (células **A** preparo, **B** treino
com Zona Livre, **C** avaliação), verificação das lacunas e **Bloco 4** (curvas, matriz, grupos,
custo × ganho), parada ética, relatório com o campo de **IA generativa**, e submissão.

## O que cada simulado treina

| Simulado | Lacunas | As armadilhas que a verificação pega |
|---|---|---|
| Farol | 13 | piso; cardinalidades; régua ajustada fora do treino (vazamento); `cat_idxs`/`cat_dims` trocados no TabNet; lote virtual que não divide o lote; sem ponto de verificação |
| Boletim | 13 | as mesmas, com categorias em texto e uma base de 783 linhas (o TabNet sofre) |
| Recomeço | 12 | preparo da validação escrito à mão; **aumento vazando para a validação** (a mesma foto passa duas vezes); camada de saída com o nome de outro modelo; modelo fora do `eval()` |
| Florada | 12 | as mesmas; o mapa de atenção mostra se o modelo olha a flor ou o céu |
| Voz do Cliente | 10 | **tradutor e modelo de nomes diferentes**; texto passado para minúsculas; limite de corte; aquecimento; `eval()` |
| Checagem | 10 | as mesmas; o grupo "cortados pelo limite" mostra o custo do corte em textos longos |

## Como usar

1. **No Colab (recomendado para visão e NLP):** abram o caderno, liguem a **T4 GPU** e enviem
   os arquivos da pasta quando a primeira célula pedir (`treino.csv` + `teste.csv`, ou
   `fotos.zip`). A trilha tabular roda bem em CPU.
2. **Localmente:** abram o caderno de dentro da pasta do simulado, com o `.venv` (precisa de
   `torch`, `torchvision`, `transformers` e, para o TabNet, `pytorch-tabnet`).
3. Preencham as lacunas e rodem a verificação até aparecer `VERIFICAÇÃO APROVADA`.
4. A última célula grava `submissoes/submissao_<desafio>.csv` e `relatorio_<desafio>.json`.
5. Corrijam contra o gabarito oculto, da raiz do repositório:

   ```bash
   .venv/Scripts/python simulados/corrigir.py farol simulados/tabular/farol/submissoes/submissao_farol.csv
   ```

   O corretor valida o arquivo como a plataforma (colunas, linhas, ids, classes) e imprime
   F1-macro, acurácia, a revocação da classe crítica, o F1 por classe e a matriz de confusão.
   Com `--registrar "equipe"`, anexa o resultado a `_gabaritos/placar.csv`.

Para treinar de verdade, **não abram `_gabaritos/`** antes de terminar: lá estão os rótulos
do teste e `respostas_lacunas.md`.

## Formato das submissões

| Simulado | Colunas | Valores |
|---|---|---|
| farol | `id_estudante,risco_previsto` | 0 baixo · 1 médio · 2 alto |
| boletim | `id_estudante,risco_previsto` | 0 baixo · 1 médio · 2 alto |
| recomeco | `id_foto,classe_prevista` | `metal`, `papel`, `papelao`, `plastico`, `vidro` |
| florada | `id_foto,classe_prevista` | `dente_de_leao`, `girassol`, `margarida`, `rosa`, `tulipa` |
| voz_do_cliente | `id_mensagem,rotulo_previsto` | 0 insatisfeito · 1 satisfeito |
| checagem | `id_mensagem,rotulo_previsto` | 0 falsa · 1 verdadeira |

## Fontes, licenças e como remontar as bases

As bases são montadas por [`_preparo/preparar_bases.py`](_preparo/preparar_bases.py)
(determinístico, semente 2026). Nas bases da Fase 1 o teste oficial não tem gabarito público:
o teste oculto daqui é recortado das linhas rotuladas.

| Simulado | Fonte | Licença / citação |
|---|---|---|
| Farol | dados oficiais da Operação Farol (`fase1/tabular/MLP/treino.csv`) | OIAA 2026 |
| Boletim | [UCI 320 — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) | CC BY 4.0 · Cortez & Silva (2008) |
| Recomeço | [TrashNet](https://github.com/garythung/trashnet) sem a classe *trash* (as 2.390 fotos da Missão Recomeço) | Thung & Yang (2016) |
| Florada | [TensorFlow flower_photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) | Fotos do Flickr sob CC BY 2.0 |
| Voz do Cliente | dados oficiais da Operação Voz do Cliente (`fase1/nlp/treino.csv`) | OIAA 2026 |
| Checagem | [Fake.br Corpus](https://github.com/roneysco/Fake.br-Corpus) | Monteiro et al. (2018), PROPOR |

Decisões de montagem:

- **Farol:** as 17 colunas `curso_*` voltaram a ser uma coluna `curso` (categoria com 17 valores).
- **Boletim:** categorias por extenso (escola, disciplina, ocupação, motivo, responsável); `G2`
  removida; teste separado por estudante.
- **Recomeço / Florada:** `fotos.zip` com `treino/<classe>/NNNN.jpg` e `teste/NNNN.jpg`, lado
  menor em 256 px (o preparo dos pesos corta 224). Florada usa metade de cada espécie.
- **Checagem:** versão de tamanho normalizado; falsas subamostradas para 30%.
- Em todas as tabelas, toda categoria do teste aparece no público (senão embeddings e TabNet
  não conseguem prever).

Para remontar: `.venv/Scripts/python simulados/_preparo/preparar_bases.py <pasta_com_os_brutos>`
(a lista de arquivos esperados está no topo do script).
