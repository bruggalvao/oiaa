# Simulados da Fase 1 — OIAA 2026

Seis desafios de treino, **dois por trilha**, no mesmo formato dos cadernos oficiais da Fase 1:
história com um "cliente", os quatro blocos (`explorar → configurar → modelar → avaliar`),
células `SUA VEZ` com **LACUNAS**, célula de verificação, parada obrigatória de ética,
relatório em JSON e arquivo de submissão com tamanho fixo. A métrica oficial é sempre o
**F1-macro**.

Os dados são **reais e públicos**; os cenários são simulados. O rótulo do conjunto de teste
não está na pasta do desafio: ele fica em `_gabaritos/` e só o corretor o lê.

| Trilha | Simulado | Pasta | Dado | Classes | Público / teste | O que ele treina além do oficial |
|---|---|---|---|---|---|---|
| Tabular | 📒 **Operação Boletim** | `tabular/boletim/` | Estudantes de duas escolas públicas (Portugal) | risco de reprovação baixo / médio / alto | 783 / 261 | Base pequena: oscilação da validação entre sementes; comparar com uma regra simples; registros repetidos do mesmo estudante |
| Tabular | 💓 **Operação Batimento** | `tabular/batimento/` | 2.114 cardiotocografias fetais | normal / suspeito / patológico (78 / 14 / 8%) | 1.691 / 423 | Desequilíbrio forte; `StandardScaler` dentro do `Pipeline`; fadiga de alarme |
| Visão | 🛰️ **Missão Sentinela** | `visao/sentinela/` | Recortes do satélite Sentinel-2 (EuroSAT) | floresta / pastagem / lavoura / campo / rio | 2.200 / 250 | Deslocamento de domínio (Europa → Cerrado); custo diferente por par de classes |
| Visão | 🌼 **Missão Florada** | `visao/florada/` | Fotos amadoras de flores (TF Flowers) | margarida / dente-de-leão / rosa / girassol / tulipa | 2.269 / 300 | Escrever um extrator novo (histograma de matiz + recorte central); viés de coleta |
| NLP | 📦 **Operação Pós-Venda** | `nlp/pos_venda/` | Comentários reais de e-commerce brasileiro (Olist) | insatisfeito (0, ~32%) / satisfeito (1) | 31.471 / 900 | Textos muito curtos; a "zona cinzenta" da nota 3; culpa da entrega × do lojista |
| NLP | 🔎 **Operação Checagem** | `nlp/checagem/` | Notícias falsas e verdadeiras (Fake.br) | falsa (0, ~30%) / verdadeira (1) | 4.442 / 700 | Colapso do Naive Bayes sob desequilíbrio (`class_prior`); estilo × verdade; atalho do tamanho do texto |

Cada pasta tem o caderno (`*_COLAB.ipynb`), os dados (`treino.csv`/`teste.csv` ou
`treino_fotos.npz`/`teste_fotos.npz`) e, depois de rodar, a subpasta `submissoes/`.

## Como usar

1. Abra o caderno da pasta do simulado (localmente com o `.venv`, ou no Colab enviando os
   arquivos de dados quando a primeira célula pedir).
2. Siga os blocos de cima para baixo, preencha as lacunas e rode a verificação até aparecer
   `VERIFICAÇÃO APROVADA`.
3. A última célula grava `submissoes/submissao_<desafio>.csv` e `relatorio_<desafio>.json`.
4. Corrija contra o gabarito oculto, da raiz do repositório:

   ```bash
   .venv/Scripts/python simulados/corrigir.py boletim simulados/tabular/boletim/submissoes/submissao_boletim.csv
   ```

   O corretor valida o arquivo como a plataforma faria (colunas, número de linhas, ids,
   classes válidas) e imprime F1-macro, acurácia, a revocação da classe crítica, o F1 por
   classe e a matriz de confusão. Com `--registrar "nome da equipe"`, anexa o resultado a
   `_gabaritos/placar.csv`.

Para treinar de verdade, **não abra `_gabaritos/`** antes de terminar. Lá também está
`respostas_lacunas.md`, com as respostas das lacunas e configurações de referência.

## Formato das submissões

| Simulado | Colunas do CSV | Valores |
|---|---|---|
| boletim | `id_estudante,risco_previsto` | 0 baixo · 1 médio · 2 alto |
| batimento | `id_exame,estado_previsto` | 0 normal · 1 suspeito · 2 patológico |
| sentinela | `id_foto,classe_prevista` | `floresta`, `pastagem`, `lavoura`, `campo`, `rio` |
| florada | `id_foto,classe_prevista` | `margarida`, `dente_de_leao`, `rosa`, `girassol`, `tulipa` |
| pos_venda | `id_mensagem,rotulo_previsto` | 0 insatisfeito · 1 satisfeito |
| checagem | `id_mensagem,rotulo_previsto` | 0 falsa · 1 verdadeira |

## Fontes, licenças e como remontar as bases

As bases foram montadas por `_preparo/preparar_bases.py` (determinístico, semente 2026), a
partir dos arquivos originais:

| Simulado | Fonte | Licença / citação |
|---|---|---|
| Boletim | [UCI 320 — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) | CC BY 4.0 · Cortez, P.; Silva, A. (2008). *Using data mining to predict secondary school student performance.* |
| Batimento | [UCI 193 — Cardiotocography](https://archive.ics.uci.edu/dataset/193/cardiotocography) | CC BY 4.0 · Ayres-de-Campos, D. et al. (2000). *SisPorto 2.0.* |
| Sentinela | [EuroSAT RGB (Zenodo 7711810)](https://zenodo.org/records/7711810) | MIT · Helber, P. et al. (2019). *EuroSAT.* Contém dados Copernicus Sentinel-2. |
| Florada | [TensorFlow flower_photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) | Fotos do Flickr sob CC BY 2.0 (créditos no `LICENSE.txt` do arquivo original) |
| Pós-Venda | [olist/work-at-olist-data](https://github.com/olist/work-at-olist-data) | CC BY-NC-SA 4.0 (uso não comercial) · Olist, *Brazilian E-Commerce Public Dataset* |
| Checagem | [roneysco/Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus) | Monteiro, R. A. et al. (2018). *Contributions to the Study of Fake News in Portuguese.* PROPOR 2018. |

Decisões de montagem que importam para quem resolve:

- **Boletim:** a nota final `G3` virou o risco (≥12 baixo, 10–11 médio, <10 alto). `G2` foi
  removida porque quase determina `G3`. O teste foi separado **por estudante** (quem faz
  Português e Matemática não aparece dos dois lados).
- **Batimento:** contagens de eventos convertidas para "por 10 minutos"; as colunas do laudo
  morfológico (`A`…`SUSP`, `CLASS`) foram removidas porque praticamente codificam a resposta.
- **Sentinela:** 5 das 10 classes do EuroSAT, com quantidades desiguais de propósito;
  imagens reduzidas de 64×64 para 48×48.
- **Florada:** 70% de cada espécie (mantém o desequilíbrio original), recorte quadrado
  central, 48×48.
- **Pós-Venda:** nota 1–2 = insatisfeito, 4–5 = satisfeito; nota 3 fora do treino (uma
  amostra sem rótulo fica em `nota3_amostra.csv`); textos repetidos deduplicados.
- **Checagem:** versão de tamanho normalizado do corpus; falsas subamostradas para 30%.

Para remontar do zero: baixe as seis fontes numa pasta (a lista de subpastas esperadas está
no topo de `_preparo/preparar_bases.py`) e rode
`.venv/Scripts/python simulados/_preparo/preparar_bases.py <pasta>`. O script precisa de
`xlrd` (para o `.xls` da cardiotocografia) e `pillow`.
