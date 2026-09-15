# OIAA 2026 — Fase 1

Repositório de trabalho para os desafios da **Olimpíada de IA (OIAA) 2026**. Estamos na **Fase 1**.

## O que é a Fase 1

Três desafios de classificação supervisionada, mesma métrica oficial (**F1-macro**), mesmo pipeline de 4 blocos (`explorar → configurar → modelar → avaliar`):

| # | Pasta | Dado | Tarefa | Classes | Teste oculto |
|---|---|---|---|---|---|
| 1 | `fase1/tabular/` | Tabela — estudantes | Risco de evasão | 3 (baixo/médio/alto) | 900 linhas |
| 2 | `fase1/visao/` | Imagem — fotos de resíduos | Tipo de material | 4 (papel/plástico/vidro/metal) | 240 linhas |
| 3 | `fase1/nlp/` | Texto — avaliações de produto de um marketplace | Cliente satisfeito ou insatisfeito | 2 (0 = insatisfeito, **classe crítica e minoritária, ~29%**; 1 = satisfeito) | 900 linhas |

O desafio tabular destrava os outros dois: o que se aprende ali se repete quase sem mudança em visão e texto.

## Estrutura do repositório

- `material-curso/` — material de estudo (não editar; é referência, transcrito da plataforma oficial). Índice completo em [material-curso/README.md](material-curso/README.md).
- `fase1/` — área de trabalho dos desafios.
  - `Guia_Desafios_Fase1_OIAA2026.pdf` — enunciado oficial dos três desafios.
  - `tabular/` — desafio 1. Contém `Tabular_COLAB.ipynb` (caderno preenchido), `treino.csv`/`teste.csv` (cópia local dos dados oficiais) e `submissoes/` com `submissao_tabular.csv` e `relatorio_tabular.json`.
  - `visao/` — desafio 2 (ainda vazio — caderno/dados a adicionar).
  - `nlp/` — desafio 3. Contém `NLP_COLAB.ipynb` (caderno preenchido), `treino.csv`/`teste.csv` (cópia local dos dados oficiais; o caderno baixa da URL e só usa a cópia se a rede falhar) e `submissoes/` com os entregáveis `submissao_nlp.csv` e `relatorio_nlp.json`.

Convenção de saída: cada desafio grava seus entregáveis em `fase1/<desafio>/submissoes/` (sem repetir o nome do desafio na subpasta). O caderno já aponta para lá via `PASTA_SUBMISSAO`.

## Como consultar o material de estudo

Antes de sugerir uma abordagem, prefira checar o material já traduzido em `material-curso/` em vez de recorrer a conhecimento genérico de ML — o curso fixa convenções específicas (nomes de variáveis, faixas de hiperparâmetro, ordem dos blocos) que a avaliação espera ver seguidas.

| Pergunta | Arquivo |
|---|---|
| Qual modelo/hiperparâmetro usar em cada desafio | [05-referencia-parametros.md](material-curso/05-referencia-parametros.md) |
| Como estruturar um caderno (blocos, pipeline canônico) | [06-caderno-4-blocos.md](material-curso/06-caderno-4-blocos.md) |
| Conceitos de tabular (treino/validação/teste, árvore/floresta/logística) | [02-cap2-tabular.md](material-curso/02-cap2-tabular.md) |
| Conceitos de visão (descritores, padronização, SVM) | [03-cap3-visao.md](material-curso/03-cap3-visao.md) |
| Conceitos de texto (TF-IDF, n-gramas, vazamento) | [04-cap4-texto.md](material-curso/04-cap4-texto.md) |
| Cheat sheet rápido durante o desafio | [10-cola-rapida.md](material-curso/10-cola-rapida.md) |
| Termos desconhecidos | [07-glossario.md](material-curso/07-glossario.md) |

## Regras não negociáveis do desafio (valem para os três cadernos)

1. **Dividir treino/validação antes de ajustar qualquer coisa.** `fit`/`fit_transform` só no treino; validação só vê `transform`/`predict`. Vale para `StandardScaler`, `TfidfVectorizer`, imputação — qualquer coisa que estime parâmetros dos dados.
2. **Métrica oficial é F1-macro**, não acurácia — os três desafios são desbalanceados (a classe majoritária infla a acurácia e esconde o erro nas classes raras).
3. **`class_weight='balanced'`** costuma ser o ganho isolado mais relevante (tabular e visão). Naive Bayes não aceita `class_weight`; usar `class_prior`. Nos dois desafios feitos, o `balanced` completo puxou demais para a classe minoritária e um **peso intermediário** ficou melhor: NLP `{0: 1.4, 1: 1}` (mesmo F1, mais recall); tabular `{0: 1, 1: 2.2, 2: 1.1}` (mesmo F1, +1pp de acurácia, que é o desempate).
4. **`max_iter` alto em `LogisticRegression`** (2000–3000) — o padrão (100) frequentemente não converge, e o aviso passa despercebido.
5. **Limitar `max_depth`** em árvore/floresta — `None` cresce até sobreajustar.
6. **Um fator por vez**, registrando o efeito (caderno de bordo) — mudar vários hiperparâmetros juntos impede saber o que funcionou.
7. Comparar sempre **na validação**, nunca no treino. Um resultado bom demais é sinal de vazamento, não de sorte.
8. Vale a **melhor submissão**, não a última.
9. **Justificativas do relatório curtas e diretas** (`resposta_N` na célula "O relatório da sua equipe"): 3–5 frases por resposta, só o que decide — configuração, número na validação, principal achado, principal limite. Sem narrar o processo inteiro; o log um-fator-por-vez fica no "Caderno de bordo", não na justificativa.

## Skills do projeto

- `.claude/skills/jupyter-notebook/` (Claude Code) e `.agents/skills/jupyter-notebook/` (cópia idêntica para outros agentes) — skill para criar/editar `.ipynb` (templates + `scripts/new_notebook.py`). Ao alterar uma, replique na outra. O `SKILL.md` tem uma seção "OIAA-specific rules" que vale acima da orientação genérica: os cadernos-desafio (`*_COLAB.ipynb`) não são refatorados nem re-escafoldados, só preenchidos; o script roda com o `.venv` do projeto.

## Estado dos desafios

| Desafio | Status | Melhor validação | Configuração |
|---|---|---|---|
| Tabular | caderno preenchido, submissão gerada | F1-macro 0.706 / acurácia 0.754 (CV 5×6: 0.707 ± 0.021) | `RandomForestClassifier(n_estimators=300, max_depth=10, class_weight={0: 1, 1: 2.2, 2: 1.1})`, 35 variáveis originais |
| Visão | não iniciado | — | — |
| NLP | caderno preenchido, submissão gerada | F1-macro 0.948 / recall₀ 0.935 | `LinearSVC(C=0.3, class_weight={0: 1.4, 1: 1})` + TF-IDF (1,2), 50k termos, `min_df=3`, `sublinear_tf`, `strip_accents='unicode'` |

O log um-fator-por-vez de cada desafio está na célula "Caderno de bordo" do próprio notebook; não repita os experimentos, parta dele. No tabular, o teto dos três modelos permitidos é ~0.70–0.71 de F1 (a classe Médio fica em F1 ≈ 0.50 e nada moveu isso além do ruído); atributos derivados e remoção de colunas pioraram. Com só 809 linhas de validação, uma divisão isolada oscila ±0.02 — confirme qualquer comparação com validação cruzada repetida antes de decidir.

## Ambiente

- Os `*_COLAB.ipynb` são notebooks Colab — ao editar, preserve a estrutura de células e siga o pipeline canônico do material (dividir → ajustar só no treino → treinar → avaliar com F1-macro).
- **Execute os notebooks com o kernel do `.venv`** (`oiaa-venv`, registrado em `.venv/share/jupyter/kernels/`; comando completo no `SKILL.md` do jupyter-notebook). O kernelspec `python3` padrão da máquina aponta para o Python global, com outra versão do scikit-learn — a mesma floresta deu F1 0.693 lá e 0.706 no `.venv`. Pela mesma razão, o número que o Colab imprimir pode diferir em ~1pp do registrado aqui; a configuração é robusta na validação cruzada, o número exato não.
- **Para rodar um notebook localmente (execução de células, teste de código):** verifique se existe um `.venv` na raiz do repositório; se não existir, crie um antes de instalar qualquer pacote ou executar o notebook.

  ```bash
  python -m venv .venv
  .venv/Scripts/activate   # Windows
  pip install -r requirements.txt   # ou instalar pandas/scikit-learn/jupyter conforme o notebook exigir
  ```

  Nunca instale pacotes no Python global da máquina.
