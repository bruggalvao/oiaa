# Referência de modelos e hiperparâmetros

Consolidação das 15 tabelas de referência do material. **Todo parâmetro não declarado assume o padrão em silêncio.**

## Qual modelo em qual desafio

| Modelo | Tabular (1) | Visão (2) | Texto (3) | Precisa padronizar? |
|---|---|---|---|---|
| `DecisionTreeClassifier` | ✅ | — | ❌ (se perde com milhares de colunas) | Não |
| `RandomForestClassifier` | ✅ padrão-ouro | ✅ (perde para o SVM) | ❌ | Não |
| `LogisticRegression` | ✅ (sofre com interações) | ✅ | ✅ costuma ser a melhor | **Sim** |
| `SVC` (RBF) | — | ✅ costuma vencer | ❌ (lento, alta dimensão) | **Sim, obrigatório** |
| `LinearSVC` | — | — | ✅ | Sim |
| `MultinomialNB` | — | — | ✅ linha de base rápida | — |
| `TfidfVectorizer` | — | — | ✅ obrigatório | — |
| `StandardScaler` | opcional | ✅ | — | — |

---

## `DecisionTreeClassifier` — a árvore

```python
from sklearn.tree import DecisionTreeClassifier
modelo = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)
```

**Fortes:** única que se desenha num papel e se explica para qualquer pessoa. Não precisa de padronização. Lida com numéricas e categóricas sem preparo. Captura interações naturalmente.
**Frágil:** muito instável (mudar poucos exemplos muda a árvore inteira). Vicia em decorar se crescer sem limite. Fronteira sempre em degraus retos — ruim para fronteiras diagonais ou curvas.
**Use quando** a explicação importar mais que o último ponto de desempenho. **Evite quando** quiser o melhor desempenho possível — a floresta faz o mesmo melhor.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `max_depth` | `None` | Até onde a árvore cresce. `None` cresce até as folhas ficarem puras = quase sempre sobreajuste |
| `min_samples_leaf` | `1` | Mínimo de exemplos por folha. Aumentar impede cercar casos isolados do treino |
| `class_weight` | `None` | `'balanced'` compensa desbalanceamento durante o treino |
| `min_samples_split` | `2` | Mínimo de exemplos para um nó poder ser dividido |
| `criterion` | `'gini'` | Medida de impureza para escolher os cortes. Alternativa: `'entropy'` |
| `max_features` | `None` | Quantas variáveis considerar em cada corte. Aceita número, fração ou `'sqrt'` |
| `max_leaf_nodes` | `None` | Teto de folhas — outra forma de limitar complexidade |
| `ccp_alpha` | `0.0` | Poda por complexidade de custo. Valores entre 0.001 e 0.02 podam depois de crescida |
| `random_state` | `None` | Semente. Fixar garante reprodutibilidade |

---

## `RandomForestClassifier` — a floresta

```python
from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1)
```

**Fortes:** melhor escolha inicial em dados tabulares. Muito mais estável que uma árvore. Resiste a sobreajuste (cada árvore vê uma amostra diferente). Fornece importância de variáveis de graça. **Não precisa de padronização.**
**Frágil:** perde a explicação individual. Mais lenta para treinar e prever. Não faz fronteiras diagonais suaves.
**Use quando** o dado for tabular e o objetivo for desempenho com pouco ajuste. **Evite quando** houver restrição forte de tempo ou a explicação caso a caso for obrigatória.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `n_estimators` | `100` | Quantas árvores votam. Acima de ~300 o ganho costuma ser pequeno |
| `class_weight` | `None` | `'balanced'` ou `'balanced_subsample'`. **É o ganho isolado mais relevante do desafio tabular** |
| `max_depth` | `None` | Profundidade máxima de cada árvore |
| `min_samples_leaf` | `1` | Mínimo por folha. Aumentar suaviza |
| `max_features` | `'sqrt'` | Variáveis sorteadas por corte. Aceita fração (`0.5`) |
| `bootstrap` | `True` | Se cada árvore vê reamostragem com reposição |
| `oob_score` | `False` | Estima o erro com as amostras deixadas de fora, sem gastar dados de validação |
| `criterion` | `'gini'` | Medida usada para escolher os cortes. Alternativa: `'entropy'` |
| `n_jobs` | `None` | `-1` usa todos os núcleos e acelera bastante |
| `random_state` | `None` | Semente |

---

## `LogisticRegression` — a soma de pesos

```python
from sklearn.linear_model import LogisticRegression
modelo = LogisticRegression(C=1.0, max_iter=2000)          # tabular
modelo = LogisticRegression(C=4, max_iter=3000, class_weight="balanced")  # texto
```

**Fortes:** rápida, estável, transparente (dá para olhar o peso de cada variável). Excelente com muitas variáveis e poucos exemplos — o caso típico de texto.
**Frágil:** fronteira sempre reta. **Não captura interações.** Exige padronização quando as escalas diferem. `max_iter=100` frequentemente não converge.
**Use quando** o problema for aproximadamente linear ou houver muitas variáveis (TF-IDF). **Evite quando** houver interação clara entre variáveis.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `C` | `1.0` | **Inverso** da regularização — atenção à inversão: `C < 1` regulariza mais e simplifica. Em texto, 2 a 8 costuma render mais |
| `max_iter` | `100` | Iterações para convergir. **Use 2000 ou 3000** em visão e texto |
| `class_weight` | `None` | `'balanced'` compensa desbalanceamento |
| `penalty` | `'l2'` | Tipo de regularização: `'l1'`, `'l2'`, `'elasticnet'` ou `None`. `'l1'` zera o peso de palavras irrelevantes |
| `solver` | `'lbfgs'` | Algoritmo de otimização. Para `'l1'` é preciso `'liblinear'` ou `'saga'` |
| `fit_intercept` | `True` | Se estima o termo constante |
| `random_state` | `None` | Semente |

---

## `SVC` — SVM com núcleo RBF (visão)

```python
from sklearn.svm import SVC
modelo = SVC(C=4, kernel="rbf", gamma="scale")
```

**Fortes:** fronteiras curvas que contornam grupos de formato complicado. Muito eficaz com classes bem separadas e número moderado de exemplos.
**Frágil:** **exige padronização** (mede distância). Lento com muitos exemplos. `C` e `gamma` interagem — ajustar um sem o outro leva a sobreajuste. Não explica decisões.
**Use quando** os dados forem numéricos, padronizados, de dimensão moderada e com fronteira curva (o cenário do desafio de visão). **Evite quando** houver dezenas de milhares de exemplos ou não puder padronizar.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `C` | `1.0` | Quanto tolera errar no treino. Baixo = fronteira mais lisa; alto = fronteira retorcida (sobreajuste). **Faixa útil aqui: 1 a 12** |
| `gamma` | `'scale'` | Alcance da influência de cada exemplo. Alto = bolhas em volta dos exemplos; baixo = fronteira alisada. Aceita `'scale'`, `'auto'` ou número |
| `kernel` | `'rbf'` | Forma da fronteira: `'linear'` (reta), `'rbf'` (curvas), `'poly'` (polinômios) |
| `degree` | `3` | Grau do polinômio. Só com `kernel='poly'` |
| `class_weight` | `None` | `'balanced'` compensa desbalanceamento |
| `probability` | `False` | Habilita `predict_proba`, mas deixa o treino bem mais lento |
| `random_state` | `None` | Semente |

---

## `LinearSVC` — SVM linear (texto)

```python
from sklearn.svm import LinearSVC
modelo = LinearSVC(C=1, class_weight="balanced")
```

Mesma ideia de maior margem, **só com fronteira reta**, otimizado para muitos atributos. Com milhares de colunas os dados quase sempre já são separáveis por um plano — a curva do RBF vira custo desnecessário.
⚠️ **Não tem `predict_proba`.** Devolve a classe, não a probabilidade. Para pontuação de confiança, use a regressão logística.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `C` | `1.0` | Penalidade por erro no treino. **Lógica inversa à da logística: `C` maior ajusta mais ao treino** |
| `penalty` | `'l2'` | Tipo de regularização |
| `loss` | `'squared_hinge'` | Função de perda. Aceita `'hinge'` (formulação clássica do SVM) |
| `class_weight` | `None` | `'balanced'` compensa desbalanceamento |
| `max_iter` | `1000` | Iterações máximas. Se avisar não convergência, aumente |
| `dual` | automático | Formulação do problema. O padrão resolve bem quando há mais colunas que exemplos |
| `random_state` | `None` | Semente |

---

## `MultinomialNB` — Naive Bayes (texto)

```python
from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB(alpha=0.3)
```

**Fortes:** absurdamente rápido mesmo com vocabulário grande. Funciona bem com poucos exemplos. Clássico consagrado e excelente linha de base.
**Frágil:** assume variáveis independentes (falso em texto). **Não aceita `class_weight`** — dá erro; o caminho equivalente é `class_prior`. Costuma perder para a logística quando há dados suficientes.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `alpha` | `1.0` | **Suavização de Laplace.** Resolve o problema da palavra nunca vista numa classe, cuja probabilidade zero zeraria a conta toda. Valores 0.1–0.5 confiam mais nos dados e rendem mais em corpus grande |
| `fit_prior` | `True` | Se aprende a frequência das classes a partir dos dados. `False` assume classes igualmente prováveis |
| `class_prior` | `None` | Fixa manualmente a probabilidade de cada classe. **Substituto do `class_weight`** |

---

## `TfidfVectorizer` — o tradutor (texto)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(max_features=4000, ngram_range=(1,2), min_df=2)
```

> Os parâmetros dele mudam o resultado **mais que a escolha do classificador**.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `max_features` | `None` | Tamanho do vocabulário (mantém os termos mais frequentes). Pequeno descarta sinal, grande aumenta o sobreajuste. **É a decisão que mais muda o resultado neste desafio** |
| `ngram_range` | `(1, 1)` | Faixa de n-gramas. `(1,2)` inclui palavras isoladas e pares — transforma "conta bloqueada" em unidade própria |
| `min_df` | `1` | Frequência documental mínima. Descarta termos raríssimos que o modelo associaria a uma classe por acidente |
| `max_df` | `1.0` | Descarta termos presentes em mais que essa fração dos documentos |
| `sublinear_tf` | `False` | `True` aplica escala logarítmica — a décima ocorrência não vale dez vezes a primeira. Costuma ajudar |
| `strip_accents` | `None` | `'unicode'` remove acentos e unifica "voce" com "você". **Em português informal isso rende pontos** |
| `lowercase` | `True` | Converte para minúsculas antes de separar as palavras |
| `stop_words` | `None` | Lista de palavras a descartar. Em geral desnecessário — o IDF já derruba as comuns |
| `binary` | `False` | `True` considera só presença, ignorando quantas vezes o termo aparece |
| `norm` | `'l2'` | Normaliza cada linha, para mensagens longas não pesarem mais que curtas |
| `token_pattern` | regex padrão | Define o que conta como palavra. O padrão descarta palavras de uma letra |

---

## `StandardScaler` — o transformador de apoio

```python
from sklearn.preprocessing import StandardScaler
escalador = StandardScaler()
X_padronizado = escalador.fit_transform(treino_X)
```

Não é um modelo: converte cada variável para média zero e desvio um, colocando todas na mesma régua.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `with_mean` | `True` | Subtrai a média, centralizando em zero |
| `with_std` | `True` | Divide pelo desvio-padrão, deixando a dispersão igual a um |
| `copy` | `True` | Trabalha sobre uma cópia em vez de alterar os dados originais |

**Precisam:** SVM e Regressão Logística (usam distância e gradiente).
**Não precisa:** Floresta Aleatória e árvores (comparam limiares).
Testar com e sem é a forma de confirmar na prática.
