# Cola rápida — para usar durante o desafio

## Números para não esquecer

| | Tabular | Visão | Texto |
|---|---|---|---|
| Classes | 3 (baixo/médio/alto) | 4 (papel/plástico/vidro/metal) | 2 (golpe/legítima) |
| Público | 3.600 linhas × 12 colunas | — | — |
| Distribuição | baixo 2.592 (72%) · médio 684 · alto 324 (9%) | — | — |
| Submissão (teste oculto) | **900 linhas** | **240 linhas** | **900 linhas** |
| Métrica oficial | F1-macro | F1-macro | F1-macro |
| Baseline do caderno | acc ~80% / F1-macro ~64% | — | — |
| Modelo preguiçoso | acc 72% / F1-macro ~28% | — | — |

Imagem: 48 × 48 × 3 = **6.912 números** → **14 descritores**.

## Os 5 erros que custam mais

1. **Vazamento** — ajustar `StandardScaler`/`TfidfVectorizer` **antes** de dividir, ou usar `fit_transform` na validação. Não dá erro e a métrica melhora.
2. **`class_weight=None`** — o padrão dos três modelos. Em problema desbalanceado, ninguém compensa nada. `'balanced'` é o ganho isolado mais relevante do tabular.
3. **`max_iter=100`** na `LogisticRegression` — frequentemente não converge, e o aviso é fácil de ignorar. Use 2000–3000.
4. **`max_depth=None`** — deixa a árvore crescer sem limite = sobreajuste quase garantido.
5. **Rodar células fora de ordem** no Jupyter. Quando estranhar: `Kernel → Restart & Run All`.

## Método de trabalho

- **Um fator por vez**, e anote o efeito no caderno de bordo. Mudar tudo junto impede saber o que funcionou.
- **Vale a melhor submissão, não a última.** Experimentar não custa nada.
- **Existe pontuação parcial** — acertar o passo conta.
- Comparem sempre **na validação**, nunca no treino.
- Onde investir o tempo: **tabular** → `class_weight`, `max_depth`, `n_estimators`; **visão** → padronização + par (`C`, `gamma`); **texto** → **o vetorizador antes do classificador** (`max_features`, `ngram_range`).

## Diagnóstico rápido de sintomas

| Sintoma | Diagnóstico provável | Ação |
|---|---|---|
| Ótimo no treino, ruim na validação | Sobreajuste | Baixar `max_depth`, subir `min_samples_leaf`, baixar `C` |
| Ruim nos dois conjuntos | Subajuste | Modelo mais complexo, mais variáveis, mais iterações |
| Acurácia alta, F1-macro baixo | Desbalanceamento | `class_weight="balanced"`, olhar a matriz de confusão |
| Resultado suspeito de bom | Vazamento | Conferir se algo foi ajustado antes da divisão |
| `NotFittedError` | `predict` antes do `fit` | Inverter as linhas |
| Aviso de não convergência | `max_iter` baixo | Subir para 2000–3000 |
| Vai bem aqui, mal em outro ambiente | Deslocamento de domínio | Reportar; verificar condições de captura |
| Caía com o tempo, sem mexer no código | Deriva conceitual | Plano de monitoramento + retreino periódico |

## Cheat sheet de código

```python
# ---------- 1. imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# ---------- 2. explorar
dados.shape
dados["alvo"].value_counts()

# ---------- 3. dividir (SEMPRE antes de qualquer fit)
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- 4. modelar
modelo = RandomForestClassifier(n_estimators=300, max_depth=12,
                                class_weight="balanced", n_jobs=-1, random_state=42)
modelo = make_pipeline(StandardScaler(), SVC(C=4, kernel="rbf", gamma="scale"))
modelo.fit(X_tr, y_tr)

# ---------- 5. texto: nunca fit_transform na validação
vet  = TfidfVectorizer(max_features=4000, ngram_range=(1,2), min_df=2,
                       strip_accents="unicode", sublinear_tf=True)
X_tr = vet.fit_transform(txt_tr)
X_va = vet.transform(txt_va)

# ---------- 6. avaliar
pred = modelo.predict(X_va)
f1_score(y_va, pred, average="macro")    # metrica oficial
confusion_matrix(y_va, pred)             # onde o erro mora
classification_report(y_va, pred)         # precisao/revocacao por classe
```

## Regras que valem para tudo

> **Qualquer operação que estime parâmetros a partir dos dados deve ser ajustada só no treino e apenas aplicada aos demais.**
> Vale para vocabulário, IDF, média, desvio-padrão e valor de imputação.

> **Modelo bom não existe em abstrato: existe modelo bom para um problema.**

> **Todo parâmetro não declarado assume, em silêncio, um valor padrão.**
