# Respostas das lacunas e configurações de referência

> Spoiler. Use só depois de tentar.

## Lacunas

| Simulado | Respostas, na ordem das lacunas |
|---|---|
| Boletim | 1 `y` · 2 `DecisionTreeClassifier` · 3 `elif` · 4 `RandomForestClassifier` · 5 `else` · 6 `LogisticRegression` · 7 `fit` · 8 `predict` |
| Batimento | 1 `y` · 2 `DecisionTreeClassifier` · 3 `elif` · 4 `RandomForestClassifier` · 5 `else` · 6 `LogisticRegression` · 7 `StandardScaler` · 8 `fit` · 9 `predict` |
| Sentinela | 1 `y` · 2 `LogisticRegression` · 3 `elif` · 4 `RandomForestClassifier` · 5 `else` · 6 `SVC` · 7 `StandardScaler` · 8 `fit` · 9 `predict` |
| Florada | 1 `12:36` · 2 `y` · 3 `LogisticRegression` · 4 `elif` · 5 `RandomForestClassifier` · 6 `else` · 7 `SVC` · 8 `StandardScaler` · 9 `fit` · 10 `predict` |
| Pós-Venda | 1 `y` · 2 `TfidfVectorizer` · 3 `LogisticRegression` · 4 `elif` · 5 `LinearSVC` · 6 `else` · 7 `MultinomialNB` · 8 `fit_transform` · 9 `transform` · 10 `fit` · 11 `predict` |
| Checagem | igual ao Pós-Venda |

## Configurações de referência (sem ZONA LIVRE)

F1-macro medido na validação do próprio caderno e no teste oculto (`corrigir.py`). A
coluna "boa" é um ponto de partida, não o teto: a ZONA LIVRE ainda rende.

| Simulado | Padrão do caderno | Validação / teste | Configuração boa | Validação / teste |
|---|---|---|---|---|
| Boletim | Árvore, profundidade 10 | 0,596 / 0,643 | Árvore, profundidade 3 | 0,753 / 0,727 |
| Batimento | Regressão Logística sem padronizar | 0,727 / 0,735 | Floresta, profundidade 12, 300 árvores | 0,856 / 0,906 |
| Sentinela | SVM sem padronizar, C=1 | 0,823 / 0,771 | SVM padronizado, C=10 | 0,891 / 0,860 |
| Florada | SVM sem padronizar, C=1, 14 descritores | 0,475 / 0,453 | SVM padronizado, C=3, "todas + matiz e centro" | 0,612 / 0,581 |
| Pós-Venda | Naive Bayes, 400 termos, unigramas | 0,876 / 0,878 | SVM Linear, 50k termos, (1,2), C=0,3 | 0,913 / 0,898 |
| Checagem | Naive Bayes, 400 termos, unigramas | 0,666 / 0,657 | SVM Linear, 50k termos, (1,2), C=0,3 | 0,872 / 0,871 |

Armadilhas deliberadas:

- **Boletim:** a configuração padrão perde para a "regra do boletim" (0,695 na validação).
  Profundidade 3 ganha dela por ~6 pontos, acima da oscilação de ±2 entre sementes.
- **Checagem:** Naive Bayes com vocabulário grande (50k, bigramas) **colapsa**: F1 0,46 e
  recall da falsa 5%. É o caso de uso do `class_prior`.
- **Sentinela/Florada:** ligar a padronização é o maior ganho isolado.
