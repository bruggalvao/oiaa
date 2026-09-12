# Gabarito das atividades

Todas as atividades interativas do material, com resposta e o raciocínio por trás dela.
As atividades valem **medalha** e o progresso fica no `localStorage` do navegador.

## Inventário

| Cap. | Atividades |
|---|---|
| 1 | Bússola ética (4 questões) · Jogo da memória (6 pares) · Laboratório de gráficos · Ligação certa (6 pares) · Prova final (8 questões) · Explicação escrita |
| 2 | Ponto ótimo · Parâmetro e efeito (6 pares) · Escolha do modelo (5 cenários) · Diagnóstico · Duas versões · Monte o experimento · Caça ao erro · Matriz viva · Galeria de fronteiras · Código (menu + digitado) · Prova final (7 questões) |
| 3 | Pixel a número · Que descritor separa (3 questões) · Oficina de filtros (3 questões) · Resolva a confusão (2 questões) · Preveja a saída · Parâmetro e efeito (6 pares) · Código (menu + digitado) · Prova final (4 questões) |
| 4 | Calculadora TF-IDF · Oficina de texto (3 questões) · Caça ao vazamento · A investigação (3 questões) · Parâmetro e efeito (6 pares) · Código (menu + digitado) · Monte o pipeline · Prova final (6 questões) |

---

## Capítulo 1

### Bússola ética

| Pergunta | Resposta | Por quê |
|---|---|---|
| A lista de estudantes em risco vai ser usada pela escola. Que regra de uso entregar junto? | Uma regra dizendo **quem pode acessar, por quanto tempo vale, o que não pode ser feito e quem responde pela decisão** | Sistemas que apontam pessoas precisam de protocolo institucional, não só de código. Divulgar para toda a escola seria o oposto: expõe justamente quem está vulnerável |
| Qual erro é mais grave num sistema de apoio pedagógico? | Marcar como **risco baixo** um estudante que estava prestes a desistir | Errar para mais gasta tempo da equipe; errar para menos deixa sem apoio justamente quem precisava. Essa conclusão não vem da matemática, vem de um juízo sobre custo — e o modelo não faz esse juízo |
| "Trabalha" é indicador indireto de condição socioeconômica. O que fazer? | **Reconhecer a tensão** (aumenta a detecção mas concentra a sinalização sobre grupo vulnerável) e discutir isso abertamente na entrega | Remover não resolve: outras variáveis correlacionadas reconstroem o sinal. Ignorar também não serve. Problemas assim não têm solução puramente técnica |
| Pedir a uma IA que explique uma mensagem de erro é permitido? | **Sim** — entender um erro é aprendizado | A linha não está na ferramenta, está em quem produz a solução |

### Jogo da memória (6 pares)

| Problema | Exemplo que o ilustra |
|---|---|
| Dado faltante | Campo de colesterol preenchido com 0, que não significa zero e sim "não mediram" |
| Desbalanceamento | 97% das mensagens são normais e 3% são golpe, e o modelo responde "normal" para tudo |
| Vazamento | A mesma foto aparece no treino e no teste, e a nota fica linda e mentirosa |
| Atalho | O modelo separa notícia falsa pelo tamanho do texto, e não pelo conteúdo |
| Viés de coleta | O app foi testado só em hospital de capital e erra sistematicamente no interior |
| Dado sensível | Informação de saúde, origem racial, opinião política e qualquer dado de menor de idade |

### Ligação certa — pergunta → gráfico

| Pergunta de análise | Gráfico | Por quê |
|---|---|---|
| Os grupos se misturam nessa variável? | **Histograma sobreposto** | Mostra as distribuições lado a lado e a região onde as classes se confundem |
| Qual a mediana e onde estão os valores fora da curva? | **Boxplot** | Resume mediana, quartis, limites e atípicos num desenho compacto |
| Essa variável tem dois picos separados? | **Violino** | Um boxplot com duas concentrações parece igual a um com uma só; só o violino revela |
| Quem usa transporte escolar tem mais risco? | **Barras de proporção** | Variável binária: mostra a composição das classes dentro de cada valor |
| O risco só é alto quando faltas E queda de notas aparecem juntas? | **Dispersão** | É interação — o único gráfico que cruza duas variáveis |
| Qual grupo tem a maior média, e qual a incerteza? | **Barras de média com erro** | A barra mostra a média, a linha mostra a confiabilidade. Cuidado: a média esconde a dispersão |

### Prova final do Capítulo 1

| # | Pergunta | Resposta |
|---|---|---|
| 1 | Diferença central entre programar uma regra e treinar um modelo | Na regra **uma pessoa define o critério**; no modelo **o algoritmo encontra o critério a partir de exemplos** |
| 2 | Por que guardamos parte dos dados escondida | Para medir se o modelo **generaliza** — se acerta em casos que nunca viu |
| 3 | `In [7]` em cima e `In [3]` embaixo | As células foram **executadas fora de ordem**, e isso explica comportamento estranho |
| 4 | Colesterol = 0 em 172 registros | **Ausência disfarçada**: o exame não foi feito e alguém preencheu com zero |
| 5 | Descobrir se quem trabalha tem mais risco — qual gráfico | **Barras de proporção** (variável binária: quartis e histogramas não fazem sentido com dois valores) |
| 6 | Médias diferentes mas caixas muito sobrepostas | A variável **ajuda pouco sozinha**, porque há muita mistura entre os grupos |
| 7 | Único gráfico capaz de revelar interações | **Dispersão** |
| 8 | "Faltas" é a variável mais importante — isso prova causa? | **Não.** O modelo encontra associação estatística, e a causa real pode ser um fator que ninguém mediu |

**Explicação escrita (Camada 3):** *Por que uma lista de estudantes em risco pode ajudar ou prejudicar as mesmas pessoas?*
Checklist da resposta: (1) falei do que a escola **faz** com a lista, não só do modelo; (2) citei um uso que ajuda e um que prejudica; (3) usei ao menos um termo técnico do capítulo corretamente.

---

## Capítulo 2

### Parâmetro e efeito (6 pares)

| Parâmetro | Efeito |
|---|---|
| `max_depth` | Até onde a árvore pode crescer. Sem limite quase sempre gera sobreajuste |
| `n_estimators` | Quantas árvores votam. Mais estabiliza e custa tempo |
| `min_samples_leaf` | Mínimo por folha. Aumentar suaviza o modelo |
| `class_weight` | Compensa o desbalanceamento entre classes. Vem `None` — é o ganho isolado mais relevante do desafio |
| `C` | Inverso da regularização na logística. `< 1` regulariza mais |
| `max_iter` | Iterações da logística para convergir. Padrão 100 frequentemente não basta |

### Escolha do modelo (5 cenários)

| Cenário | Resposta | Por quê |
|---|---|---|
| Conselho escolar precisa de justificativa por escrito de cada sinalização | **Árvore de Decisão** | A sequência de perguntas se explica para qualquer pessoa. A floresta acertaria mais, mas ninguém explica 300 árvores num conselho. **O melhor desempenho não é sempre o melhor modelo** |
| Tabular, 4.000 linhas, pouco tempo, melhor resultado sem muito ajuste | **Floresta Aleatória** | Resiste a sobreajuste, não exige padronização, lida com interações, vai bem quase de fábrica |
| Texto com TF-IDF: 5.000 colunas, 3.000 exemplos | **Regressão Logística ou SVM Linear** | Modelos lineares lidam bem com esparso e alta dimensão. A árvore se perde; o RBF fica lento |
| Descritores de imagem padronizados, fronteira curva | **SVM com núcleo RBF** | Mede distância e desenha fronteiras curvas que contornam grupos. Sem padronizar, se perde |
| Linha de base em texto que treine em menos de 1 s | **Naive Bayes** | Imbatível em velocidade. Se um modelo sofisticado não superar o NB, algo está errado no pipeline |

### Exercício · Diagnóstico (`NotFittedError`)

```python
classificador = RandomForestClassifier(n_estimators=200)
palpites = classificador.predict(teste_X)   # <- predict ANTES do fit
classificador.fit(treino_X, treino_y)
```

1. **O `predict` foi chamado antes do `fit`**, então o modelo não tinha parâmetros ajustados. Correção: inverter as duas linhas.
2. **Mesmo invertido sobra problema:** todos os outros parâmetros continuam no padrão, incluindo `class_weight=None`, que em problema desbalanceado não compensa nada. **Rodar sem erro e rodar bem são coisas diferentes.**

### Exercício · Duas versões, uma certa

| | Código | Veredito |
|---|---|---|
| **A** | `fit_transform` no `X` **inteiro**, depois `train_test_split` | ❌ Média e desvio calculados incluindo a validação = **vazamento** |
| **B** | `train_test_split` primeiro, depois `fit_transform` no treino e `transform` na validação | ✅ **Correta** |

A versão A **não dá erro nenhum** — é assim que o problema passa despercebido.

### Exercício · Monte o experimento (ordem)

```
1. atributos = musicas.drop(columns=["genero"])        # separar características do alvo
2. train_test_split(atributos, generos, stratify=...)  # dividir treino/validação
3. classificador = RandomForestClassifier(...)         # construir o modelo
4. classificador.fit(treino_X, treino_y)               # treinar
5. palpites = classificador.predict(teste_X)           # prever
6. f1_score(teste_y, palpites, average="macro")        # medir
```

Cada passo depende do anterior existir. **A ordem não é convenção: é dependência.**

### Exercício · Caça ao erro (97% de acurácia e errado)

```python
escalador = StandardScaler()
X = escalador.fit_transform(X)   # <-- LINHA ERRADA: antes de dividir
X_tr, X_va, y_tr, y_va = train_test_split(X, y, stratify=y)
```

O escalador calculou média e desvio usando **todos** os dados, incluindo os de avaliação. Os 97% são ilusórios. Correto: dividir primeiro, ajustar o escalador só com o treino. **Nenhum erro apareceu.**

*(Variante do mesmo exercício: `modelo.fit(X_valida, y_valida)` chamado depois do fit no treino — o modelo passa a conhecer os dados de avaliação e a nota deixa de significar qualquer coisa.)*

### Exercícios de código (respostas)

```python
# menu / digitado - cap 2
if quero_explicar_a_decisao:
    modelo = DecisionTreeClassifier(max_depth=5)
elif quero_o_melhor_desempenho:
    modelo = RandomForestClassifier(n_estimators=300, n_jobs=-1)
else:
    modelo = LogisticRegression(max_iter=2000)
modelo.fit(X_treino, y_treino)
generos = modelo.predict(X_valida)

# stratify recebe y; average="macro"
X_treino, X_valida, y_treino, y_valida = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
nota = f1_score(y_valida, previsoes, average="macro")
```

**O parâmetro para cada objetivo:**

| Objetivo | Parâmetro |
|---|---|
| Impedir que a árvore cresça até decorar cada exemplo | `max_depth=6` |
| Compensar o desbalanceamento entre classes | `class_weight="balanced"` |
| Exigir pelo menos 5 exemplos em cada folha | `min_samples_leaf=5` |
| Dar mais iterações para a logística convergir | `max_iter=3000` |

### Prova final do Capítulo 2

| # | Pergunta | Resposta |
|---|---|---|
| 1 | 99% no treino e 62% na validação | **Sobreajuste** — decorou o treino e não generaliza. Reduzir `max_depth` ou aumentar `min_samples_leaf` |
| 2 | Por que a logística vai pior neste desafio | Os dados foram construídos **com interações**, e modelos lineares só somam efeitos isolados |
| 3 | Não declarei `class_weight` — o que acontece? | Assume o padrão `None` **em silêncio** e não compensa o desbalanceamento |
| 4 | Parâmetro vs. hiperparâmetro | Parâmetro o modelo **aprende** no treino; hiperparâmetro **vocês escolhem** antes |
| 5 | 80% de acurácia e 64% de F1-macro | **Desbalanceamento**: vai bem na majoritária e mal na minoritária |
| 6 | Número alto em "real alto, previsto baixo" | **Falsos negativos na classe crítica** — estudantes em risco que passaram despercebidos |
| 7 | Por que F1-macro é a métrica oficial | A média simples entre classes dá o mesmo peso à minoria, impedindo que ignorá-la passe batido |

**Explicação escrita:** *Um modelo tem 99% no treino e 62% na validação. Expliquem e digam o que fariam.*
Checklist: (1) nomeei o fenômeno; (2) expliquei por que a diferença treino/validação denuncia o problema; (3) propus ação concreta e disse o que esperava.

---

## Capítulo 3

### Que descritor separa estes materiais

| Par | Resposta |
|---|---|
| Metal × Papel | **Área reluzente** — metal reflete de forma concentrada, papel é fosco e difuso |
| Plástico colorido × Vidro transparente | **Saturação** — plástico tem cor forte, vidro tende ao neutro translúcido (o brilho seria parecido) |
| Depois de extrair 14 descritores, o problema passa a ser… | **Um problema tabular comum**, resolvido com os mesmos modelos do Cap. 2 |

### Oficina de filtros — o que cada filtro destrói

| Filtro | Descritor afetado | Consequência |
|---|---|---|
| Escala de cinza 100% | **Saturação média zera** | Cor e forma são informações independentes. Se o problema depende de cor, converter para cinza joga fora o sinal principal |
| Desfoque máximo | **Densidade de bordas despenca** (contraste também) | É o que uma foto tremida em produção faz |
| Dessaturar o plástico | Fica parecido com o **metal** (naturalmente pouco saturado) | Pré-processamento pode **destruir sinal**, não só limpar ruído |

### Resolva a confusão (vidro × plástico transparentes)

1. Descritor novo mais defensável: **intensidade e concentração dos reflexos** — o vidro tem índice de refração maior e gera reflexos concentrados; o plástico espalha. A resposta veio de entender a **física do material**, não de tentar parâmetros.
2. Se o descritor novo não melhorar nada: **verificar com um gráfico** se ele realmente separa as duas classes confundidas, antes de mexer no modelo. Distribuições sobrepostas = o descritor não carrega a informação suposta.

### Preveja a saída

```python
folhas = carregar_imagens(800)      # 800 fotos
tabela = medir_descritores(folhas)  # 14 medidas por foto
print(tabela.shape)                 # -> (800, 14)
```

A representação bruta teria 6.912 colunas por imagem.

### Parâmetro e efeito em visão

| Parâmetro | Efeito |
|---|---|
| `C` (no `SVC`) | Quanto tolera errar no treino. Alto demais retorce a fronteira (bolhas em volta de pontos isolados) |
| `gamma` | Alcance da influência de cada exemplo. Alto cria ilhas, baixo alisa. **Interage com `C`** |
| `kernel` | Forma da fronteira: reta, curva ou polinomial |
| `n_estimators` | Quantas árvores votam. Acima de ~300 o ganho é mínimo |
| `StandardScaler` | Todas as variáveis na mesma régua (média 0, desvio 1). SVM e logística precisam; floresta é indiferente |
| `max_iter` | Iterações da logística. Em visão, use 2000 ou 3000 |

### Exercícios de código (respostas)

```python
# menu
if modelo_escolhido == "Regressao Logistica":
    nucleo = LogisticRegression(C=forca_C, max_iter=3000)
elif modelo_escolhido == "SVM":
    nucleo = SVC(C=forca_C, kernel="rbf")
else:
    nucleo = RandomForestClassifier(n_estimators=250)
modelo = make_pipeline(StandardScaler(), nucleo)

# digitado
modelo = make_pipeline(StandardScaler(), SVC(C=8))
modelo.fit(X_treino, y_treino)
previsoes = modelo.predict(X_valida)

# escreva a extração
def extrair(imagem):
    cinza = imagem.converter_para_cinza()
    brilho    = cinza.media()            # tendência central
    contraste = cinza.desvio_padrao()    # dispersão
    reluzente = (cinza > 205).media()    # proporção de verdadeiros
    return [brilho, contraste, reluzente]
```

⚠️ Cuidado: `LinearRegression` existe mas prevê números contínuos, não classes. E a classe do SVM é **`SVC`** — escrever `SVM` não funciona.
Truque útil: **a média de uma comparação verdadeiro/falso devolve a proporção de verdadeiros.**

### Prova final do Capítulo 3

| # | Pergunta | Resposta |
|---|---|---|
| 1 | Por que não entregar os pixels direto | **Variáveis demais para poucos exemplos**, e a representação é frágil a qualquer deslocamento do objeto |
| 2 | Padronização faz diferença para quê | Modelos que medem **distância ou usam gradiente** (SVM, logística). Árvores são indiferentes |
| 3 | Vai bem no galpão e mal na cooperativa vizinha | **Deslocamento de domínio** |
| 4 | Nem vocês conseguiriam classificar as imagens erradas | O limite pode estar na **qualidade do rótulo**, não no modelo. Vale reportar |

**Explicação escrita:** *Por que 14 descritores funcionam melhor que 6.912 pixels?*
Checklist: (1) expliquei o que os descritores medem; (2) disse por que menos variáveis com mais significado ajudam; (3) liguei ao problema tabular do Cap. 2.

---

## Capítulo 4

### Caça ao vazamento

```python
tradutor = TfidfVectorizer(ngram_range=(1,2))
treino_vec = tradutor.fit_transform(treino_txt)
teste_vec  = tradutor.fit_transform(teste_txt)   # <-- LINHA ERRADA
```

O `fit` do vetorizador estima **vocabulário e IDF**, e o IDF depende das frequências de todo o conjunto recebido. O correto ali é **`transform`**. Nada quebra, nenhuma mensagem aparece, e a métrica ainda melhora — por isso esse erro derruba projeto de gente experiente.

### Oficina de texto (3 questões)

| Pergunta | Resposta |
|---|---|
| Golpe educado, sem urgência e sem link | **O escudo tende a errar** — os marcadores fortes que ele aprendeu não aparecem. Problema adversarial ao vivo |
| Frase cheia de gírias fora do corpus | **Nenhuma palavra entra na conta** — palavras fora do vocabulário. Consequência de justiça: quem escreve em registro diferente recebe pior tratamento sem que ninguém tenha programado isso |
| Trocar n-grama de 1 para 2 palavras | `conta bloqueada` passa a aparecer como **uma unidade só** — a informação de ordem sendo recuperada |

### A investigação (94% → 71% em dois meses, sem mudar o código)

| # | Pergunta | Resposta |
|---|---|---|
| 1 | Causa mais provável | **Deriva conceitual** — quem escreve golpes adaptou a redação e a distribuição mudou |
| 2 | Como perceber antes do prejuízo, sem rótulo verdadeiro | **Monitorar a distribuição das entradas**: proporção de termos fora do vocabulário, comprimento médio, distribuição das pontuações do modelo |
| 3 | O que entra no plano além do retreino | **Rotina periódica de retreino e monitoramento** — o problema vai se repetir indefinidamente. A solução não é um modelo, é um **processo** |

### Parâmetro e efeito em texto

| Parâmetro | Efeito |
|---|---|
| `max_features` | Tamanho do vocabulário. **A decisão que mais muda o resultado neste desafio** |
| `ngram_range` | Palavras isoladas ou também expressões. `(1,2)` faz "conta bloqueada" virar termo próprio |
| `min_df` | Descarta termos raríssimos, que seriam associados a uma classe por acidente |
| `sublinear_tf` | Escala logarítmica: a décima ocorrência não vale dez vezes a primeira |
| `alpha` (Naive Bayes) | Suavização de Laplace — evita que palavra nunca vista zere a conta. Valores 0.1–0.5 |
| `class_prior` | Substitui o `class_weight`, que o `MultinomialNB` **não aceita** (dá erro) |

### Exercícios de código (respostas)

```python
# menu / digitado - cap 4
vetorizador = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X_treino_vec = vetorizador.fit_transform(X_treino)   # treino: estima E aplica
X_valida_vec = vetorizador.transform(X_valida)       # validação: SÓ aplica
modelo = LogisticRegression(max_iter=3000)
modelo.fit(X_treino_vec, y_treino)
previsoes = modelo.predict(X_valida_vec)
```

**Configure o vetorizador:**

| Objetivo | Parâmetro |
|---|---|
| Limitar o vocabulário aos 4000 termos mais frequentes | `max_features=4000` |
| Capturar expressões de duas palavras ("nao recomendo") | `ngram_range=(1,2)` |
| Descartar termos em menos de 3 documentos | `min_df=3` |
| Unificar "voce" e "você" | `strip_accents="unicode"` |

⚠️ `CountVectorizer` existe, mas só conta ocorrências — sem o fator IDF que derruba o peso das palavras comuns.

### Monte o pipeline de texto (ordem)

```
1. train_test_split(avaliacoes, notas, stratify=notas)   # dividir PRIMEIRO
2. tradutor = TfidfVectorizer(...)
3. tradutor.fit_transform(treino_txt)
4. tradutor.transform(teste_txt)
5. classificador.fit(treino_vec, treino_y)
6. classificador.predict(teste_vec)
```

**Ponto crítico:** a divisão vem **antes** de qualquer coisa que aprenda com os dados.

### Prova final do Capítulo 4

| # | Pergunta | Resposta |
|---|---|---|
| 1 | O que o TF-IDF faz com a palavra "de" | **Derruba o peso quase a zero**, porque ela não distingue nada |
| 2 | Por que `fit_transform` no treino e `transform` na validação | O `fit` **estima** vocabulário e IDF a partir dos dados, e esses parâmetros não podem incorporar informação da validação |
| 3 | Característica mais perigosa do vazamento | **Não gera erro nenhum, roda normalmente e ainda melhora a métrica** |
| 4 | Aumentar a sensibilidade para pegar mais golpes — e a precisão? | **Tende a cair** — mais alarmes disparados, incluindo falsos |
| 5 | Por que o modelo tem prazo de validade | Quem escreve golpes **adapta a redação** em resposta ao filtro, mudando a distribuição |
| 6 | Estrutura em que quase tudo é zero | **Matriz esparsa** |

**Explicação escrita:** *O que é vazamento de dados e por que é perigoso justamente por não dar erro?*
Checklist: (1) defini vazamento com exemplo concreto do pipeline; (2) expliquei por que a métrica interna fica boa mesmo assim; (3) disse a consequência prática na competição.

> Esta é a pergunta que mais separa quem operou de quem compreendeu.
