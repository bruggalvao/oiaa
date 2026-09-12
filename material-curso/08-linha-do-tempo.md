# Como chegamos até aqui — 12 marcos

Os modelos dos cadernos têm data, autor e um problema concreto que resolveram. **5 dos 12 marcos são literalmente as ferramentas de vocês.**

![Linha do tempo](img/06-linha-do-tempo.png)

| Ano | Marco | O que aconteceu | Onde aparece |
|---|---|---|---|
| **1950** | Turing faz a pergunta | Alan Turing publica um artigo perguntando se máquinas podem pensar — e logo troca a pergunta por outra, mais útil: seria possível conversar com uma máquina sem perceber que é uma máquina? A original ele considerava vaga demais. | — |
| **1956** | O nome nasce | Num encontro de verão em Dartmouth (EUA), um grupo pequeno se reúne para investigar se a inteligência pode ser descrita com precisão suficiente para uma máquina simular. Ali a expressão "inteligência artificial" passa a ser usada. | — |
| **1958** | O perceptron — e o primeiro exagero | Frank Rosenblatt constrói uma máquina que ajusta pesos a partir de exemplos em vez de seguir regras escritas. A imprensa anuncia que máquinas assim logo andariam, falariam e teriam consciência. Levou décadas para chegar a bem menos que isso. | — |
| **1958** | A regressão logística ganha forma | David Cox formaliza a análise de sequências binárias, consolidando o método. A curva em S é ainda mais antiga: veio do estudo do crescimento de populações, no século XIX. | **Cap. 2 · `LogisticRegression`** |
| **1966** | ELIZA e o espelho | Joseph Weizenbaum escreve um programa que devolve as frases da pessoa em forma de pergunta. Era casamento de padrões, sem nenhuma compreensão — e mesmo assim muita gente se afeiçoou e passou a lhe contar coisas íntimas. O próprio autor ficou incomodado. | — |
| **1970-80** | Os invernos da IA | As promessas não se cumprem no prazo. O financiamento seca, laboratórios fecham e o termo fica tão desgastado que pesquisadores passam a evitá-lo nos pedidos de verba. Acontece mais de uma vez. | — |
| **1972** | Karen Spärck Jones e o peso da palavra rara | A linguista e cientista da computação britânica propõe que a importância de um termo seja inversamente proporcional ao número de documentos em que ele aparece. Ela defendia que computação era importante demais para ser deixada só para os homens. | **Cap. 4 · o IDF do `TfidfVectorizer`** |
| **1984** | Breiman e as árvores de decisão | Leo Breiman, com Jerome Friedman, Richard Olshen e Charles Stone, publica o trabalho que estabelece as árvores de classificação e regressão: dividir os dados por perguntas sucessivas, escolhendo a cada passo a que melhor separa. | **Cap. 2 e 3 · `DecisionTreeClassifier`** |
| **1995** | A margem máxima | Corinna Cortes e Vladimir Vapnik publicam a formulação das máquinas de vetores de suporte que se tornaria padrão. Ideia central: entre todas as fronteiras que separam duas classes, prefira a que passa mais longe dos pontos dos dois lados. | **Cap. 3 · `SVC`** |
| **1997** | O tabuleiro cai | Um computador da IBM vence o campeão mundial de xadrez numa partida oficial. A máquina não pensava como enxadrista: avaliava um número enorme de posições por segundo. O efeito simbólico foi enorme. | — |
| **2001** | Breiman de novo: a assembleia | O mesmo Leo Breiman propõe as florestas aleatórias: centenas de árvores, cada uma com amostra e subconjunto de variáveis diferentes, e a resposta é o voto da maioria. **17 anos depois da árvore sozinha.** | **Cap. 2 e 3 · `RandomForestClassifier`** |
| **2012** | A virada da visão computacional | Numa competição anual de reconhecimento de imagens, uma rede neural profunda vence com margem tão grande sobre os métodos anteriores que a área inteira muda de rumo em poucos anos. | — |

## Como cada marco liga ao material

| Marco | Ligação com o desafio |
|---|---|
| Turing (1950) | Trocar uma pergunta impossível de medir por uma testável — é o que vocês fazem ao transformar "este estudante está em risco?" numa métrica calculável |
| Dartmouth (1956) | A área tem menos de 70 anos e nasceu de uma aposta. Muita coisa nela ainda está em disputa |
| Perceptron (1958) | Primeiro ciclo de promessa exagerada — e não foi o último. Lembrar disso toda vez que alguém disser que uma IA "entende" |
| Cox (1958) | É o "sistema de pontos do campeonato": cada variável ganha um peso, os pontos somam, a soma vira probabilidade |
| ELIZA (1966) | Atribuir compreensão a um sistema que só devolve padrões é humano e antigo. Por isso "o modelo achou" é sempre uma forma de falar |
| Invernos (1970-80) | Vários métodos que vocês usam foram lapidados nesses períodos, longe dos holofotes |
| Spärck Jones (1972) | Quando vocês chamam `TfidfVectorizer`, o "IDF" é literalmente essa ideia, de 1972 |
| Breiman (1984) | É o "jogo das vinte perguntas". E `max_depth` é exatamente o limite de quantas perguntas a árvore pode fazer |
| Cortes & Vapnik (1995) | É a "linha divisória do terreno". E é por isso que o SVM exige padronizar: ele mede distância |
| Xadrez (1997) | Vencer num domínio fechado com regras claras é muito diferente de decidir sobre pessoas, com dados incompletos. **A Fase 1 é do segundo tipo** |
| Breiman (2001) | A "assembleia de árvores". `n_estimators` é quantas árvores votam |
| ImageNet (2012) | No Cap. 3 vocês fazem o caminho **anterior** a esse — descritores calculados à mão — que é mais leve, roda em CPU e ensina o que acontece por dentro |

> **Uma coisa para levar daqui:** a área já passou por pelo menos dois períodos em que se prometeu muito mais do que se entregou, e o financiamento sumiu junto. Quando alguém disser que uma IA "entende", "pensa" ou "vai resolver tudo", vale lembrar que essa frase já foi dita várias vezes — e que o trabalho sério costuma acontecer longe do pico do entusiasmo.
