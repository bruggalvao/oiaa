# Resumo da sessão
## O pedido
Construir uma rede neural para classificar risco_evasao em treino.csv, seguindo a metodologia das aulas de MLP em _ref, testando várias possibilidades de camadas/ativações/configurações, e superar o F1-macro de 0.72. No final, gerar submissao_tabular.csv com o melhor modelo.

## Ambiente
TensorFlow estava quebrado localmente (ModuleNotFoundError: tensorflow.python). Contornei com Keras 3.15.1 + backend torch. Como o Keras levava ~30s por treino, construí um harness em PyTorch puro (_lab.py) que roda uma config em 4 folds em ~8s — isso viabilizou a varredura ampla. O modelo final foi traduzido de volta para Keras.

## Os dados
3236 alunos, 35 variáveis, sem nulos. Classes desbalanceadas: Baixo 49.9%, Médio 17.9%, Alto 32.2%. Baselines no holdout de 25%: LogReg 0.693, RF 0.703, HGB 0.677, MLP simples 0.707.

## O que foi testado
Features (_exp02): três variantes. A v2 (deltas entre semestres, médias, flags de sem_avaliacao para distinguir nota 0.0 de "não avaliado", interações) deu 0.7132 contra 0.6993 das colunas cruas. StandardScaler venceu MinMax, Robust, Quantile e Power.

## Arquiteturas e ativações (_exp03): 180 combinações — 18 arquiteturas (de (16,) a (256,128,64,32)) × 10 ativações (relu, tanh, elu, selu, gelu, swish, leaky_relu, mish, softplus, sigmoid). Resultado que mudou o plano: tudo caiu entre 0.694 e 0.714. Arquitetura e ativação quase não importam neste dataset.

Hiperparâmetros de treino (_exp04): busca por coordenadas nas 3 melhores arquiteturas. class_weight balanceado foi o fator dominante (+0.024 a +0.031). BatchNorm sempre piorou, dropout 0.5 foi desastroso (0.639), SGD ruim, label_smoothing não ajudou, val_loss melhor que val_f1 para early stopping.

Ensemble e regra de decisão (_exp05): média de 5 sementes deu ganho pequeno mas consistente (~+0.005). Já o ajuste de multiplicadores por classe falhou — pesos erráticos (0.5, 1.41, 0.71), porque class_weight já corrige o desbalanceamento e a fatia de calibração era pequena demais.

## Entrega
gerar_submissao.py — lê treino.csv e teste.csv, gera 
submissao_tabular.csv
 (1078 linhas, formato id_estudante,risco_previsto idêntico ao anterior, ids na ordem correta).

Modelo: 128-64-32 com selu, AdamW (lr 1e-3, wd 1e-4), batch 16, dropout 0.2, class_weight balanceado, ensemble de 5 sementes. Segue as aulas: Sequential/Dense/softmax de 3 saídas/sparse_categorical_crossentropy/EarlyStopping com restore_best_weights. Duas etapas: valida em holdout de 25%, depois retreina com os 3236 alunos para prever o teste, sempre com fatia interna separada para o early stopping.

Resultado e a ressalva
F1-macro 71.64%, acurácia 76.27%. Isso empata com seus 0.72, não supera.

O gargalo é a classe Médio: F1 de 0.53 contra 0.85 (Baixo) e 0.77 (Alto). Dos 145 alunos de risco médio na validação, 34 vão para Baixo e 26 para Alto.

Ficou pendente o que tem mais chance de destravar isso: features v3 (a alavanca com maior retorno comprovado — a v2 já rendeu +0.014), ensemble misturando arquiteturas diferentes em vez de só sementes, e a regra de decisão calibrada por OOF cross-fitting em vez da fatia pequena que falhou.

## Arquivos criados
gerar_submissao.py (entrega), _lab.py e _lab2.py (harness), _exp01 a _exp06 (experimentos), _res03/_res04/_res05 e _log03 a _log06 (resultados e logs da varredura), além dos .txt extraídos dos PDFs em _ref/.