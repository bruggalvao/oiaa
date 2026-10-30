# Respostas das lacunas e configurações de referência

> Spoiler. Use só depois de tentar.

## Tabular (Farol e Boletim) — 13 lacunas

| # | Resposta | Onde |
|---|---|---|
| 1 | `"macro"` | piso (Bloco 1) |
| 2 | `nunique` | cardinalidades (Bloco 1) |
| 3 | `y` | `stratify` (célula A) |
| 4 | `CATEGORICAS` | `pd.get_dummies(..., columns=...)` (rede densa) |
| 5 | `X_tr_p` | a régua da rede densa aprende só com o treino |
| 6 | `X_tr[NUMERICAS]` | a régua da rede com embeddings: só numéricas, só treino |
| 7 | `True` | ponto de verificação |
| 8, 9 | `idx_cat`, `dim_cat` | `cat_idxs` recebe posições; `cat_dims`, cardinalidades |
| 10 | `taxa` | `optimizer_params=dict(lr=...)` do TabNet |
| 11 | `128` (ou qualquer divisor do lote) | lote virtual do TabNet |
| 12 | `AdamW` | otimizador |
| 13 | `CosineAnnealingLR` | agendador |

## Visão (Recomeço e Florada) — 12 lacunas

| # | Resposta |
|---|---|
| 1 | `"macro"` |
| 2 | `transforms` (`PESOS.transforms()`) |
| 3 | `preparo_tr` (treino, com aumento) |
| 4 | `preparo_val` (validação, sem aumento) |
| 5, 6, 7 | `fc` (ResNet-50) · `classifier[1]` (EfficientNet-B0) · `heads.head` (ViT-B/16) |
| 8 | `True` |
| 9, 10 | `AdamW`, `taxa_estagio1` |
| 11 | `taxa_estagio2` |
| 12 | `eval` |

## NLP (Voz do Cliente e Checagem) — 10 lacunas

| # | Resposta |
|---|---|
| 1 | `"macro"` |
| 2 | `NOME` (tradutor) |
| 3 | `dados["mensagem"]` (texto cru, sem `.lower()`) |
| 4 | `y` |
| 5 | `limite` |
| 6 | `NOME` (modelo — o mesmo do tradutor) |
| 7 | `AdamW` |
| 8 | `int(aquecimento / 100 * total)` |
| 9 | `True` |
| 10 | `eval` |

## Configurações de referência

Medidas neste repositório (CPU, sem GPU); F1-macro na validação do caderno / no teste oculto.
Os valores iniciais dos cadernos são modestos de propósito.

| Simulado | Configuração | Validação | Teste oculto |
|---|---|---|---|
| Farol | padrão do caderno (rede densa, taxa 1e-4, 40 épocas) | 0,489 | 0,496 |
| Farol | rede densa, taxa 1e-3, camadas (256, 128), dropout 0,3 | 0,649 | 0,671 |
| Farol | rede com embeddings, mesma configuração | 0,663 | 0,674 |
| Farol | TabNet, taxa 2e-2, n_d 16, n_steps 4 | 0,672 | 0,670 |
| Farol | floresta da Fase 1 (célula de comparação) | 0,658 | — |
| Boletim | padrão do caderno | 0,418 | 0,374 |
| Boletim | rede densa, taxa 1e-3, (128, 64), dropout 0,3 | 0,601 | 0,667 |
| Boletim | rede com embeddings, mesma configuração | 0,654 | 0,680 |
| Boletim | TabNet, taxa 2e-2 (base pequena demais: diverge) | 0,338 | 0,339 |
| Recomeço | EfficientNet-B0, 3 + 8 épocas, taxa 1e-3 / 1e-4, aumento leve | 0,861 | 0,889 |
| Florada | EfficientNet-B0, 3 + 8 épocas, taxa 1e-3 / 1e-4, aumento leve | 0,890 | 0,921 |
| Voz do Cliente | BERTimbau, **só 1% do acervo e 1 época** (teste de CPU) | 0,763 | 0,780 |
| Checagem | DistilBERT, só 10% do acervo e 1 época (fica no piso) | 0,399 | 0,412 |

Visão e NLP foram calibrados sem GPU; com a T4 e o acervo inteiro, os números de NLP devem
passar de 0,90 (a marca da Fase 1 no Voz do Cliente é 0,948 com TF-IDF + SVM linear).
