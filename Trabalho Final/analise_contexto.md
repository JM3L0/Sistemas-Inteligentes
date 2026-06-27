# Análise Completa do Contexto do Trabalho

## 📋 Especificações do Trabalho
- **Disciplina:** Sistemas Inteligentes
- **Objetivo:** Aplicar técnicas de SI para resolver um problema real usando dataset de artigo Qualis A (>2018)
- **Entrega:** 01-02 de Julho

### Entregáveis
| Componente | Valor | Status |
|---|---|---|
| Artigo (template SBC, até 8 pgs) | 4.0 pts | ❌ **Não iniciado** |
| Apresentação oral (20 min + 5 min arguição) | 4.0 pts | ❌ **Não iniciado** |
| Repositório Git (público) | 2.0 pts | ❌ **Não verificado** |

## 🧬 Sobre o Projeto
- **Dataset:** TCGA Glioma (UCI ML Repository, id=759)
- **Problema:** Classificação binária de gliomas (LGG vs GBM)
- **839 instâncias**, 23 atributos (1 contínuo, 2 categóricos, 20 binários moleculares)
- **Modelos:** KNN, MLP, BernoulliNB

## ✅ O que JÁ está feito (Código - Notebook)
Todas as **7 etapas** do plano de codificação estão **completas e executadas**:

1. **Etapa 1 — Setup e Carregamento** ✅ (dataset carregado, 839 inst., 23 atributos)
2. **Etapa 2 — Pré-processamento** ✅ (StandardScaler, OneHotEncoder, remoção de nulos/duplicatas)
3. **Etapa 3 — Divisão dos Dados** ✅ (Hold-Out 70/30, K-Fold Estratificado 5 folds)
4. **Etapa 4 — Implementação dos Modelos** ✅ (Pipeline + GridSearchCV para KNN, MLP, BernoulliNB)
5. **Etapa 5 — Métricas** ✅ (Acurácia, Precisão, Recall, F1, Kappa, AUC-ROC)
6. **Etapa 6 — Análise de Erros** ✅ (FP, FN, erros por mutação IDH1/TP53/ATRX)
7. **Etapa 7 — Visualizações** ✅ (distribuição classes, heatmap, matrizes confusão, curvas ROC, comparação métricas)

### Resultados Obtidos
| Métrica | KNN | MLP | BernoulliNB |
|---|---|---|---|
| Acurácia | **0.8730** | 0.8016 | 0.8651 |
| Precisão | 0.8246 | **0.8333** | 0.8000 |
| Recall | 0.8868 | 0.6604 | **0.9057** |
| F1 | **0.8545** | 0.7368 | 0.8496 |
| Kappa | **0.7421** | 0.5810 | 0.7281 |
| AUC | 0.9038 | 0.8874 | **0.9083** |

### Análise de Erros
| Modelo | Total Erros | FP | FN | Erros IDH1 Wildtype |
|---|---|---|---|---|
| KNN | **32** | 20 | 12 | 23 |
| MLP | 50 | 14 | **36** | 40 |
| BernoulliNB | 34 | 24 | **10** | 25 |

## ❌ O que FALTA fazer

### Parte 2: Artigo (Template SBC, até 8 páginas)
O artigo **não foi iniciado**. Estrutura obrigatória:

1. **Introdução** — Contexto clínico (glioma, LGG vs GBM), problema, lacuna, objetivo
2. **Trabalhos Relacionados** — Tasci et al. (2022) + posicionamento do trabalho
3. **Metodologia** — Dataset, pré-processamento, modelos, validação, métricas
4. **Resultados** — Tabelas, matrizes, curvas ROC, análise de erros, discussão
5. **Conclusão** — Retomada, modelo mais adequado, contribuições, limitações, trabalhos futuros

### Apresentação Oral
- Slides não foram criados
- 20 minutos de exposição + 5 minutos de arguição

### Repositório Git
- Necessário ter repositório público no GitHub/GitLab

> [!IMPORTANT]
> A **codificação está 100% concluída**. O próximo passo é **escrever o artigo no template SBC** e preparar a **apresentação oral**. A data de entrega é **01-02 de Julho**.
