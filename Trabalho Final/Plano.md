# Plano Detalhado Completo

## PARTE 1: CODIFICAÇÃO

### ETAPA 1 — Setup e Carregamento

#### O que fazer
- Instalar a biblioteca `ucimlrepo`
- Carregar o dataset TCGA Glioma (id=759)
- Realizar uma exploração inicial dos dados

#### Código

```python
!pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

base = fetch_ucirepo(id=759)
X = base.data.features
y = base.data.targets
```

#### Exploração inicial

```python
X.shape
X.dtypes
X.isnull().sum()
X.duplicated().sum()

y['Grade'].value_counts()
y['Grade'].value_counts(normalize=True)

X.describe()
```

#### Análises recomendadas
- Quantidade de instâncias e atributos
- Tipos de dados
- Valores ausentes
- Duplicatas
- Distribuição das classes (LGG e GBM)
- Estatísticas descritivas
- Heatmap de correlação

#### Justificativa
A análise exploratória permite compreender o comportamento dos dados e fundamentar as decisões metodológicas adotadas posteriormente.

---

### ETAPA 2 — Pré-processamento

#### Identificação dos tipos de atributos

**Contínua**
- `Age_at_diagnosis`

**Categóricas**
- `Gender`
- `Race`

**Binárias moleculares**
- `IDH1`
- `TP53`
- `ATRX`
- `PTEN`
- `EGFR`
- `CIC`
- `MUC16`
- `PIK3CA`
- `NF1`
- `PIK3R1`
- `FUBP1`
- `RB1`
- `NOTCH1`
- `BCOR`
- `CSMD3`
- `SMARCA4`
- `GRIN2A`
- `IDH2`
- `FAT4`
- `PDGFRA`

#### Estratégia

- StandardScaler para atributos contínuos
- OneHotEncoder para atributos categóricos
- Manter atributos binários sem normalização

#### Código

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

colunas_continuas = ['Age_at_diagnosis']
colunas_categoricas = ['Gender', 'Race']

colunas_binarias = [
    col for col in X.columns
    if col not in colunas_continuas + colunas_categoricas
]

preprocessor = ColumnTransformer(
    transformers=[
        ('continuas', StandardScaler(), colunas_continuas),
        ('categoricas', OneHotEncoder(drop='first', handle_unknown='ignore'), colunas_categoricas),
        ('binarias', 'passthrough', colunas_binarias)
    ]
)
```

#### Limpeza dos dados

- Remover registros com valores ausentes
- Converter `Age_at_diagnosis` para numérico, se necessário
- Documentar quantidade de registros removidos

#### Justificativa

As variáveis binárias já estão padronizadas entre 0 e 1 e não necessitam normalização. Alterar essa escala pode introduzir distorções na representação das mutações.

---

### ETAPA 3 — Divisão dos Dados

#### Hold-Out (70/30)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y['Grade'],
    test_size=0.30,
    stratify=y['Grade'],
    random_state=42
)
```

#### K-Fold Estratificado (5 folds)

```python
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

#### Cuidados

- Utilizar estratificação
- Utilizar `random_state=42`
- Aplicar o K-Fold somente sobre o conjunto de treinamento

---

### ETAPA 4 — Implementação dos Modelos

Cada modelo será executado dentro de um Pipeline e otimizado por GridSearchCV.

---

## Modelo 1 — KNN

### Justificativa

Classifica amostras pela proximidade entre pacientes. É um modelo simples e interpretável para dados biomédicos.

### Hiperparâmetros

```python
from sklearn.neighbors import KNeighborsClassifier

params_knn = {
    'modelo__n_neighbors': [3, 5, 7, 9, 11],
    'modelo__weights': ['uniform', 'distance'],
    'modelo__metric': ['euclidean', 'manhattan', 'hamming']
}
```

---

## Modelo 2 — MLP

### Justificativa

Capaz de capturar relações não lineares entre mutações genéticas.

### Hiperparâmetros

```python
from sklearn.neural_network import MLPClassifier

params_mlp = {
    'modelo__hidden_layer_sizes': [(32,), (64,), (32,16), (64,32)],
    'modelo__activation': ['relu', 'tanh'],
    'modelo__solver': ['adam', 'lbfgs'],
    'modelo__alpha': [0.0001, 0.001, 0.01],
    'modelo__learning_rate': ['constant', 'adaptive']
}

mlp = MLPClassifier(
    random_state=42,
    early_stopping=True,
    max_iter=500
)
```

---

## Modelo 3 — Bernoulli Naive Bayes

### Justificativa

Foi desenvolvido especificamente para atributos binários, sendo adequado ao perfil do dataset.

### Hiperparâmetros

```python
from sklearn.naive_bayes import BernoulliNB

params_bernoulli = {
    'modelo__alpha': [1e-9, 1e-3, 0.1, 0.5, 1.0],
    'modelo__binarize': [None, 0.0, 0.5],
    'modelo__fit_prior': [True, False]
}
```

---

### Estrutura do Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def executar_modelo(modelo, params, nome):

    pipeline = Pipeline([
        ('preprocessamento', preprocessor),
        ('modelo', modelo)
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=kfold,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print(f'\n=== {nome} ===')
    print(grid.best_params_)
    print(grid.best_score_)

    return grid.best_estimator_
```

---

### ETAPA 5 — Métricas

#### Métricas utilizadas

- Acurácia
- Precisão
- Recall
- F1-Score
- Kappa de Cohen
- AUC-ROC

#### Código

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def avaliar_modelo(modelo, nome):

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:,1]

    print(f'\n=== {nome} ===')

    print('Acurácia:', accuracy_score(y_test, y_pred))
    print('Precisão:', precision_score(y_test, y_pred, pos_label='GBM'))
    print('Recall:', recall_score(y_test, y_pred, pos_label='GBM'))
    print('F1:', f1_score(y_test, y_pred, pos_label='GBM'))
    print('Kappa:', cohen_kappa_score(y_test, y_pred))
    print('AUC:', roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=['LGG','GBM']
    )

    disp.plot()
```

---

### ETAPA 6 — Análise de Erros

#### Coleta dos erros

```python
erros_knn = X_test[y_pred_knn != y_test].copy()
erros_mlp = X_test[y_pred_mlp != y_test].copy()
erros_nb = X_test[y_pred_nb != y_test].copy()
```

#### Informações adicionadas

```python
erros_knn['real'] = y_test[y_pred_knn != y_test]
erros_knn['predito'] = y_pred_knn[y_pred_knn != y_test]
```

#### Análises

1. Erros associados às mutações:
   - IDH1
   - TP53
   - ATRX

2. Comparação entre:
   - Falsos Positivos
   - Falsos Negativos

3. Tabela comparativa

| Modelo | Total de Erros | FP | FN | Erros com IDH1 Wildtype |
|----------|----------|----------|----------|----------|
| KNN | ? | ? | ? | ? |
| MLP | ? | ? | ? | ? |
| BernoulliNB | ? | ? | ? | ? |

---

### ETAPA 7 — Visualizações

Produzir:

1. Distribuição das classes
2. Heatmap de correlação
3. Matrizes de confusão
4. Curvas ROC
5. Comparação das métricas
6. Heatmap ou tabela de erros

---

# PARTE 2: ARTIGO

## Seção 1 — Introdução

### Contexto clínico
- Conceito de glioma
- Diferença entre LGG e GBM
- Impacto clínico

### Problema
- Diagnóstico molecular é caro e demorado
- Necessidade de apoio computacional

### Lacuna
- Predomínio de trabalhos baseados em ensembles complexos
- Pouca investigação sobre modelos simples e interpretáveis

### Objetivo
Avaliar KNN, MLP e BernoulliNB para classificação de gliomas utilizando o dataset TCGA e analisar seus padrões de erro.

---

## Seção 2 — Trabalhos Relacionados

### Tasci et al. (2022)

- Ensemble baseado em votação
- Feature Selection usando WOE, RFE, RF e LASSO
- Acurácia de aproximadamente 87,6%

### Posicionamento do trabalho

Diferentemente dos trabalhos anteriores, esta pesquisa avalia modelos simples adequados ao perfil binário do dataset e investiga seus padrões de erro.

---

## Seção 3 — Metodologia

### 3.1 Dataset
- TCGA
- 839 instâncias
- 23 atributos
- Classes LGG e GBM

### 3.2 Pré-processamento
- Tratamento de nulos
- StandardScaler
- OneHotEncoder
- Preservação das variáveis binárias

### 3.3 Modelos
- KNN
- MLP
- BernoulliNB

### 3.4 Validação
- Hold-Out 70/30
- K-Fold Estratificado (5 folds)

### 3.5 Métricas
- Acurácia
- Precisão
- Recall
- F1
- Kappa
- AUC-ROC

---

## Seção 4 — Resultados

### 4.1 Resultados Quantitativos
Tabela consolidada das métricas.

### 4.2 Matrizes de Confusão
Análise dos erros por classe.

### 4.3 Curvas ROC
Comparação das capacidades discriminativas.

### 4.4 Análise de Erros
Avaliação dos erros associados às mutações relevantes.

### 4.5 Discussão
Interpretação dos resultados e comparação com a literatura.

---

## Seção 5 — Conclusão

- Retomada do objetivo
- Modelo mais adequado
- Contribuição da análise de erros
- Limitações
- Trabalhos futuros

---

# Ordem Recomendada

1. Setup e carregamento
2. Pré-processamento
3. Divisão dos dados
4. Implementação dos modelos
5. GridSearchCV
6. Avaliação
7. Análise de erros
8. Visualizações
9. Escrita do artigo