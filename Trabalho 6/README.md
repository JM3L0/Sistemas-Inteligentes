# Trabalho 6 — Sistemas Inteligentes
## Regressão Linear Múltipla com Dados de Imagem de Ultrassom Bovino

---

## 📌 Objetivo

Prever o **peso corporal** de bovinos da raça Anglo-Nubian utilizando **regressão linear múltipla**, com base em medidas morfométricas e de ultrassom coletadas em campo.

**Meta de precisão:** R² ≥ 0.8

---

## 📂 Arquivos do Projeto

| Arquivo | Descrição |
|---|---|
| `Dados_Projeto_Imagem_Ultrassom.xlsx.csv` | Base de dados original (214 registros) |
| `regressao_ultrassom.py` | Script Python modularizado (execução direta) |
| `regressao_ultrassom.ipynb` | Jupyter Notebook com explicações por etapa |

---

## 🗃️ Base de Dados

O dataset contém dados de **214 animais** coletados em diferentes propriedades do Maranhão e Piauí. Cada registro inclui informações de identificação, medidas de ultrassom e medidas morfométricas do animal.

### Variável Alvo (Y)
| Coluna | Descrição |
|---|---|
| `PESO` | Peso corporal do animal em quilogramas |

### Features (X) — Preditores
As features foram selecionadas em dois grupos:

**Medidas de ultrassom e biométricas do exame:**
| Coluna | Descrição |
|---|---|
| `IDADE` | Idade do animal em dias |
| `AOL (cm²)` | Área de Olho de Lombo |
| `COL (cm)` | Comprimento do Olho de Lombo |
| `POL (cm)` | Profundidade do Olho de Lombo |
| `RATIO (cm)` | Razão entre dimensões do Olho de Lombo |
| `EGE (mm)` | Espessura de Gordura na Eminência |
| `MOL` | Morfologia do Olho de Lombo |
| `EC` | Escore de Condição Corporal |

**Medidas morfométricas (a partir de AC):**
| Coluna | Descrição |
|---|---|
| `AC` | Altura na Cernelha |
| `AG` | Altura na Garupa |
| `CC` | Comprimento Corporal |
| `AP` | Altura do Peito |
| `P.C` | Profundidade do Corpo |
| `CT` | Comprimento do Tronco |
| `CO` | Comprimento Oblíquo |
| `CCAB` | Comprimento da Cabeça |
| `LIL` | Largura Interna do Lombo |
| `LIS` | Largura Interna da Sobreespalha |
| `Cga` | Circunferência da Garganta |
| `Cper` | Circunferência do Pescoço |
| `PerPe` | Perímetro do Peito |
| `Ccau` | Circunferência da Cauda |
| `DC` | Diâmetro do Corpo |

> **Nota:** As colunas `LR`, `LCAB` e `CE` foram automaticamente descartadas por conterem mais de 50% de valores ausentes (presentes apenas em machos reprodutores).

---

## ⚙️ Pipeline — Etapas de Execução

### Etapa 1 — Carregamento de Dados

```python
df = carregar_dados("Dados_Projeto_Imagem_Ultrassom.xlsx.csv")
```

O arquivo CSV é lido com `pandas` (usado **somente para I/O**). Todos os cálculos subsequentes são feitos com estruturas Python puras (listas de listas).

---

### Etapa 2 — Filtrar X e Y

```python
X, y, cols_x = filtrar_xy(df)
```

**O que acontece aqui:**
1. Define `PESO` como variável alvo (Y)
2. Lista todas as colunas candidatas a features (X)
3. Converte o formato numérico brasileiro (`1,23` → `1.23`)
4. Remove colunas com **mais de 50% de valores ausentes** (NaN)
5. Remove todas as linhas que ainda contenham NaN
6. Retorna `X` e `y` como **listas Python puras**

---

### Etapa 3 — Pré-Processamento: Normalização Min-Max

```python
X_norm, minimos, maximos = normalizar_minmax(X)
```

**Fórmula aplicada por feature:**

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Por quê normalizar?**  
As features têm escalas muito diferentes (IDADE em dias vs. medidas em cm). Sem normalização, features com valores maiores dominariam os coeficientes e prejudicariam a inversão da matriz.

**Implementação manual:** percorre cada coluna calculando mínimo e máximo com loops puros, sem usar `numpy` ou `min()`/`max()` do Python nos cálculos principais.

---

### Etapa 4 — Divisão Treino/Teste (70/30, seed=50)

```python
X_treino, y_treino, X_teste, y_teste = dividir_treino_teste(X_norm, y, 0.7, seed=50)
```

| Conjunto | Proporção | Amostras |
|---|---|---|
| Treino | 70% | 149 |
| Teste | 30% | 65 |

**Embaralhamento com seed fixo:**  
Utiliza um **Gerador Congruencial Linear (LCG)** implementado manualmente com o algoritmo de Fisher-Yates shuffle. O LCG garante reprodutibilidade sem depender do módulo `random`.

```
Xₙ₊₁ = (a · Xₙ + c) mod m
a = 1.664.525  |  c = 1.013.904.223  |  m = 2³²  (constantes de Knuth)
```

---

### Etapa 5 — Regressão Linear (Equação Normal com Ridge)

```python
beta = regressao_linear(X_treino, y_treino)
```

**Modelo matemático:**

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p$$

**Solução pela Equação Normal com Regularização Ridge:**

$$\beta = (X^T X + \lambda I)^{-1} X^T y$$

**Por que a regularização (λ)?**  
Com 23 features altamente correlacionadas entre si (medidas do mesmo animal), a matriz $X^T X$ tende a ser quase-singular (determinante próximo de zero), tornando a inversão numericamente instável. O termo $\lambda I$ (com λ = 0,0001) estabiliza a diagonal sem alterar significativamente os coeficientes — essa técnica é chamada de **Regressão Ridge** ou **Regularização de Tikhonov**.

**Sub-funções implementadas manualmente:**

| Função | O que faz |
|---|---|
| `_adicionar_bias(X)` | Adiciona coluna de 1s para o intercepto β₀ |
| `_transpor(M)` | Calcula Xᵀ percorrendo índices com loops |
| `_mult_mat(A, B)` | Multiplicação de matrizes com três loops aninhados |
| `_inverter_gauss(M)` | Inversão por Eliminação de Gauss-Jordan com pivotamento parcial |
| `_regularizar_XtX(M)` | Soma λ na diagonal principal de XᵀX |
| `_mult_mat_vec(M, v)` | Multiplica matriz por vetor |

**Eliminação de Gauss-Jordan:**  
A inversão é feita criando a matriz aumentada `[M | I]` e aplicando operações elementares de linha até transformar o lado esquerdo na identidade — o lado direito se torna a inversa de M.

---

### Etapa 6 — Cálculo do R²

```python
r2 = calcular_r2(y_teste, y_pred_teste)
```

**Fórmula do Coeficiente de Determinação:**

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

| Termo | Significado |
|---|---|
| $SS_{res}$ | Soma dos quadrados dos resíduos (erro do modelo) |
| $SS_{tot}$ | Soma dos quadrados totais (variância dos dados) |
| $\bar{y}$ | Média dos valores reais |

**Interpretação:**
- R² = 1 → modelo perfeito
- R² = 0 → modelo equivale a prever sempre a média
- R² < 0 → modelo pior que prever a média

---

### Etapa 7 — Validação da Precisão

```python
aprovado, status = validar_precisao(r2, limiar=0.8)
```

Verifica se o R² no conjunto de **teste** atinge o limiar mínimo de **0.8**.

---

## 📊 Resultados Obtidos

| Métrica | Valor |
|---|---|
| Amostras totais | 214 |
| Features utilizadas | 23 |
| R² no treino | 0.7435 |
| **R² no teste** | **0.8391** |
| Limiar mínimo | 0.8000 |
| **Resultado** | **✅ APROVADO** |

---

## 🏗️ Arquitetura do Código

```
regressao_ultrassom.py
│
├── MÓDULO 1 — carregar_dados()
│     └── Leitura do CSV com pandas (único uso de biblioteca)
│
├── MÓDULO 2 — filtrar_xy()
│     ├── _converter_br_para_float()   ← formato "1,23" → 1.23
│     └── Seleção, limpeza e extração de X e y
│
├── MÓDULO 3 — normalizar_minmax()
│     ├── _min_lista()                 ← mínimo manual
│     └── _max_lista()                 ← máximo manual
│
├── MÓDULO 4 — dividir_treino_teste()
│     └── _embaralhar_lcg()            ← LCG + Fisher-Yates manual
│
├── MÓDULO 5 — regressao_linear()
│     ├── _adicionar_bias()
│     ├── _transpor()
│     ├── _mult_mat()
│     ├── _inverter_gauss()            ← Gauss-Jordan manual
│     ├── _regularizar_XtX()           ← Ridge manual
│     ├── _mult_mat_vec()
│     └── prever()
│
├── MÓDULO 6 — calcular_r2()
│     └── _media()                     ← média manual
│
├── MÓDULO 7 — validar_precisao()
│
└── executar_pipeline()                ← orquestra tudo
```

---

## 🚀 Como Executar

### Opção 1 — Script Python
```bash
python regressao_ultrassom.py
```

### Opção 2 — Jupyter Notebook
```bash
jupyter notebook regressao_ultrassom.ipynb
```
Execute as células em ordem (Shift+Enter).

---

## 📚 Referências Teóricas

- **Regressão Linear Múltipla:** Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- **Equação Normal:** Montgomery, D. C. (2012). *Introduction to Linear Regression Analysis*. Wiley.
- **Regularização de Tikhonov:** Tikhonov, A. N. (1963). Solution of incorrectly formulated problems.
- **Eliminação de Gauss-Jordan:** Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*. Johns Hopkins.
- **LCG:** Knuth, D. E. (1997). *The Art of Computer Programming, Vol. 2*. Addison-Wesley.
