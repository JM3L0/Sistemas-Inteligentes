# Passo a Passo do Código — Regressão Linear Múltipla

> **Arquivo:** `regressao_ultrassom.py`  
> **Disciplina:** Sistemas Inteligentes  
> **Objetivo:** Prever o **PESO** de bovinos com R² ≥ 0.8

---

## Visão Geral do Fluxo

```
CSV  →  [1] Carregar  →  [2] Filtrar X,Y  →  [3] Normalizar
     →  [4] Dividir   →  [5] Regredir     →  [6] R²  →  [7] Validar
```

---

## Passo 1 — Carregar os Dados

```python
def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, encoding="utf-8")
    return df
```

**O que faz:**  
Lê o arquivo CSV e retorna um DataFrame do pandas.  
`pandas` é utilizado **somente aqui**, apenas para leitura do arquivo.  
Todo o restante do código usa listas Python puras.

**Resultado:** 214 registros com 56 colunas brutas.

---

## Passo 2 — Filtrar X e Y

```python
def filtrar_xy(df):
    col_y = "PESO"
    cols_candidatas_x = [
        "IDADE", "AOL (cm²)", "COL (cm)", "POL (cm)", "RATIO (cm)",
        "EGE (mm)", "MOL", "EC",
        "AC", "AG", "CC", "AP", "P.C", "CT", "CO", "CCAB",
        "LR", "LCAB", "LIL", "LIS", "Cga", "Cper", "PerPe", "Ccau",
        "DC", "CE"
    ]
```

**O que faz (em 4 sub-passos):**

### 2.1 — Define Y e os candidatos a X
- **Y** = coluna `PESO` (o que queremos prever)
- **X** = lista de colunas de medidas do animal

### 2.2 — Converte formato brasileiro
```python
def _converter_br_para_float(valor):
    if isinstance(valor, str):
        return valor.replace(",", ".")   # "5,32" → "5.32"
    return valor
```
Os dados usam vírgula como separador decimal (padrão BR).  
Essa função converte cada valor antes de transformar em número.

### 2.3 — Remove colunas com muitos valores vazios
```python
limite_nan = 0.5 * len(subset)
colunas_densas = [c for c in cols_x if subset[c].isna().sum() <= limite_nan]
```
Qualquer coluna com **mais de 50% de NaN** é descartada automaticamente.  
No nosso caso: `LR`, `LCAB` e `CE` foram removidas (existem só em machos).

### 2.4 — Remove linhas com NaN restante
```python
subset = subset.dropna()
```
Após remover as colunas esparsas, elimina as linhas que ainda têm célula vazia.

**Resultado:** 214 amostras limpas com 23 features.

---

## Passo 3 — Pré-Processamento: Normalização Min-Max

```python
def normalizar_minmax(X):
    for j in range(n_features):
        coluna = [X[i][j] for i in range(n_amostras)]
        minimos.append(_min_lista(coluna))
        maximos.append(_max_lista(coluna))

    for i in range(n_amostras):
        for j in range(n_features):
            denom = maximos[j] - minimos[j]
            linha.append((X[i][j] - minimos[j]) / denom)
```

**Por que normalizar?**  
As features têm escalas completamente diferentes:
- `IDADE` pode ser 2000+ dias
- `P.C` é ~10 cm

Sem normalizar, as features com valores grandes dominam o cálculo e distorcem os coeficientes β.

**Fórmula aplicada coluna a coluna:**

```
x_norm = (x - x_min) / (x_max - x_min)
```

Todos os valores ficam no intervalo [0, 1].

**Funções auxiliares (sem usar min/max do Python):**
```python
def _min_lista(lst):
    minimo = lst[0]
    for v in lst:
        if v < minimo:
            minimo = v
    return minimo
```

**Resultado:** Matriz X com todos os valores entre 0 e 1.

---

## Passo 4 — Divisão Treino/Teste (70/30, seed=50)

```python
def dividir_treino_teste(X, y, razao_treino=0.7, seed=50):
    indices = _embaralhar_lcg(len(X), seed)
    n_treino = int(len(X) * razao_treino)
    X_treino = X_emb[:n_treino]
    X_teste  = X_emb[n_treino:]
```

**Por que embaralhar antes de dividir?**  
Os dados estão ordenados por propriedade/fazenda. Sem embaralhar, treino e teste teriam animais de fazendas completamente diferentes, gerando um modelo enviesado.

**Como o embaralhamento é feito (Gerador Congruencial Linear):**
```python
def _embaralhar_lcg(n, seed=50):
    a, c, m = 1_664_525, 1_013_904_223, 2 ** 32
    estado = seed
    for i in range(n - 1, 0, -1):
        estado = (a * estado + c) % m   # gera próximo número pseudo-aleatório
        j = estado % (i + 1)            # índice aleatório dentro do range
        indices[i], indices[j] = indices[j], indices[i]  # troca (Fisher-Yates)
```

É um gerador de números pseudo-aleatórios implementado à mão, com a fórmula:

```
X(n+1) = (a * X(n) + c) mod m
```

O `seed=50` garante que **sempre o mesmo embaralhamento** seja gerado — resultado reproduzível.

**Resultado:**
- Treino: 149 amostras (70%)
- Teste: 65 amostras (30%)

---

## Passo 5 — Treinamento: Regressão Linear

Esta é a etapa central. O modelo calcula os coeficientes **β** que minimizam o erro quadrático.

### 5.1 — Adicionar coluna de bias (intercepto)

```python
def _adicionar_bias(X):
    return [[1.0] + row[:] for row in X]
```

Acrescenta uma coluna de `1s` à esquerda da matriz X.  
Isso permite que o modelo aprenda um **intercepto β₀** (valor base quando todas as features são zero).

```
X original:              X com bias:
[x1, x2, x3]    →    [1, x1, x2, x3]
[x1, x2, x3]    →    [1, x1, x2, x3]
```

### 5.2 — Transpor a matriz

```python
def _transpor(M):
    n_lin, n_col = len(M), len(M[0])
    return [[M[i][j] for i in range(n_lin)] for j in range(n_col)]
```

Inverte linhas e colunas. Necessário para calcular XᵀX.

### 5.3 — Multiplicar matrizes

```python
def _mult_mat(A, B):
    for i in range(n):
        for j in range(m):
            soma = 0.0
            for k in range(p):
                soma += A[i][k] * B[k][j]
            C[i][j] = soma
```

Multiplicação clássica de matrizes com três loops aninhados.  
Usada para calcular **XᵀX** e depois **XᵀX⁻¹ · Xᵀy**.

### 5.4 — Regularização Ridge (Tikhonov)

```python
def _regularizar_XtX(XtX, lam=1e-4):
    for i in range(n):
        XtX_reg[i][i] += lam   # soma lambda na diagonal principal
```

**Por que é necessário?**  
Com 23 features que medem o mesmo animal de formas parecidas (altura, comprimento, circunferências), a matriz XᵀX fica quase-singular — seu determinante fica próximo de zero e a inversão se torna numericamente instável.

Somando um pequeno valor `λ = 0.0001` na diagonal principal, a matriz se torna invertível sem alterar muito os coeficientes. Isso é chamado de **Regressão Ridge**.

### 5.5 — Inverter a matriz (Gauss-Jordan)

```python
def _inverter_gauss(M):
    # Cria matriz aumentada [M | Identidade]
    aug = [M[i][:] + [1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        # Pivotamento: escolhe a linha com maior valor absoluto
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]

        # Normaliza a linha do pivô (divide pelo elemento pivô)
        pivo = aug[col][col]
        for j in range(2*n):
            aug[col][j] /= pivo

        # Zera todos os outros elementos da coluna
        for row in range(n):
            if row != col:
                fator = aug[row][col]
                for j in range(2*n):
                    aug[row][j] -= fator * aug[col][j]

    # A parte direita da matriz aumentada é a inversa
    return [aug[i][n:] for i in range(n)]
```

**Como funciona:**
1. Cria a matriz `[M | I]` (M ao lado da identidade)
2. Aplica operações de linha até M virar a identidade
3. O que sobra no lado direito é M⁻¹

O **pivotamento parcial** (trocar linhas para colocar o maior elemento como pivô) evita divisão por valores muito pequenos.

### 5.6 — Equação Normal: calcular β

```python
def regressao_linear(X_treino, y_treino, lam=1e-4):
    X_b     = _adicionar_bias(X_treino)   # adiciona coluna de 1s
    Xt      = _transpor(X_b)              # Xᵀ
    XtX     = _mult_mat(Xt, X_b)         # XᵀX
    XtX_reg = _regularizar_XtX(XtX, lam) # XᵀX + λI
    XtXi    = _inverter_gauss(XtX_reg)   # (XᵀX + λI)⁻¹
    Xty     = _mult_mat_vec(Xt, y_treino) # Xᵀy
    beta    = _mult_mat_vec(XtXi, Xty)   # β = (XᵀX + λI)⁻¹ Xᵀy
    return beta
```

**A fórmula em passos:**

```
β = (XᵀX + λI)⁻¹ · Xᵀy
```

Cada `β[i]` é o **peso** que o modelo deu para a feature `i` na predição do peso.

### 5.7 — Fazer predições

```python
def prever(X, beta):
    X_b = _adicionar_bias(X)
    return [sum(X_b[i][j] * beta[j] for j in range(len(beta)))
            for i in range(len(X_b))]
```

Para cada amostra, calcula:
```
ŷ = β₀·1 + β₁·x₁ + β₂·x₂ + ... + β₂₃·x₂₃
```

**Resultado:** vetor β com 24 valores (1 bias + 23 pesos).

---

## Passo 6 — Calcular o R²

```python
def calcular_r2(y_real, y_pred):
    media_y = _media(y_real)
    ss_res = sum((y_real[i] - y_pred[i]) ** 2 for i in range(len(y_real)))
    ss_tot = sum((y_real[i] - media_y) ** 2 for i in range(len(y_real)))
    return 1.0 - (ss_res / ss_tot)
```

**O que cada parte significa:**

| Variável | Fórmula | Significado |
|---|---|---|
| `media_y` | `Σyᵢ / n` | Média dos pesos reais |
| `ss_res` | `Σ(yᵢ - ŷᵢ)²` | Soma dos erros ao quadrado do modelo |
| `ss_tot` | `Σ(yᵢ - ȳ)²` | Variância total dos dados |
| `R²` | `1 - ss_res/ss_tot` | Quanto o modelo explica da variação |

**Interpretação do R²:**
- `R² = 1.0` → modelo perfeito (explica 100%)
- `R² = 0.84` → modelo explica 84% da variação do peso ✅
- `R² = 0.0` → modelo não é melhor que prever sempre a média

---

## Passo 7 — Validar a Precisão

```python
def validar_precisao(r2, limiar=0.8):
    aprovado = r2 >= limiar
    status = "[OK] APROVADO" if aprovado else "[FALHOU] REPROVADO"
    return aprovado, status
```

Simplesmente compara o R² obtido no conjunto de **teste** com o limiar mínimo de **0.8**.

---

## Resultado Final

```
============================================================
  RESUMO FINAL
============================================================
  Amostras totais : 214
  Features usadas : 23
  Amostras treino : 149
  Amostras teste  : 65
  R2 Treino       : 0.7435
  R2 Teste        : 0.8391   ← explica 83.9% da variação do peso
  Limiar          : 0.8000
  Resultado       : [OK] APROVADO
============================================================
```

---

## Resumo das Funções

| Função | Módulo | Implementada manualmente? |
|---|---|---|
| `carregar_dados()` | 1 — Carregamento | ❌ usa pandas (só para leitura) |
| `_converter_br_para_float()` | 2 — Filtro | ✅ |
| `filtrar_xy()` | 2 — Filtro | ✅ |
| `_min_lista()` / `_max_lista()` | 3 — Normalização | ✅ |
| `normalizar_minmax()` | 3 — Normalização | ✅ |
| `_embaralhar_lcg()` | 4 — Divisão | ✅ (LCG + Fisher-Yates) |
| `dividir_treino_teste()` | 4 — Divisão | ✅ |
| `_adicionar_bias()` | 5 — Regressão | ✅ |
| `_transpor()` | 5 — Regressão | ✅ |
| `_mult_mat()` | 5 — Regressão | ✅ |
| `_inverter_gauss()` | 5 — Regressão | ✅ (Gauss-Jordan) |
| `_regularizar_XtX()` | 5 — Regressão | ✅ (Ridge) |
| `_mult_mat_vec()` | 5 — Regressão | ✅ |
| `regressao_linear()` | 5 — Regressão | ✅ (Equação Normal) |
| `prever()` | 5 — Regressão | ✅ |
| `_media()` | 6 — R² | ✅ |
| `calcular_r2()` | 6 — R² | ✅ |
| `validar_precisao()` | 7 — Validação | ✅ |
| `executar_pipeline()` | Pipeline | ✅ (orquestra tudo) |
