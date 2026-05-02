# -*- coding: utf-8 -*-
"""
=============================================================
TRABALHO 6 - SISTEMAS INTELIGENTES
Regressao Linear Multipla - Dados de Imagem de Ultrassom
=============================================================
Objetivo : Prever o PESO do animal com base nas medidas
           morfometricas coletadas a partir da coluna AC.
Divisao  : 70% Treino / 30% Teste  |  Seed = 50
Meta     : R2 >= 0.8
=============================================================
"""

import sys
import io
import pandas as pd

# Garante saida UTF-8 no Windows (evita UnicodeEncodeError no terminal)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==============================================================
# MÓDULO 1 — CARREGAMENTO DE DADOS
# ==============================================================

def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Lê o arquivo CSV e retorna um DataFrame bruto.
    Pandas é usado apenas para I/O; todos os cálculos são manuais.
    """
    df = pd.read_csv(caminho, encoding="utf-8")
    print(f"[1] Dados carregados: {len(df)} registros, {len(df.columns)} colunas.")
    return df


# ==============================================================
# MÓDULO 2 — FILTRAR X e Y
# ==============================================================

def _converter_br_para_float(valor):
    """Converte string no formato brasileiro ('1,23') para float."""
    if isinstance(valor, str):
        return valor.replace(",", ".")
    return valor


def filtrar_xy(df: pd.DataFrame):
    """
    Define Y = PESO e X = medidas morfometricas a partir de AC
    mais medidas de ultrassom (AOL, EGE, MOL, EC) e IDADE,
    que sao parte do mesmo exame e essenciais para predicao de peso.

    Remove colunas com mais de 50% de valores ausentes e
    elimina as linhas restantes com NaN.
    Retorna X (lista de listas), y (lista) e nomes das features.
    """
    col_y = "PESO"

    # Features principais: medidas morfometricas (a partir de AC)
    # + medidas de ultrassom e biometricas do mesmo exame
    cols_candidatas_x = [
        # --- Medidas de ultrassom e biometricas do exame ---
        "IDADE",
        "AOL (cm\u00b2)", "COL (cm)", "POL (cm)", "RATIO (cm)",
        "EGE (mm)", "MOL", "EC",
        # --- Medidas morfometricas (a partir de AC) ---
        "AC", "AG", "CC", "AP", "P.C", "CT", "CO", "CCAB",
        "LR", "LCAB", "LIL", "LIS", "Cga", "Cper", "PerPe", "Ccau",
        "DC", "CE"
    ]

    # Selecionar colunas existentes no DataFrame
    cols_x = [c for c in cols_candidatas_x if c in df.columns]

    # Trabalhar apenas com as colunas necessárias
    subset = df[[col_y] + cols_x].copy()

    # Converter separador decimal BR → ponto
    for col in subset.columns:
        subset[col] = subset[col].apply(_converter_br_para_float)
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    # Remover colunas com mais de 50% de NaN
    limite_nan = 0.5 * len(subset)
    colunas_densas = [c for c in cols_x if subset[c].isna().sum() <= limite_nan]
    removidas = set(cols_x) - set(colunas_densas)
    if removidas:
        print(f"    Colunas removidas por excesso de NaN (>50%): {removidas}")
    cols_x = colunas_densas

    # Remover linhas com NaN remanescente
    subset = subset[[col_y] + cols_x].dropna()

    y = subset[col_y].tolist()
    X = subset[cols_x].values.tolist()

    return X, y, cols_x


# ==============================================================
# MÓDULO 3 — PRÉ-PROCESSAMENTO (Normalização Min-Max)
# ==============================================================

def _min_lista(lst: list) -> float:
    """Retorna o mínimo de uma lista."""
    minimo = lst[0]
    for v in lst:
        if v < minimo:
            minimo = v
    return minimo


def _max_lista(lst: list) -> float:
    """Retorna o máximo de uma lista."""
    maximo = lst[0]
    for v in lst:
        if v > maximo:
            maximo = v
    return maximo


def normalizar_minmax(X: list):
    """
    Aplica a normalização Min-Max em cada feature:
        x_norm = (x - min) / (max - min)
    Retorna X_normalizado, lista de mínimos e lista de máximos.
    """
    n_amostras = len(X)
    n_features = len(X[0])

    minimos, maximos = [], []
    for j in range(n_features):
        coluna = [X[i][j] for i in range(n_amostras)]
        minimos.append(_min_lista(coluna))
        maximos.append(_max_lista(coluna))

    X_norm = []
    for i in range(n_amostras):
        linha = []
        for j in range(n_features):
            denom = maximos[j] - minimos[j]
            if denom == 0.0:
                linha.append(0.0)
            else:
                linha.append((X[i][j] - minimos[j]) / denom)
        X_norm.append(linha)

    return X_norm, minimos, maximos


# ==============================================================
# MÓDULO 4 — DIVISÃO TREINO / TESTE (70/30, seed=50)
# ==============================================================

def _embaralhar_lcg(n: int, seed: int = 50) -> list:
    """
    Gera uma permutação aleatória de índices 0..n-1 usando um
    Gerador Congruencial Linear (LCG).
    Parâmetros de Knuth: a=1664525, c=1013904223, m=2^32.
    """
    a, c, m = 1_664_525, 1_013_904_223, 2 ** 32
    estado = seed
    indices = list(range(n))

    for i in range(n - 1, 0, -1):
        estado = (a * estado + c) % m
        j = estado % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]

    return indices


def dividir_treino_teste(X: list, y: list, razao_treino: float = 0.7, seed: int = 50):
    """
    Embaralha os dados com seed fixo e divide em treino/teste.
    """
    indices = _embaralhar_lcg(len(X), seed)
    X_emb = [X[i] for i in indices]
    y_emb = [y[i] for i in indices]

    n_treino = int(len(X) * razao_treino)

    X_treino = X_emb[:n_treino]
    y_treino = y_emb[:n_treino]
    X_teste  = X_emb[n_treino:]
    y_teste  = y_emb[n_treino:]

    print(f"[2] Divisão 70/30 (seed={seed}): Treino={len(y_treino)} | Teste={len(y_teste)}")
    return X_treino, y_treino, X_teste, y_teste


# ==============================================================
# MÓDULO 5 — REGRESSÃO LINEAR (Equação Normal)
# ==============================================================

def _adicionar_bias(X: list) -> list:
    """Adiciona coluna de 1s à esquerda (termo independente β₀)."""
    return [[1.0] + row[:] for row in X]


def _transpor(M: list) -> list:
    """Transpõe uma matriz M (lista de listas)."""
    n_lin, n_col = len(M), len(M[0])
    return [[M[i][j] for i in range(n_lin)] for j in range(n_col)]


def _mult_mat(A: list, B: list) -> list:
    """Multiplicação de matrizes A × B."""
    n, m, p = len(A), len(B[0]), len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            soma = 0.0
            for k in range(p):
                soma += A[i][k] * B[k][j]
            C[i][j] = soma
    return C


def _inverter_gauss(M: list) -> list:
    """
    Inverte a matriz M usando Eliminação de Gauss-Jordan
    com pivotamento parcial.
    """
    n = len(M)
    # Matriz aumentada [M | I]
    aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]

    for col in range(n):
        # Pivotamento parcial
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivo = aug[col][col]
        if abs(pivo) < 1e-12:
            raise ValueError("Matriz singular — não é possível inverter.")

        # Normalizar linha do pivô
        for j in range(2 * n):
            aug[col][j] /= pivo

        # Zerar demais linhas nessa coluna
        for row in range(n):
            if row != col:
                fator = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= fator * aug[col][j]

    return [aug[i][n:] for i in range(n)]


def _mult_mat_vec(M: list, v: list) -> list:
    """Multiplica matriz M por vetor v."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _regularizar_XtX(XtX: list, lam: float = 1e-4) -> list:
    """
    Adiciona regularizacao de Tikhonov (Ridge) a diagonal de XtX:
        XtX_reg = XtX + lambda * I
    Isso evita singularidade quando features sao altamente correlacionadas.
    """
    n = len(XtX)
    XtX_reg = [row[:] for row in XtX]   # copia profunda
    for i in range(n):
        XtX_reg[i][i] += lam
    return XtX_reg


def regressao_linear(X_treino: list, y_treino: list, lam: float = 1e-4) -> list:
    """
    Calcula os coeficientes beta da regressao linear multipla pela
    Equacao Normal com regularizacao Ridge:
        beta = (XtX + lambda*I)^-1 * Xty
    Retorna o vetor beta (bias + pesos das features).
    """
    X_b  = _adicionar_bias(X_treino)          # n x (p+1)
    Xt   = _transpor(X_b)                      # (p+1) x n
    XtX  = _mult_mat(Xt, X_b)                 # (p+1) x (p+1)
    XtX_reg = _regularizar_XtX(XtX, lam)      # regularizacao Ridge
    XtXi = _inverter_gauss(XtX_reg)           # inversa por Gauss-Jordan
    Xty  = _mult_mat_vec(Xt, y_treino)        # (p+1)
    beta = _mult_mat_vec(XtXi, Xty)           # (p+1)
    return beta


def prever(X: list, beta: list) -> list:
    """
    Gera predições ŷ = Xβ para um conjunto de amostras X.
    """
    X_b = _adicionar_bias(X)
    return [sum(X_b[i][j] * beta[j] for j in range(len(beta)))
            for i in range(len(X_b))]


# ==============================================================
# MÓDULO 6 — CÁLCULO DO R²
# ==============================================================

def _media(valores: list) -> float:
    """Calcula a média aritmética."""
    return sum(valores) / len(valores)


def calcular_r2(y_real: list, y_pred: list) -> float:
    """
    Calcula o Coeficiente de Determinacao R2:
        SS_res = soma( (yi - yi_pred)^2 )
        SS_tot = soma( (yi - media_y)^2 )
        R2     = 1 - SS_res / SS_tot
    """
    media_y = _media(y_real)
    ss_res = sum((y_real[i] - y_pred[i]) ** 2 for i in range(len(y_real)))
    ss_tot = sum((y_real[i] - media_y) ** 2 for i in range(len(y_real)))
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


# ==============================================================
# MÓDULO 7 — VALIDAÇÃO DA PRECISÃO
# ==============================================================

def validar_precisao(r2: float, limiar: float = 0.8) -> tuple:
    """
    Verifica se o R2 atinge o limiar minimo de precisao.
    Retorna (bool aprovado, str status).
    """
    aprovado = r2 >= limiar
    status = "[OK] APROVADO" if aprovado else "[FALHOU] REPROVADO"
    return aprovado, status


# ==============================================================
# PIPELINE PRINCIPAL
# ==============================================================

def executar_pipeline(caminho_csv: str) -> dict:
    """
    Orquestra todas as etapas do pipeline de regressao.
    """
    linha = "=" * 60

    # 1. Carregar
    df = carregar_dados(caminho_csv)

    # 2. Filtrar
    X, y, cols_x = filtrar_xy(df)

    # 3. Pré-processamento
    X_norm, mins, maxs = normalizar_minmax(X)

    # 4. Divisão
    X_tr, y_tr, X_te, y_te = dividir_treino_teste(X_norm, y, 0.7, seed=50)

    # 5. Regressão
    beta = regressao_linear(X_tr, y_tr)

    # 6. R²
    y_pred_te = prever(X_te, beta)
    y_pred_tr = prever(X_tr, beta)
    r2_te = calcular_r2(y_te, y_pred_te)
    r2_tr = calcular_r2(y_tr, y_pred_tr)
    print(f"[3] R2 Treino: {r2_tr:.4f}  |  R2 Teste: {r2_te:.4f}")

    # 7. Validacao
    aprovado, status = validar_precisao(r2_te)

    # Resumo
    print(f"\n{linha}")
    print("  RESUMO FINAL")
    print(f"{linha}")
    print(f"  Amostras totais : {len(y)}")
    print(f"  Amostras treino : {len(y_tr)}")
    print(f"  Amostras teste  : {len(y_te)}")
    print(f"  R2 Treino       : {r2_tr:.4f}")
    print(f"  R2 Teste        : {r2_te:.4f}")
    print(f"  Limiar          : 0.8000")
    print(f"  Resultado       : {status}")
    print(f"{linha}\n")

    # Mostrar primeiras 10 predicoes vs real
    print("  Predicoes vs Valores Reais (primeiros 10 do teste):")
    print(f"  {'Real':>10} | {'Predito':>10} | {'Erro':>10}")
    print("  " + "-" * 36)
    for i in range(min(10, len(y_te))):
        erro = abs(y_te[i] - y_pred_te[i])
        print(f"  {y_te[i]:>10.2f} | {y_pred_te[i]:>10.2f} | {erro:>10.2f}")

    return {
        "beta": beta,
        "r2_treino": r2_tr,
        "r2_teste": r2_te,
        "aprovado": aprovado,
        "y_teste": y_te,
        "y_pred_teste": y_pred_te,
        "features": cols_x,
    }


# ==============================================================
# ENTRADA
# ==============================================================

if __name__ == "__main__":
    import os

    # Ajusta o caminho para rodar de qualquer diretório
    base = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(base, "Dados_Projeto_Imagem_Ultrassom.xlsx.csv")

    resultados = executar_pipeline(caminho)
