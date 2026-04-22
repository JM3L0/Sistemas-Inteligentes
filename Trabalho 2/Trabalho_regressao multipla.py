import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Regressão Linear Múltipla com Descida do Gradiente
# Dataset: mt_cars.csv
# Objetivo: Prever mpg a partir de wt, hp e disp
# =============================================================================

# --- 1. Carregar os dados ---
df = pd.read_csv("mt_cars.csv")
print("=" * 60)
print("REGRESSÃO LINEAR MÚLTIPLA - DESCIDA DO GRADIENTE")
print("=" * 60)
print(f"\nDataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
print(f"\nPrimeiras 5 linhas:")
print(df.head())

# --- 2. Selecionar variáveis ---
# Features: wt (peso), hp (cavalos), disp (cilindrada)
# Target: mpg (milhas por galão)
features = ["wt", "hp", "disp"]
X = df[features].values  # matriz (n, p)
y = df["mpg"].values     # vetor (n,)

n, p = X.shape  # n = amostras, p = features
print(f"\nVariáveis independentes (X): {features}")
print(f"Variável dependente (Y): mpg")
print(f"Número de amostras: {n}")
print(f"Número de features: {p}")

# --- 3. Normalização (Feature Scaling) ---
# Normalizar cada feature (Z-score) para acelerar a convergência
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y_norm = (y - y_mean) / y_std

print(f"\n--- Estatísticas das Features ---")
for i, feat in enumerate(features):
    print(f"  {feat}: média = {X_mean[i]:.4f}, desvio = {X_std[i]:.4f}")
print(f"  mpg: média = {y_mean:.4f}, desvio = {y_std:.4f}")

# --- 4. Adicionar coluna de 1s para o intercepto (bias) ---
# X_norm agora tem shape (n, p+1): [1, x1, x2, ..., xp]
X_norm_bias = np.column_stack([np.ones(n), X_norm])

# Inicializar vetor de pesos: [w0, w1, w2, ..., wp]
W = np.zeros(p + 1)

# Hiperparâmetros
taxa_aprendizado = 0.01  # learning rate (alpha)
num_iteracoes = 1000     # número de épocas

print(f"\n--- Hiperparâmetros ---")
print(f"Taxa de aprendizado (alfa): {taxa_aprendizado}")
print(f"Número de iterações: {num_iteracoes}")
print(f"Pesos iniciais: {W}")

# --- 5. Função de Custo (MSE) ---
def calcular_custo(X, y, W):
    """Calcula o erro quadrático médio (MSE)."""
    n = len(y)
    y_pred = X @ W  # produto matricial
    custo = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
    return custo

# --- 6. Descida do Gradiente (forma vetorizada) ---
historico_custo = []

print(f"\n--- Executando Descida do Gradiente ---")

for i in range(num_iteracoes):
    # Previsão atual: ŷ = X · W
    y_pred = X_norm_bias @ W

    # Calcular o gradiente: ∇J = (1/n) * Xᵀ · (ŷ - y)
    erro = y_pred - y_norm
    gradiente = (1 / n) * (X_norm_bias.T @ erro)

    # Atualizar os pesos: W = W - α * ∇J
    W = W - taxa_aprendizado * gradiente

    # Calcular e armazenar o custo
    custo = calcular_custo(X_norm_bias, y_norm, W)
    historico_custo.append(custo)

    # Exibir progresso a cada 100 iterações
    if (i + 1) % 100 == 0 or i == 0:
        pesos_str = ", ".join([f"w{j}={W[j]:.4f}" for j in range(len(W))])
        print(f"  Iteração {i+1:4d} | Custo: {custo:.6f} | {pesos_str}")

# --- 7. Desnormalizar os pesos para escala original ---
# y_norm = W[0] + W[1]*x1_norm + W[2]*x2_norm + ...
# (y - y_mean)/y_std = W[0] + Σ Wi * (xi - xi_mean)/xi_std
# y = y_mean + y_std*W[0] + Σ (y_std*Wi/xi_std) * xi - Σ (y_std*Wi*xi_mean/xi_std)
W_original = np.zeros(p + 1)
for j in range(p):
    W_original[j + 1] = y_std * W[j + 1] / X_std[j]

W_original[0] = y_mean + y_std * W[0] - np.sum(y_std * W[1:] * X_mean / X_std)

print(f"\n--- Resultado Final ---")
print(f"Pesos normalizados:")
for j in range(len(W)):
    nome = "bias" if j == 0 else features[j - 1]
    print(f"  w_{nome} = {W[j]:.6f}")

print(f"\nPesos na escala original:")
for j in range(len(W_original)):
    nome = "intercepto" if j == 0 else features[j - 1]
    print(f"  w_{nome} = {W_original[j]:.6f}")

# Montar equação
termos = [f"{W_original[0]:.4f}"]
for j in range(p):
    sinal = "+" if W_original[j + 1] >= 0 else "-"
    termos.append(f"{sinal} {abs(W_original[j + 1]):.4f} * {features[j]}")
equacao = "mpg = " + " ".join(termos)
print(f"\nEquação: {equacao}")
print(f"Custo final (MSE normalizado): {historico_custo[-1]:.6f}")

# --- 8. Métricas de avaliação ---
y_pred_final = X @ W_original[1:] + W_original[0]
residuos = y - y_pred_final
mse = np.mean(residuos ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuos))

# R² (coeficiente de determinação)
ss_res = np.sum(residuos ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# R² ajustado
r2_ajustado = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

print(f"\n--- Métricas de Avaliação ---")
print(f"MSE       (Erro Quadrático Médio):       {mse:.4f}")
print(f"RMSE      (Raiz do MSE):                 {rmse:.4f}")
print(f"MAE       (Erro Absoluto Médio):         {mae:.4f}")
print(f"R²        (Coeficiente de Determinação): {r2:.4f}")
print(f"R² Ajust. (R² Ajustado):                 {r2_ajustado:.4f}")

# --- 9. Visualizações ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Regressão Linear Múltipla - Descida do Gradiente", fontsize=14, fontweight="bold")

# Gráfico 1: Convergência do custo
axes[0, 0].plot(range(1, num_iteracoes + 1), historico_custo, color="green", linewidth=1.5)
axes[0, 0].set_xlabel("Iteração", fontsize=11)
axes[0, 0].set_ylabel("Custo (MSE)", fontsize=11)
axes[0, 0].set_title("Convergência do Custo")
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Valores reais vs preditos
axes[0, 1].scatter(y, y_pred_final, color="steelblue", edgecolors="navy", alpha=0.7, s=60)
min_val = min(y.min(), y_pred_final.min())
max_val = max(y.max(), y_pred_final.max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], color="red", linewidth=2,
                linestyle="--", label="Linha ideal (y = ŷ)")
axes[0, 1].set_xlabel("Valores Reais (mpg)", fontsize=11)
axes[0, 1].set_ylabel("Valores Preditos (mpg)", fontsize=11)
axes[0, 1].set_title("Real vs Predito")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Resíduos
axes[1, 0].scatter(y_pred_final, residuos, color="coral", edgecolors="darkred", alpha=0.7, s=60)
axes[1, 0].axhline(y=0, color="black", linewidth=1, linestyle="--")
axes[1, 0].set_xlabel("Valores Preditos (mpg)", fontsize=11)
axes[1, 0].set_ylabel("Resíduos", fontsize=11)
axes[1, 0].set_title("Gráfico de Resíduos")
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Importância dos pesos (coeficientes normalizados)
cores = ["steelblue", "coral", "seagreen"]
bars = axes[1, 1].bar(features, W[1:], color=cores, edgecolor="navy", alpha=0.8)
axes[1, 1].set_xlabel("Feature", fontsize=11)
axes[1, 1].set_ylabel("Peso (normalizado)", fontsize=11)
axes[1, 1].set_title("Importância das Features (Pesos Normalizados)")
axes[1, 1].grid(True, alpha=0.3, axis="y")
# Adicionar rótulos nas barras
for bar, val in zip(bars, W[1:]):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("regressao_multipla_resultado.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nGráfico salvo como 'regressao_multipla_resultado.png'")
