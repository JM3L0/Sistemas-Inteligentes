import pandas as pd
<<<<<<< HEAD

salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')

salary.columns

y = salary['Salary']
X = salary[['Years_Experience']]

from sklearn.linear_model import train_test_split
=======
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Regressão Linear Simples com Descida do Gradiente
# Dataset: mt_cars.csv
# Objetivo: Prever mpg (milhas por galão) a partir de wt (peso do carro)
# =============================================================================

# --- 1. Carregar os dados ---
df = pd.read_csv("mt_cars.csv")
print("=" * 60)
print("REGRESSÃO LINEAR SIMPLES - DESCIDA DO GRADIENTE")
print("=" * 60)
print(f"\nDataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
print(f"\nPrimeiras 5 linhas:")
print(df.head())

# --- 2. Selecionar variáveis ---
# x = peso (wt), y = consumo (mpg)
x = df["wt"].values
y = df["mpg"].values

n = len(x)  # número de amostras
print(f"\nVariável independente (X): wt (peso)")
print(f"Variável dependente (Y): mpg (milhas por galão)")
print(f"Número de amostras: {n}")

# --- 3. Normalização (Feature Scaling) ---
# Normalizar x para acelerar a convergência do gradiente
x_mean = np.mean(x)
x_std = np.std(x)
x_norm = (x - x_mean) / x_std

y_mean = np.mean(y)
y_std = np.std(y)
y_norm = (y - y_mean) / y_std

print(f"\nMédia de X: {x_mean:.4f}, Desvio padrão de X: {x_std:.4f}")
print(f"Média de Y: {y_mean:.4f}, Desvio padrão de Y: {y_std:.4f}")

# --- 4. Inicializar parâmetros ---
# y = w1 * x + w0 (w0 = intercepto/bias, w1 = coeficiente angular)
w0 = 0.0  # intercepto (bias)
w1 = 0.0  # coeficiente angular (slope)

# Hiperparâmetros
taxa_aprendizado = 0.01  # learning rate (alpha)
num_iteracoes = 1000     # número de épocas/iterações

print(f"\n--- Hiperparâmetros ---")
print(f"Taxa de aprendizado (alfa): {taxa_aprendizado}")
print(f"Número de iterações: {num_iteracoes}")
print(f"Pesos iniciais: w0 = {w0}, w1 = {w1}")

# --- 5. Função de Custo (MSE - Mean Squared Error) ---
def calcular_custo(x, y, w0, w1):
    """Calcula o erro quadrático médio (MSE)."""
    n = len(x)
    y_pred = w0 + w1 * x
    custo = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
    return custo

# --- 6. Descida do Gradiente ---
historico_custo = []

print(f"\n--- Executando Descida do Gradiente ---")

for i in range(num_iteracoes):
    # Previsão atual
    y_pred = w0 + w1 * x_norm

    # Calcular os gradientes (derivadas parciais)
    # dJ/dw0 = (1/n) * Σ(y_pred - y)
    # dJ/dw1 = (1/n) * Σ((y_pred - y) * x)
    erro = y_pred - y_norm
    grad_w0 = (1 / n) * np.sum(erro)
    grad_w1 = (1 / n) * np.sum(erro * x_norm)

    # Atualizar os pesos
    w0 = w0 - taxa_aprendizado * grad_w0
    w1 = w1 - taxa_aprendizado * grad_w1

    # Calcular e armazenar o custo
    custo = calcular_custo(x_norm, y_norm, w0, w1)
    historico_custo.append(custo)

    # Exibir progresso a cada 100 iterações
    if (i + 1) % 100 == 0 or i == 0:
        print(f"  Iteração {i+1:4d} | Custo: {custo:.6f} | w0: {w1:.6f} | w1: {w1:.6f}")

# --- 7. Desnormalizar os pesos para escala original ---
# y_norm = w0 + w1 * x_norm
# (y - y_mean)/y_std = w0 + w1 * (x - x_mean)/x_std
# y = y_std * w0 + y_std * w1 * (x - x_mean)/x_std + y_mean
# y = (y_std * w1 / x_std) * x + (y_mean + y_std * w0 - y_std * w1 * x_mean / x_std)
w1_original = y_std * w1 / x_std
w0_original = y_mean + y_std * w0 - y_std * w1 * x_mean / x_std

print(f"\n--- Resultado Final ---")
print(f"Pesos normalizados:  w0 = {w0:.6f}, w1 = {w1:.6f}")
print(f"Pesos originais:     w0 = {w0_original:.6f}, w1 = {w1_original:.6f}")
print(f"Equação da reta: mpg = {w0_original:.4f} + ({w1_original:.4f}) * wt")
print(f"Custo final (MSE normalizado): {historico_custo[-1]:.6f}")

# --- 8. Métricas de avaliação ---
y_pred_final = w0_original + w1_original * x
residuos = y - y_pred_final
mse = np.mean(residuos ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuos))

# R² (coeficiente de determinação)
ss_res = np.sum(residuos ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"\n--- Métricas de Avaliação ---")
print(f"MSE  (Erro Quadrático Médio):   {mse:.4f}")
print(f"RMSE (Raiz do MSE):             {rmse:.4f}")
print(f"MAE  (Erro Absoluto Médio):     {mae:.4f}")
print(f"R²   (Coeficiente de Determinação): {r2:.4f}")

# --- 9. Visualizações ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Regressão Linear Simples - Descida do Gradiente", fontsize=14, fontweight="bold")

# Gráfico 1: Dados + Reta de regressão
axes[0].scatter(x, y, color="steelblue", edgecolors="navy", alpha=0.7, s=60, label="Dados reais")
x_line = np.linspace(x.min(), x.max(), 100)
y_line = w0_original + w1_original * x_line
axes[0].plot(x_line, y_line, color="red", linewidth=2, label=f"Reta: y = {w0_original:.2f} + ({w1_original:.2f})x")
axes[0].set_xlabel("Peso (wt)", fontsize=11)
axes[0].set_ylabel("Consumo (mpg)", fontsize=11)
axes[0].set_title("Dados e Reta de Regressão")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Curva de custo (convergência)
axes[1].plot(range(1, num_iteracoes + 1), historico_custo, color="green", linewidth=1.5)
axes[1].set_xlabel("Iteração", fontsize=11)
axes[1].set_ylabel("Custo (MSE)", fontsize=11)
axes[1].set_title("Convergência do Custo")
axes[1].grid(True, alpha=0.3)

# Gráfico 3: Resíduos
axes[2].scatter(y_pred_final, residuos, color="coral", edgecolors="darkred", alpha=0.7, s=60)
axes[2].axhline(y=0, color="black", linewidth=1, linestyle="--")
axes[2].set_xlabel("Valores Preditos (mpg)", fontsize=11)
axes[2].set_ylabel("Resíduos", fontsize=11)
axes[2].set_title("Gráfico de Resíduos")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("regressao_simples_resultado.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nGráfico salvo como 'regressao_simples_resultado.png'")
>>>>>>> 28d0cb93690f90e1c2360906490d89797f9a86f9
