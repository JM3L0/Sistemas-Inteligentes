import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import random

class MapaVisibilidade:
    def __init__(self, largura, altura):
        self.largura = largura
        self.altura = altura
        self.obstaculos = [] # Lista de triângulos (cada um é uma lista de 3 vértices)

    def gerar_triangulo_equilatero(self, cx, cy, lado):
        """Calcula os 3 vértices de um triângulo equilátero dado o centro e o lado."""
        h = (lado * np.sqrt(3)) / 2
        
        v1 = (cx, cy + (2/3)*h)                # Topo
        v2 = (cx - lado/2, cy - (1/3)*h)       # Base Esquerda
        v3 = (cx + lado/2, cy - (1/3)*h)       # Base Direita
        
        return [v1, v2, v3]

    def adicionar_obstaculos_aleatorios(self, qtd, lado_triangulo):
        tentativas = 0
        while len(self.obstaculos) < qtd and tentativas < 100:
            # Sorteia um centro dentro dos limites do mapa (com margem)
            cx = random.uniform(lado_triangulo, self.largura - lado_triangulo)
            cy = random.uniform(lado_triangulo, self.altura - lado_triangulo)
            
            novo_triangulo = self.gerar_triangulo_equilatero(cx, cy, lado_triangulo)
            
            # Aqui poderíamos adicionar uma lógica de colisão para evitar sobreposição
            self.obstaculos.append(novo_triangulo)
            tentativas += 1

    def plotar_mapa(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.largura)
        ax.set_ylim(0, self.altura)
        ax.set_aspect('equal')
        
        for tri in self.obstaculos:
            # Fecha o polígono repetindo o primeiro vértice no final
            tri_fechado = tri + [tri[0]]
            xs, ys = zip(*tri_fechado)
            ax.fill(xs, ys, "red", alpha=0.6, edgecolor="black")
            
        plt.title(f"Mapa de Visibilidade: {len(self.obstaculos)} Obstáculos")
        plt.grid(True)
        plt.show()

# --- Exemplo de Uso ---
meu_mapa = MapaVisibilidade(largura=100, altura=100)
meu_mapa.adicionar_obstaculos_aleatorios(qtd=10, lado_triangulo=10)
meu_mapa.plotar_mapa()