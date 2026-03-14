import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import random

class MapaVisibilidade:
    def __init__(self, largura, altura):
        self.largura = largura
        self.altura = altura
        self.obstaculos = [] # Lista de triângulos (cada um é uma lista de 3 vértices)

    # ========== MÉTODOS DE COLISÃO COM DETERMINANTES ==========

    def determinante(self, A, B, P):
        """
        Calcula o determinante (produto vetorial 2D) dos vetores AB e AP.
        Retorna:
          > 0 se P está à esquerda da aresta AB
          < 0 se P está à direita da aresta AB
          = 0 se P é colinear com AB
        
        Fórmula: det = (Bx - Ax)*(Py - Ay) - (By - Ay)*(Px - Ax)
        """
        return (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])

    def ponto_dentro_triangulo(self, P, triangulo):
        """
        Verifica se o ponto P está dentro do triângulo usando determinantes.
        P está dentro se os 3 determinantes (um para cada aresta) têm o MESMO sinal.
        """
        A, B, C = triangulo

        d1 = self.determinante(A, B, P)
        d2 = self.determinante(B, C, P)
        d3 = self.determinante(C, A, P)

        tem_negativo = (d1 < 0) or (d2 < 0) or (d3 < 0)
        tem_positivo = (d1 > 0) or (d2 > 0) or (d3 > 0)

        # Se todos têm o mesmo sinal (todos positivos OU todos negativos), está dentro
        return not (tem_negativo and tem_positivo)

    def segmentos_se_cruzam(self, A, B, C, D):
        """
        Verifica se o segmento AB cruza o segmento CD usando determinantes.
        Dois segmentos se cruzam se:
          - C e D estão em lados OPOSTOS de AB (determinantes com sinais diferentes)
          - A e B estão em lados OPOSTOS de CD (determinantes com sinais diferentes)
        """
        d1 = self.determinante(A, B, C)
        d2 = self.determinante(A, B, D)
        d3 = self.determinante(C, D, A)
        d4 = self.determinante(C, D, B)

        # Caso geral: sinais opostos em ambos os testes
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        # Casos especiais: ponto colinear sobre o segmento
        if d1 == 0 and self._no_segmento(A, B, C):
            return True
        if d2 == 0 and self._no_segmento(A, B, D):
            return True
        if d3 == 0 and self._no_segmento(C, D, A):
            return True
        if d4 == 0 and self._no_segmento(C, D, B):
            return True

        return False

    def _no_segmento(self, A, B, P):
        """Verifica se P está sobre o segmento AB (quando já sabemos que são colineares)."""
        return (min(A[0], B[0]) <= P[0] <= max(A[0], B[0]) and
                min(A[1], B[1]) <= P[1] <= max(A[1], B[1]))

    def triangulos_colidem(self, tri1, tri2):
        """
        Verifica se dois triângulos colidem usando determinantes.
        Colisão acontece se:
          1) Algum vértice de tri1 está dentro de tri2, OU
          2) Algum vértice de tri2 está dentro de tri1, OU
          3) Alguma aresta de tri1 cruza alguma aresta de tri2
        """
        # Teste 1: vértices de tri1 dentro de tri2
        for v in tri1:
            if self.ponto_dentro_triangulo(v, tri2):
                return True

        # Teste 2: vértices de tri2 dentro de tri1
        for v in tri2:
            if self.ponto_dentro_triangulo(v, tri1):
                return True

        # Teste 3: arestas se cruzam
        arestas1 = [(tri1[i], tri1[(i + 1) % 3]) for i in range(3)]
        arestas2 = [(tri2[i], tri2[(i + 1) % 3]) for i in range(3)]

        for (a1, b1) in arestas1:
            for (a2, b2) in arestas2:
                if self.segmentos_se_cruzam(a1, b1, a2, b2):
                    return True

        return False

    # ========== GERAÇÃO DO MAPA ==========

    def gerar_triangulo_equilatero(self, cx, cy, lado):
        """Calcula os 3 vértices de um triângulo equilátero dado o centro e o lado."""
        h = (lado * np.sqrt(3)) / 2
        
        v1 = (cx, cy + (2/3)*h)                # Topo
        v2 = (cx - lado/2, cy - (1/3)*h)       # Base Esquerda
        v3 = (cx + lado/2, cy - (1/3)*h)       # Base Direita
        
        return [v1, v2, v3]

    def adicionar_obstaculos_aleatorios(self, qtd, lado_triangulo):
        for i in range(qtd):
            while True:
                # Sorteia um centro dentro dos limites do mapa (com margem)
                cx = random.uniform(lado_triangulo, self.largura - lado_triangulo)
                cy = random.uniform(lado_triangulo, self.altura - lado_triangulo)
                
                novo_triangulo = self.gerar_triangulo_equilatero(cx, cy, lado_triangulo)
                
                # Verifica colisão com todos os obstáculos existentes usando determinantes
                colidiu = False
                for obstaculo in self.obstaculos:
                    if self.triangulos_colidem(novo_triangulo, obstaculo):
                        colidiu = True
                        break
                
                if not colidiu:
                    self.obstaculos.append(novo_triangulo)
                    break  # Achou lugar sem colisão, vai pro próximo triângulo

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
meu_mapa.adicionar_obstaculos_aleatorios(qtd=60, lado_triangulo=10)
meu_mapa.plotar_mapa()