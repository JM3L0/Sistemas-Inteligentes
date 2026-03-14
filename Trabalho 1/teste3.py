import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import random
from dataclasses import dataclass

# ===== PARÂMETROS DO MAPA =====
LARGURA = 100               # Largura do mapa (eixo X)
ALTURA = 50                 # Altura do mapa (eixo Y)
QUANTIDADE_OBSTACULOS = 33  # Quantos triângulos tentar inserir
LADO_TRIANGULO = 10         # Tamanho do lado de cada triângulo equilátero

EPS = 1e-9  # Margem de tolerância para comparações com ponto flutuante


# ===== ESTRUTURA DE DADOS =====

@dataclass
class Triangulo:
    """Representa um triângulo com 3 vértices (tuplas x, y)."""
    v1: tuple
    v2: tuple
    v3: tuple

    def vertices(self):
        """Retorna os 3 vértices como lista."""
        return [self.v1, self.v2, self.v3]

    def arestas(self):
        """Retorna as 3 arestas como pares de vértices: [(v1,v2), (v2,v3), (v3,v1)]."""
        vs = self.vertices()
        return [(vs[i], vs[(i + 1) % 3]) for i in range(3)]


# ===== CLASSE PRINCIPAL =====

class MapaVisibilidade:

    def __init__(self, largura, altura):
        self.largura = largura
        self.altura = altura
        self.obstaculos = []
        self.quant_colisoes = 0
        self.quant_inseridos = 0

    # =========================================================
    # GEOMETRIA COMPUTACIONAL (Funções base de cálculo)
    # =========================================================

    def orientacao(self, A, B, C):
        """
        Calcula o produto vetorial 2D dos vetores AB e AC.
        Retorna:
          > 0  →  C está à ESQUERDA da reta A→B
          < 0  →  C está à DIREITA da reta A→B
          = 0  →  A, B e C são colineares (na mesma reta)
        """
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

    def ponto_dentro_triangulo(self, P, tri):
        """
        Verifica se o ponto P está dentro do triângulo usando
        o método dos semi-planos (3 testes de orientação).
        Se P estiver sempre do mesmo lado das 3 arestas, está dentro.
        """
        A, B, C = tri.vertices()

        d1 = self.orientacao(A, B, P)
        d2 = self.orientacao(B, C, P)
        d3 = self.orientacao(C, A, P)

        tem_negativo = (d1 < -EPS) or (d2 < -EPS) or (d3 < -EPS)
        tem_positivo = (d1 > EPS) or (d2 > EPS) or (d3 > EPS)

        # Se tem sinais misturados, P está fora do triângulo
        return not (tem_negativo and tem_positivo)

    def ponto_no_segmento(self, A, B, P):
        """
        Verifica se o ponto P (já sabido colinear com A e B)
        está dentro do retângulo delimitado pelo segmento AB.
        """
        return (
            min(A[0], B[0]) <= P[0] <= max(A[0], B[0]) and
            min(A[1], B[1]) <= P[1] <= max(A[1], B[1])
        )

    def segmentos_cruzam(self, A, B, C, D):
        """
        Verifica se o segmento AB cruza o segmento CD.
        Usa orientação para determinar se os pontos estão
        em lados opostos de cada reta.
        """
        d1 = self.orientacao(A, B, C)
        d2 = self.orientacao(A, B, D)
        d3 = self.orientacao(C, D, A)
        d4 = self.orientacao(C, D, B)

        # Caso geral: pontos em lados opostos (sinais trocados)
        if (d1 * d2 < 0) and (d3 * d4 < 0):
            return True

        # Casos especiais: ponto colinear encostando no segmento
        if abs(d1) < EPS and self.ponto_no_segmento(A, B, C):
            return True
        if abs(d2) < EPS and self.ponto_no_segmento(A, B, D):
            return True
        if abs(d3) < EPS and self.ponto_no_segmento(C, D, A):
            return True
        if abs(d4) < EPS and self.ponto_no_segmento(C, D, B):
            return True

        return False

    # =========================================================
    # DETECÇÃO DE COLISÃO ENTRE TRIÂNGULOS
    # =========================================================

    def triangulos_colidem(self, tri1, tri2):
        """
        Verifica colisão real entre dois triângulos em 3 etapas:
        1. Algum vértice de tri1 está dentro de tri2?
        2. Algum vértice de tri2 está dentro de tri1?
        3. Alguma aresta de tri1 cruza alguma aresta de tri2?
        """
        # Etapa 1: vértices de tri1 dentro de tri2
        for vertice in tri1.vertices():
            if self.ponto_dentro_triangulo(vertice, tri2):
                return True

        # Etapa 2: vértices de tri2 dentro de tri1
        for vertice in tri2.vertices():
            if self.ponto_dentro_triangulo(vertice, tri1):
                return True

        # Etapa 3: cruzamento de arestas (caso "Estrela de Davi")
        for a1, b1 in tri1.arestas():
            for a2, b2 in tri2.arestas():
                if self.segmentos_cruzam(a1, b1, a2, b2):
                    return True

        return False

    # =========================================================
    # BOUNDING BOX (Filtro rápido antes da colisão real)
    # =========================================================

    def bounding_box(self, tri):
        """Retorna o menor retângulo que envolve o triângulo: (minX, maxX, minY, maxY)."""
        xs = [v[0] for v in tri.vertices()]
        ys = [v[1] for v in tri.vertices()]
        return min(xs), max(xs), min(ys), max(ys)

    def bbox_colidem(self, tri1, tri2):
        """
        Verifica se as caixas envolventes (bounding boxes) se sobrepõem.
        Se NÃO se sobrepõem, é impossível os triângulos colidirem.
        """
        minx1, maxx1, miny1, maxy1 = self.bounding_box(tri1)
        minx2, maxx2, miny2, maxy2 = self.bounding_box(tri2)

        # Se qualquer condição for verdadeira, estão separados
        return not (
            maxx1 < minx2 or  # tri1 totalmente à esquerda de tri2
            maxx2 < minx1 or  # tri2 totalmente à esquerda de tri1
            maxy1 < miny2 or  # tri1 totalmente abaixo de tri2
            maxy2 < miny1     # tri2 totalmente abaixo de tri1
        )

    # =========================================================
    # GERAÇÃO DE TRIÂNGULOS EQUILÁTEROS
    # =========================================================

    def gerar_triangulo(self, cx, cy, lado):
        """
        Cria um triângulo equilátero centrado em (cx, cy).
        A ponta aponta para cima (eixo Y positivo).
        """
        altura = (lado * np.sqrt(3)) / 2  # Altura do triângulo equilátero

        ponta_superior = (cx, cy + (2 / 3) * altura)
        base_esquerda  = (cx - lado / 2, cy - altura / 3)
        base_direita   = (cx + lado / 2, cy - altura / 3)

        return Triangulo(ponta_superior, base_esquerda, base_direita)

    # =========================================================
    # INSERÇÃO DE OBSTÁCULOS ALEATÓRIOS
    # =========================================================

    def _colide_com_algum_obstaculo(self, novo):
        """Verifica se o novo triângulo colide com qualquer obstáculo existente."""
        for obstaculo in self.obstaculos:
            # Filtro rápido: se as caixas não se tocam, pula
            if not self.bbox_colidem(novo, obstaculo):
                continue
            # Filtro preciso: verifica colisão real dos triângulos
            if self.triangulos_colidem(novo, obstaculo):
                self.quant_colisoes += 1
                return True
        return False

    def adicionar_obstaculos_aleatorios(self, qtd, lado):
        """
        Tenta inserir 'qtd' triângulos aleatórios no mapa.
        Para cada triângulo, sorteia posições até encontrar
        uma que não colida com nenhum obstáculo já existente.
        """
        margem_x = lado / 2          # Extensão horizontal do triângulo (metade da base)
        margem_y = lado / np.sqrt(3)  # Extensão vertical do triângulo (distância do centro ao topo)

        for _ in range(qtd):
            for tentativa in range(self.largura * 2):
                cx = random.uniform(margem_x, self.largura - margem_x)
                cy = random.uniform(margem_y, self.altura - margem_y)
                novo = self.gerar_triangulo(cx, cy, lado)

                if not self._colide_com_algum_obstaculo(novo):
                    self.obstaculos.append(novo)
                    self.quant_inseridos += 1
                    break

    # =========================================================
    # PLOTAGEM DO MAPA
    # =========================================================

    def plotar_mapa(self):
        """Desenha o mapa com todos os obstáculos e os pontos de início/fim."""
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.set_xlim(0, self.largura)
        ax.set_ylim(0, self.altura)
        ax.set_aspect('equal')

        # Desenha cada triângulo obstáculo
        for tri in self.obstaculos:
            vs = tri.vertices()
            xs, ys = zip(*(vs + [vs[0]]))
            ax.fill(xs, ys, color="red", alpha=0.5, edgecolor="black")

        # Pontos de referência
        ax.plot(0, 0, 'bs', label="Início")
        ax.plot(self.largura, self.altura, 'gs', label="Fim")

        plt.title(f"Obstáculos: {len(self.obstaculos)}  |  Colisões: {self.quant_colisoes}")
        plt.legend()
        plt.grid()
        plt.show()


# ===== EXECUÇÃO =====

mapa = MapaVisibilidade(LARGURA, ALTURA)

mapa.adicionar_obstaculos_aleatorios(
    QUANTIDADE_OBSTACULOS,
    LADO_TRIANGULO
)

mapa.plotar_mapa()

print(f"Colisões detectadas: {mapa.quant_colisoes}")
print(f"Obstáculos inseridos: {mapa.quant_inseridos}")
