import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import random
from dataclasses import dataclass

# ======== PARÂMETROS ========
largura = 100
altura = 50
quantidade_obstaculos = 30
lado_triangulo = 10

EPS = 1e-9

quant_colisoes = 0
quant_inseridos = 0

@dataclass
class Triangulo:
    v1: tuple
    v2: tuple
    v3: tuple

    def vertices(self):
        return [self.v1, self.v2, self.v3]


class MapaVisibilidade:

    def __init__(self, largura, altura):
        self.largura = largura
        self.altura = altura
        self.obstaculos = []

    # =============================
    # GEOMETRIA BÁSICA
    # =============================

    def determinante(self, A, B, P):
        return (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])

    def ponto_dentro_triangulo(self, P, triangulo):

        A, B, C = triangulo.vertices()

        d1 = self.determinante(A, B, P)
        d2 = self.determinante(B, C, P)
        d3 = self.determinante(C, A, P)

        tem_neg = (d1 < -EPS) or (d2 < -EPS) or (d3 < -EPS)
        tem_pos = (d1 > EPS) or (d2 > EPS) or (d3 > EPS)

        return not (tem_neg and tem_pos)

    def no_segmento(self, A, B, P):

        return (
            min(A[0], B[0]) <= P[0] <= max(A[0], B[0]) and
            min(A[1], B[1]) <= P[1] <= max(A[1], B[1])
        )

    def segmentos_se_cruzam(self, A, B, C, D):

        d1 = self.determinante(A, B, C)
        d2 = self.determinante(A, B, D)
        d3 = self.determinante(C, D, A)
        d4 = self.determinante(C, D, B)

        if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and \
           ((d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)):
            return True

        if abs(d1) < EPS and self.no_segmento(A, B, C):
            return True
        if abs(d2) < EPS and self.no_segmento(A, B, D):
            return True
        if abs(d3) < EPS and self.no_segmento(C, D, A):
            return True
        if abs(d4) < EPS and self.no_segmento(C, D, B):
            return True

        return False

    def triangulos_colidem(self, tri1, tri2):

        for v in tri1.vertices():
            if self.ponto_dentro_triangulo(v, tri2):
                return True

        for v in tri2.vertices():
            if self.ponto_dentro_triangulo(v, tri1):
                return True

        arestas1 = [(tri1.vertices()[i], tri1.vertices()[(i + 1) % 3]) for i in range(3)]
        arestas2 = [(tri2.vertices()[i], tri2.vertices()[(i + 1) % 3]) for i in range(3)]

        for a1, b1 in arestas1:
            for a2, b2 in arestas2:
                if self.segmentos_se_cruzam(a1, b1, a2, b2):
                    return True

        return False

    # =============================
    # BOUNDING BOX (OTIMIZAÇÃO)
    # =============================

    def bounding_box(self, tri):

        xs = [v[0] for v in tri.vertices()]
        ys = [v[1] for v in tri.vertices()]

        return min(xs), max(xs), min(ys), max(ys)

    def bbox_colidem(self, tri1, tri2):

        minx1, maxx1, miny1, maxy1 = self.bounding_box(tri1)
        minx2, maxx2, miny2, maxy2 = self.bounding_box(tri2)

        return not (
            maxx1 < minx2 or
            maxx2 < minx1 or
            maxy1 < miny2 or
            maxy2 < miny1
        )

    # =============================
    # GERAÇÃO DE TRIÂNGULOS
    # =============================

    def gerar_triangulo_equilatero(self, cx, cy, lado):

        h = (lado * np.sqrt(3)) / 2

        v1 = (cx, cy + (2/3) * h)
        v2 = (cx - lado/2, cy - (1/3) * h)
        v3 = (cx + lado/2, cy - (1/3) * h)

        return [v1, v2, v3]

    # =============================
    # ADICIONAR OBSTÁCULOS
    # =============================

    def adicionar_obstaculos_aleatorios(self, qtd, lado):
        global quant_colisoes, quant_inseridos
        margem = lado / np.sqrt(3)

        for _ in range(qtd):

            for tentativa in range(largura * 2):

                cx = random.uniform(margem, self.largura - margem)
                cy = random.uniform(margem, self.altura - margem)

                vertices = self.gerar_triangulo_equilatero(cx, cy, lado)

                novo = Triangulo(*vertices)

                colidiu = False

                for obstaculo in self.obstaculos:

                    if not self.bbox_colidem(novo, obstaculo):
                        continue

                    if self.triangulos_colidem(novo, obstaculo):
                        colidiu = True
                        quant_colisoes += 1
                        break

                if not colidiu:
                    self.obstaculos.append(novo)
                    quant_inseridos += 1
                    break

    # =============================
    # PLOTAGEM
    # =============================

    def plotar_mapa(self):

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_xlim(0, self.largura)
        ax.set_ylim(0, self.altura)
        ax.set_aspect('equal')

        ax.set_facecolor("#f5f5f5")

        for tri in self.obstaculos:

            vs = tri.vertices()
            tri_fechado = vs + [vs[0]]

            xs, ys = zip(*tri_fechado)

            ax.fill(xs, ys,
                    color="red",
                    alpha=0.5,
                    edgecolor="black",
                    linewidth=0.8)

        # Ponto inicial e ponto final
        ax.plot(0, 0, 'bs', markersize=10, label='Início')
        ax.plot(self.largura, self.altura, 'bs', markersize=10, label='Fim')

        plt.title(f"Mapa de Visibilidade: {len(self.obstaculos)} Obstáculos")
        plt.grid(True)

        plt.show()


# =============================
# EXECUÇÃO
# =============================

mapa = MapaVisibilidade(largura, altura)

mapa.adicionar_obstaculos_aleatorios(
    quantidade_obstaculos,
    lado_triangulo
)

mapa.plotar_mapa()
print(f"Colisões: {quant_colisoes}")
print(f"Inseridos: {quant_inseridos}")