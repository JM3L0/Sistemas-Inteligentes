import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import random
from dataclasses import dataclass

# ===== PARÂMETROS =====
largura = 100
altura = 50
quantidade_obstaculos = 30
lado_triangulo = 10

EPS = 1e-9 # Constante de precisão


@dataclass
class Triangulo:
    v1: tuple
    v2: tuple
    v3: tuple

    def vertices(self):
        return [self.v1, self.v2, self.v3]

    def arestas(self):
        vs = self.vertices()
        return [(vs[i], vs[(i+1)%3]) for i in range(3)]


class MapaVisibilidade:

    def __init__(self, largura, altura):
        self.largura = largura
        self.altura = altura
        self.obstaculos = []
        self.quant_colisoes = 0
        self.quant_inseridos = 0

    # =====================
    # GEOMETRIA
    # =====================

    def orientacao(self, A, B, C):
        return (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])

    def ponto_dentro_triangulo(self, P, tri):

        A, B, C = tri.vertices()

        d1 = self.orientacao(A,B,P)
        d2 = self.orientacao(B,C,P)
        d3 = self.orientacao(C,A,P)

        neg = (d1 < -EPS) or (d2 < -EPS) or (d3 < -EPS)
        pos = (d1 > EPS) or (d2 > EPS) or (d3 > EPS)

        return not (neg and pos)

    def ponto_no_segmento(self, A,B,P):

        return (
            min(A[0],B[0]) <= P[0] <= max(A[0],B[0]) and
            min(A[1],B[1]) <= P[1] <= max(A[1],B[1])
        )

    def segmentos_cruzam(self,A,B,C,D):

        d1 = self.orientacao(A,B,C)
        d2 = self.orientacao(A,B,D)
        d3 = self.orientacao(C,D,A)
        d4 = self.orientacao(C,D,B)

        if (d1*d2 < 0) and (d3*d4 < 0):
            return True

        if abs(d1) < EPS and self.ponto_no_segmento(A,B,C): return True
        if abs(d2) < EPS and self.ponto_no_segmento(A,B,D): return True
        if abs(d3) < EPS and self.ponto_no_segmento(C,D,A): return True
        if abs(d4) < EPS and self.ponto_no_segmento(C,D,B): return True

        return False


    # =====================
    # COLISÃO
    # =====================

    def triangulos_colidem(self, t1, t2):

        for v in t1.vertices():
            if self.ponto_dentro_triangulo(v,t2):
                return True

        for v in t2.vertices():
            if self.ponto_dentro_triangulo(v,t1):
                return True

        for a1,b1 in t1.arestas():
            for a2,b2 in t2.arestas():
                if self.segmentos_cruzam(a1,b1,a2,b2):
                    return True

        return False


    # =====================
    # BOUNDING BOX
    # =====================

    def bounding_box(self, tri):

        xs = [v[0] for v in tri.vertices()]
        ys = [v[1] for v in tri.vertices()]

        return min(xs),max(xs),min(ys),max(ys)


    def bbox_colidem(self,t1,t2):

        minx1,maxx1,miny1,maxy1 = self.bounding_box(t1)
        minx2,maxx2,miny2,maxy2 = self.bounding_box(t2)

        return not (
            maxx1 < minx2 or
            maxx2 < minx1 or
            maxy1 < miny2 or
            maxy2 < miny1
        )


    # =====================
    # GERAÇÃO TRIÂNGULOS
    # =====================

    def gerar_triangulo(self,cx,cy,lado):

        h = (lado*np.sqrt(3))/2

        v1 = (cx,cy + (2/3)*h)
        v2 = (cx - lado/2 , cy - h/3)
        v3 = (cx + lado/2 , cy - h/3)

        return Triangulo(v1,v2,v3)


    # =====================
    # GERAR OBSTÁCULOS
    # =====================

    def adicionar_obstaculos_aleatorios(self,qtd,lado):

        margem = lado/np.sqrt(3)

        for _ in range(qtd):

            for tentativa in range(self.largura*2):

                cx = random.uniform(margem,self.largura-margem)
                cy = random.uniform(margem,self.altura-margem)

                novo = self.gerar_triangulo(cx,cy,lado)

                colidiu = False

                for obs in self.obstaculos:

                    if not self.bbox_colidem(novo,obs):
                        continue

                    if self.triangulos_colidem(novo,obs):
                        colidiu = True
                        self.quant_colisoes += 1
                        break

                if not colidiu:
                    self.obstaculos.append(novo)
                    self.quant_inseridos += 1
                    break


    # =====================
    # PLOTAGEM
    # =====================

    def plotar_mapa(self):

        fig,ax = plt.subplots(figsize=(8,4))

        ax.set_xlim(0,self.largura)
        ax.set_ylim(0,self.altura)
        ax.set_aspect('equal')

        for tri in self.obstaculos:

            vs = tri.vertices()
            xs,ys = zip(*(vs+[vs[0]]))

            ax.fill(xs,ys,color="red",alpha=0.5,edgecolor="black")

        ax.plot(0,0,'bs',label="Início")
        ax.plot(self.largura,self.altura,'gs',label="Fim")

        plt.title(f"Obstáculos: {len(self.obstaculos)}  |  Colisões: {self.quant_colisoes}")
        plt.grid()
        plt.show()



# ===== EXECUÇÃO =====

mapa = MapaVisibilidade(largura,altura)

mapa.adicionar_obstaculos_aleatorios(
    quantidade_obstaculos,
    lado_triangulo
)

mapa.plotar_mapa()

print("Colisões:",mapa.quant_colisoes)
print("Inseridos:",mapa.quant_inseridos)
