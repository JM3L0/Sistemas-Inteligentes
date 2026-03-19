import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

# ===== PARÂMETROS DO MAPA =====
LARGURA = 1000
ALTURA = 1000
QUANTIDADE_OBSTACULOS = 300
LADO_TRIANGULO = 20

EPS = 0

# ===== ESTRUTURA DE DADOS =====

@dataclass(unsafe_hash=True)
class Triangulo:
    """Representa um triângulo com 3 vértices (tuplas x, y)."""
    v1: tuple
    v2: tuple
    v3: tuple

    def vertices(self):
        return [self.v1, self.v2, self.v3]

    def arestas(self):
        vs = self.vertices()
        return [(vs[0], vs[1]), (vs[1], vs[2]), (vs[2], vs[0])]


# ===== CLASSE PRINCIPAL =====

class MapaVisibilidade:

    def __init__(self, largura, altura):
        self.largura = largura
        self.altura = altura
        self.obstaculos: List[Triangulo] = []
        self.quant_colisoes = 0
        self.quant_inseridos = 0

        # Spatial hashing — inalterado
        self.tamanho_celula = LADO_TRIANGULO * 2
        self.grid: Dict[Tuple[int, int], List[Triangulo]] = {}

        # Grafo de visibilidade
        self.grafo: Dict[tuple, List[tuple]] = {}

    # =========================================================
    # BLOCO 1 — INSERCAO DE OBSTACULOS (logica original intacta)
    # =========================================================

    def gerar_triangulo(self, cx, cy, lado):
        altura = (lado * np.sqrt(3)) / 2
        ponta_superior = (cx, cy + (2 / 3) * altura)
        base_esquerda  = (cx - lado / 2, cy - altura / 3)
        base_direita   = (cx + lado / 2, cy - altura / 3)
        return Triangulo(ponta_superior, base_esquerda, base_direita)

    def orientacao(self, A, B, C):
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

    def ponto_dentro_triangulo(self, P, tri):
        A, B, C = tri.vertices()
        if self.orientacao(A, B, P) < -EPS: return False
        if self.orientacao(B, C, P) < -EPS: return False
        if self.orientacao(C, A, P) < -EPS: return False
        return True

    def triangulos_colidem(self, tri1, tri2):
        for v in tri1.vertices():
            if self.ponto_dentro_triangulo(v, tri2):
                return True
        for v in tri2.vertices():
            if self.ponto_dentro_triangulo(v, tri1):
                return True
        return False

    def obter_celulas(self, tri):
        (x1, y1), (x2, y2), (x3, y3) = tri.vertices()
        minx, maxx = min(x1, x2, x3), max(x1, x2, x3)
        miny, maxy = min(y1, y2, y3), max(y1, y2, y3)
        cx0, cx1 = int(minx // self.tamanho_celula), int(maxx // self.tamanho_celula)
        cy0, cy1 = int(miny // self.tamanho_celula), int(maxy // self.tamanho_celula)
        return [(x, y) for x in range(cx0, cx1 + 1) for y in range(cy0, cy1 + 1)]

    def _colide_com_algum_obstaculo(self, novo, celulas_novo):
        candidatos = set()
        for c in celulas_novo:
            if c in self.grid:
                candidatos.update(self.grid[c])
        for ob in candidatos:
            if self.triangulos_colidem(novo, ob):
                self.quant_colisoes += 1
                return True
        return False

    def adicionar_obstaculos_aleatorios(self, qtd, lado):
        margem_x     = (lado / 2) + EPS
        margem_y_top = (lado / np.sqrt(3)) + EPS
        margem_y_bas = (lado / (2 * np.sqrt(3))) + EPS
        for _ in range(qtd):
            for _ in range(self.largura * 2):
                cx = random.uniform(margem_x, self.largura - margem_x)
                cy = random.uniform(margem_y_bas, self.altura - margem_y_top)
                novo = self.gerar_triangulo(cx, cy, lado)
                celulas = self.obter_celulas(novo)
                if not self._colide_com_algum_obstaculo(novo, celulas):
                    self.obstaculos.append(novo)
                    self.quant_inseridos += 1
                    for c in celulas:
                        self.grid.setdefault(c, []).append(novo)
                    break

    # =========================================================
    # BLOCO 2 — CRUZAMENTO DE SEGMENTOS (para o grafo)
    # =========================================================

    def _segmentos_cruzam(self, P, Q, A, B) -> bool:
        """
        Retorna True se os segmentos PQ e AB se cruzam.
        Endpoints compartilhados NAO contam como cruzamento —
        isso permite que vertices adjacentes de um mesmo triangulo
        sempre sejam vizinhos no grafo.
        """
        def cross(O, U, V):
            return (U[0] - O[0]) * (V[1] - O[1]) - (U[1] - O[1]) * (V[0] - O[0])

        d1 = cross(A, B, P)
        d2 = cross(A, B, Q)
        d3 = cross(P, Q, A)
        d4 = cross(P, Q, B)

        # Cruzamento estrito: pontos de cada segmento em lados opostos
        if (d1 > 0 and d2 < 0 or d1 < 0 and d2 > 0) and \
           (d3 > 0 and d4 < 0 or d3 < 0 and d4 > 0):
            return True

        # Caso colinear: ponto sobre o segmento oposto
        def sobre(O, U, V):
            return (min(U[0], V[0]) <= O[0] <= max(U[0], V[0]) and
                    min(U[1], V[1]) <= O[1] <= max(U[1], V[1]))

        if d1 == 0 and sobre(P, A, B): return True
        if d2 == 0 and sobre(Q, A, B): return True
        if d3 == 0 and sobre(A, P, Q): return True
        if d4 == 0 and sobre(B, P, Q): return True

        return False

    def _caminho_livre(self, P: tuple, Q: tuple) -> bool:
        """
        Retorna True se o segmento PQ nao cruza nenhuma aresta de obstaculo,
        exceto arestas que compartilham P ou Q como endpoint.
        """
        for tri in self.obstaculos:
            for (A, B) in tri.arestas():
                # Aresta compartilha endpoint com o segmento? Ignorar.
                if A == P or A == Q or B == P or B == Q:
                    continue
                if self._segmentos_cruzam(P, Q, A, B):
                    return False
        return True

    # =========================================================
    # BLOCO 3 — CONSTRUCAO DO GRAFO DE VISIBILIDADE
    # =========================================================

    def construir_grafo(self):
        """
        Nos = todos os vertices dos obstaculos + ORIGEM(0,0) + DESTINO(largura,altura).
        Aresta existe entre P e Q se o segmento PQ nao cruza
        nenhuma aresta de nenhum triangulo.
        """
        ORIGEM  = (0.0, 0.0)
        DESTINO = (float(self.largura), float(self.altura))

        nos = []
        vistos = set()
        for tri in self.obstaculos:
            for v in tri.vertices():
                if v not in vistos:
                    nos.append(v)
                    vistos.add(v)
        for p in (ORIGEM, DESTINO):
            if p not in vistos:
                nos.append(p)
                vistos.add(p)

        n = len(nos)
        print(f"[Grafo] {n} vertices | {n*(n-1)//2} pares a testar...")

        self.grafo = {v: [] for v in nos}

        for i in range(n):
            for j in range(i + 1, n):
                P, Q = nos[i], nos[j]
                if self._caminho_livre(P, Q):
                    self.grafo[P].append(Q)
                    self.grafo[Q].append(P)

        arestas = sum(len(v) for v in self.grafo.values()) // 2
        print(f"[Grafo] {arestas} arestas de visibilidade.")

    # =========================================================
    # BLOCO 4 — A* SOBRE O GRAFO
    # =========================================================

    def astar(self):
        import heapq

        ORIGEM  = (0.0, 0.0)
        DESTINO = (float(self.largura), float(self.altura))

        def h(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        fila = [(0.0, ORIGEM)]
        veio_de = {ORIGEM: None}
        g = {ORIGEM: 0.0}

        while fila:
            _, atual = heapq.heappop(fila)
            if atual == DESTINO:
                break
            for viz in self.grafo.get(atual, []):
                novo_g = g[atual] + h(atual, viz)
                if viz not in g or novo_g < g[viz]:
                    g[viz] = novo_g
                    heapq.heappush(fila, (novo_g + h(viz, DESTINO), viz))
                    veio_de[viz] = atual

        if DESTINO not in veio_de:
            print("[A*] Caminho nao encontrado.")
            return []

        caminho, no = [], DESTINO
        while no is not None:
            caminho.append(no)
            no = veio_de[no]
        caminho.reverse()
        print(f"[A*] Caminho com {len(caminho)} vertices encontrado.")
        return caminho

    # =========================================================
    # BLOCO 5 — PLOTAGEM
    # =========================================================

    def plotar_mapa(self, caminho=None, mostrar_grafo=True):
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_xlim(0, self.largura)
        ax.set_ylim(0, self.altura)
        ax.set_aspect('equal')
        ax.set_facecolor('#12121f')
        fig.patch.set_facecolor('#12121f')

        # Obstaculos
        for tri in self.obstaculos:
            vs = tri.vertices()
            xs, ys = zip(*(vs + [vs[0]]))
            ax.fill(xs, ys, color='#c0392b', alpha=0.75, edgecolor='#e74c3c', linewidth=0.6)

        # Arestas do grafo de visibilidade
        if mostrar_grafo and self.grafo:
            plotados = set()
            for P, vizinhos in self.grafo.items():
                for Q in vizinhos:
                    chave = (min(P, Q), max(P, Q))
                    if chave not in plotados:
                        ax.plot([P[0], Q[0]], [P[1], Q[1]],
                                color='#1e3a5c', linewidth=0.25, alpha=0.9, zorder=2)
                        plotados.add(chave)

        # Caminho otimo
        if caminho and len(caminho) > 1:
            xs = [p[0] for p in caminho]
            ys = [p[1] for p in caminho]
            ax.plot(xs, ys, color='#00e5ff', linewidth=2.0,
                    zorder=5, marker='o', markersize=3, label='Caminho otimo (A*)')

        # Origem e destino
        ax.plot(0, 0, marker='s', color='#2ecc71', markersize=12,
                zorder=6, label='Origem (0, 0)')
        ax.plot(self.largura, self.altura, marker='s', color='#f1c40f',
                markersize=12, zorder=6, label=f'Destino ({self.largura}, {self.altura})')

        arestas = sum(len(v) for v in self.grafo.values()) // 2 if self.grafo else 0
        ax.set_title(
            f"Obstaculos: {len(self.obstaculos)}  |  "
            f"Vertices: {len(self.grafo)}  |  "
            f"Arestas visiveis: {arestas}",
            color='white', fontsize=11, pad=10
        )
        ax.tick_params(colors='#666')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9, loc='upper left')
        ax.grid(color='#1e1e2e', linewidth=0.4)
        plt.tight_layout()
        plt.show()


# ===== EXECUCAO =====

mapa = MapaVisibilidade(LARGURA, ALTURA)

print("Gerando obstaculos...")
mapa.adicionar_obstaculos_aleatorios(QUANTIDADE_OBSTACULOS, LADO_TRIANGULO)
print(f"Inseridos: {mapa.quant_inseridos}  |  Colisoes: {mapa.quant_colisoes}")

print("Construindo grafo de visibilidade...")
mapa.construir_grafo()

print("Calculando caminho otimo (A*)...")
caminho = mapa.astar()

mapa.plotar_mapa(caminho=caminho, mostrar_grafo=True)