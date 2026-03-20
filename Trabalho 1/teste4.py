import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

# ===== PARÂMETROS DO MAPA =====
LARGURA = 100
ALTURA = 50
QUANTIDADE_OBSTACULOS = 35
LADO_TRIANGULO = 10

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
        Endpoints compartilhados NAO contam como cruzamento.
        """
        def cross(O, U, V):
            return (U[0] - O[0]) * (V[1] - O[1]) - (U[1] - O[1]) * (V[0] - O[0])

        d1 = cross(A, B, P)
        d2 = cross(A, B, Q)
        d3 = cross(P, Q, A)
        d4 = cross(P, Q, B)

        if (d1 > 0 and d2 < 0 or d1 < 0 and d2 > 0) and \
           (d3 > 0 and d4 < 0 or d3 < 0 and d4 > 0):
            return True

        def sobre(O, U, V):
            return (min(U[0], V[0]) <= O[0] <= max(U[0], V[0]) and
                    min(U[1], V[1]) <= O[1] <= max(U[1], V[1]))

        if d1 == 0 and sobre(P, A, B): return True
        if d2 == 0 and sobre(Q, A, B): return True
        if d3 == 0 and sobre(A, P, Q): return True
        if d4 == 0 and sobre(B, P, Q): return True

        return False

    def _eixo_traversal(self, p: float, q: float, c: int) -> Tuple[int, float, float]:
        """
        Calcula os parâmetros de traversal para um único eixo.
        Retorna (step, t_max, t_delta):
          step    → direção do avanço (-1, 0 ou 1)
          t_max   → fração do segmento até cruzar a próxima borda
          t_delta → quanto t avança para atravessar uma célula inteira
        """
        tc = self.tamanho_celula
        if p == q:
            return 0, float('inf'), float('inf')
        step:    int   = 1 if q > p else -1
        borda:   float = (c + (1 if step > 0 else 0)) * tc
        t_max:   float = (borda - p) / (q - p)
        t_delta: float = tc / abs(q - p)
        return step, t_max, t_delta

    def _celulas_do_segmento(self, P: tuple, Q: tuple) -> List[Tuple[int, int]]:
        """
        Retorna exatamente as células do grid que o segmento PQ atravessa.
        A cada passo avança para a célula cuja borda (vertical ou horizontal)
        o segmento cruza primeiro, usando a fração 't' do segmento como métrica.
        """
        tc = self.tamanho_celula

        cx,  cy  = int(P[0] // tc), int(P[1] // tc)
        cx1, cy1 = int(Q[0] // tc), int(Q[1] // tc)

        step_x, t_max_x, t_delta_x = self._eixo_traversal(P[0], Q[0], cx)
        step_y, t_max_y, t_delta_y = self._eixo_traversal(P[1], Q[1], cy)

        celulas: List[Tuple[int, int]] = [(cx, cy)]

        while (cx, cy) != (cx1, cy1):
            if t_max_x < t_max_y:  # type: ignore  # próxima borda é vertical
                cx      += step_x
                t_max_x += t_delta_x
            else:                                   # próxima borda é horizontal
                cy      += step_y
                t_max_y += t_delta_y
            celulas.append((cx, cy))

        return celulas

    def _caminho_livre(self, P: tuple, Q: tuple) -> bool:
        """
        Retorna True se o segmento PQ nao cruza nenhuma aresta de obstaculo.
        Usa o grid traversal para testar apenas os triangulos nas celulas
        que o segmento PQ atravessa — em vez de testar todos os obstaculos.
        """
        candidatos: set = set()
        for celula in self._celulas_do_segmento(P, Q):
            if celula in self.grid:
                candidatos.update(self.grid[celula]) # type: ignore

        for tri in candidatos:
            for (A, B) in tri.arestas():
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

        lista_nos: List[tuple] = [] # type: ignore
        vistos: set = set()
        for tri in self.obstaculos:
            for v in tri.vertices():
                if v not in vistos:
                    lista_nos.append(v) # type: ignore
                    vistos.add(v)
        for p in (ORIGEM, DESTINO):
            if p not in vistos:
                lista_nos.append(p) # type: ignore
                vistos.add(p)

        n = len(lista_nos)
        print(f"[Grafo] {n} vertices | {n*(n-1)//2} pares a testar...")

        self.grafo = {v: [] for v in lista_nos} # type: ignore

        for i, P in enumerate(lista_nos): # type: ignore
            for Q in lista_nos[i + 1:]: # type: ignore
                if self._caminho_livre(P, Q):
                    self.grafo[P].append(Q) # type: ignore
                    self.grafo[Q].append(P) # type: ignore

        arestas = sum(len(v) for v in self.grafo.values()) // 2
        print(f"[Grafo] {arestas} arestas de visibilidade.")

    # =========================================================
    # BLOCO 4 — PLOTAGEM
    # =========================================================

    def plotar_mapa(self, mostrar_grafo=True):
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.set_xlim(0, self.largura)
        ax.set_ylim(0, self.altura)
        ax.set_aspect('equal')

        # Obstaculos
        for tri in self.obstaculos:
            vs = tri.vertices()
            xs, ys = zip(*(vs + [vs[0]]))
            ax.fill(xs, ys, color="red", alpha=0.5, edgecolor="black")

        # Arestas do grafo de visibilidade
        if mostrar_grafo and self.grafo:
            plotados = set()
            for P, vizinhos in self.grafo.items():
                for Q in vizinhos:
                    chave = (min(P, Q), max(P, Q))
                    if chave not in plotados:
                        ax.plot([P[0], Q[0]], [P[1], Q[1]],
                                color='blue', linewidth=0.3, alpha=0.5)
                        plotados.add(chave)

        # Origem e destino
        ax.plot(0, 0, 'bs')
        ax.plot(self.largura, self.altura, 'gs')

        arestas = sum(len(v) for v in self.grafo.values()) // 2 if self.grafo else 0
        plt.title(f"Obstáculos: {len(self.obstaculos)}  |  Colisões: {self.quant_colisoes}  |  Arestas visíveis: {arestas}")
        plt.grid()
        plt.show()


# ===== EXECUCAO =====

mapa = MapaVisibilidade(LARGURA, ALTURA)

print("Gerando obstaculos...")
mapa.adicionar_obstaculos_aleatorios(QUANTIDADE_OBSTACULOS, LADO_TRIANGULO)
print(f"Inseridos: {mapa.quant_inseridos}  |  Colisoes: {mapa.quant_colisoes}")

print("Construindo grafo de visibilidade...")
mapa.construir_grafo()

mapa.plotar_mapa(mostrar_grafo=True)