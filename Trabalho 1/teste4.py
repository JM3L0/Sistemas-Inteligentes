import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

# ===== PARÂMETROS DO MAPA =====
LARGURA = 100               # Largura do mapa (eixo X)
ALTURA = 50                 # Altura do mapa (eixo Y)
QUANTIDADE_OBSTACULOS = 35  # Quantos triângulos tentar inserir
LADO_TRIANGULO = 10         # Tamanho do lado de cada triângulo equilátero

EPS = 1e-9  # Margem de tolerância para comparações com ponto flutuante

# ===== ESTRUTURA DE DADOS =====

@dataclass(unsafe_hash=True)
class Triangulo:
    """Representa um triângulo com 3 vértices (tuplas x, y)."""
    v1: tuple
    v2: tuple
    v3: tuple

    def vertices(self):
        """Retorna os 3 vértices como lista."""
        return [self.v1, self.v2, self.v3]

# ===== CLASSE PRINCIPAL =====

class MapaVisibilidade:

    def __init__(self, largura, altura):
        self.largura = largura
        self.altura = altura
        self.obstaculos = []
        self.quant_colisoes = 0
        self.quant_inseridos = 0

        # Otimização: Grid para Particionamento Espacial (Spatial Hashing)
        self.tamanho_celula = LADO_TRIANGULO * 2
        self.grid: Dict[Tuple[int, int], List[Triangulo]] = {} # Dicionário que mapeia coordenadas de células para listas de triângulos que ocupam essas células

    # =========================================================
    # 1. GERAÇÃO DE TRIÂNGULOS EQUILÁTEROS
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
    # 2. FILTRO PRECISO — GEOMETRIA COMPUTACIONAL
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
        Verifica se o ponto P está dentro do triângulo.
        Como a função `gerar_triangulo` cria sempre vértices na mesma 
        ordem (sentido anti-horário), não precisamos checar se os sinais 
        são misturados. Qualquer ponto interno nunca terá orientação negativa.
        """
        A, B, C = tri.vertices()

        # Short-circuit: se o ponto está à direita de qualquer aresta, está fora.
        if self.orientacao(A, B, P) < -EPS: return False
        if self.orientacao(B, C, P) < -EPS: return False
        if self.orientacao(C, A, P) < -EPS: return False

        return True

    def triangulos_colidem(self, tri1, tri2):
        """
        Verifica colisão real entre dois triângulos em 2 etapas:
        1. Algum vértice de tri1 está dentro de tri2?
        2. Algum vértice de tri2 está dentro de tri1?
        """
        # Etapa 1: vértices de tri1 dentro de tri2
        for vertice in tri1.vertices():
            if self.ponto_dentro_triangulo(vertice, tri2):
                return True

        # Etapa 2: vértices de tri2 dentro de tri1
        for vertice in tri2.vertices():
            if self.ponto_dentro_triangulo(vertice, tri1):
                return True

        return False

    # =========================================================
    # 3. ORQUESTRAÇÃO — INSERÇÃO DE OBSTÁCULOS ALEATÓRIOS
    # =========================================================

    def obter_celulas(self, tri: Triangulo) -> List[Tuple[int, int]]:
        """Retorna uma lista contendo as coordenadas das células do grid que este triângulo ocupa."""
        (x1, y1), (x2, y2), (x3, y3) = tri.vertices()
        minx, maxx = min(x1, x2, x3), max(x1, x2, x3)
        miny, maxy = min(y1, y2, y3), max(y1, y2, y3)
        
        # Arredonda para baixo para obter as coordenadas das células
        celula_min_x = int(minx // self.tamanho_celula)
        celula_max_x = int(maxx // self.tamanho_celula)
        celula_min_y = int(miny // self.tamanho_celula)
        celula_max_y = int(maxy // self.tamanho_celula)
        
        celulas = []
        for x in range(celula_min_x, celula_max_x + 1):
            for y in range(celula_min_y, celula_max_y + 1):
                celulas.append((x, y))
        return celulas

    def _colide_com_algum_obstaculo(self, novo: Triangulo, celulas_novo: List[Tuple[int, int]]) -> bool:
        """Verifica colisão focando APENAS nos vizinhos que habitam as mesmas áreas no grid."""
        candidatos_a_colisao = set()
        
        # Pega todos os obstáculos que dividem as mesmas células do grid
        for celula in celulas_novo:
            if celula in self.grid:
                candidatos_a_colisao.update(self.grid[celula])
                
        # Testa a colisão apenas contra esse pequeno grupo
        for obstaculo in candidatos_a_colisao:
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
        margem_x = (lado / 2) + EPS                         # Extensão horizontal do triângulo
        margem_y_topo = (lado / np.sqrt(3)) + EPS           # Distância do centro à ponta de cima
        margem_y_base = (lado / (2 * np.sqrt(3))) + EPS     # Distância do centro à base

        for _ in range(qtd):
            for tentativa in range(self.largura * 2):
                cx = random.uniform(margem_x, self.largura - margem_x)          # Sorteia a coordenada x do centro do triângulo
                cy = random.uniform(margem_y_base, self.altura - margem_y_topo) # Sorteia a coordenada y do centro do triângulo
                novo = self.gerar_triangulo(cx, cy, lado)                       # Gera o triângulo
                
                celulas_novo = self.obter_celulas(novo)

                if not self._colide_com_algum_obstaculo(novo, celulas_novo):    # Verifica se o novo triângulo colide com algum obstáculo existente
                    self.obstaculos.append(novo)                                # Adiciona o novo triângulo à lista de obstáculos
                    self.quant_inseridos += 1                                   # Incrementa o contador de obstáculos inseridos
                    
                    for celula in celulas_novo:                                 # Adiciona o novo triângulo ao grid
                        if celula not in self.grid:                             # Se a célula não existir, cria
                            self.grid[celula] = []                              # Cria a célula 
                        self.grid[celula].append(novo)                          # Adiciona o novo triângulo à célula
                        
                    break

    # =========================================================
    # 4. SAÍDA — PLOTAGEM DO MAPA
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
        ax.plot(0, 0, 'bs')
        ax.plot(self.largura, self.altura, 'gs')

        plt.title(f"Obstáculos: {len(self.obstaculos)}  |  Colisões: {self.quant_colisoes}")
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