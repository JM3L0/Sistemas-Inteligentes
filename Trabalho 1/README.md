# Mapa de Visibilidade com Obstáculos Triangulares

## Descrição Geral

Este programa gera um **mapa 2D** com obstáculos em formato de **triângulos equiláteros** posicionados aleatoriamente, sem sobreposição. Utiliza técnicas de **geometria computacional** para detecção de colisão e **particionamento espacial (Spatial Hashing)** para otimizar a performance.

---

## Parâmetros Configuráveis

| Parâmetro              | Valor Padrão | Descrição                                    |
|------------------------|:------------:|----------------------------------------------|
| `LARGURA`              | 100          | Largura do mapa (eixo X)                     |
| `ALTURA`               | 50           | Altura do mapa (eixo Y)                      |
| `QUANTIDADE_OBSTACULOS`| 35           | Quantidade de triângulos a inserir            |
| `LADO_TRIANGULO`       | 10           | Comprimento do lado de cada triângulo         |
| `EPS`                  | 1e-9         | Tolerância para comparações de ponto flutuante|

---

## Estrutura do Código

### Classe `Triangulo`

Dataclass que armazena os 3 vértices (`v1`, `v2`, `v3`) de um triângulo como tuplas `(x, y)`.

### Classe `MapaVisibilidade`

Classe principal com 4 responsabilidades:

---

### 1. Geração de Triângulos — `gerar_triangulo(cx, cy, lado)`

Cria um triângulo equilátero **centrado** em `(cx, cy)` com a ponta voltada para cima.

**Cálculos:**

A altura de um triângulo equilátero de lado $l$ é:

$$h = \frac{l \cdot \sqrt{3}}{2}$$

O centroide divide a altura na razão **2:1** (do vértice superior à base). Portanto, os vértices são calculados como:

| Vértice         | Coordenada X          | Coordenada Y              |
|-----------------|-----------------------|---------------------------|
| Ponta superior  | `cx`                  | `cy + (2/3) * h`          |
| Base esquerda   | `cx - lado/2`         | `cy - h/3`                |
| Base direita    | `cx + lado/2`         | `cy - h/3`                |

> O centroide fica a `2h/3` da base e `h/3` do topo, garantindo que `(cx, cy)` é o centro geométrico real do triângulo.

---

### 2. Detecção de Colisão — Geometria Computacional

#### `orientacao(A, B, C)` — Produto Vetorial 2D

Calcula o **produto vetorial** dos vetores $\vec{AB}$ e $\vec{AC}$:

$$\text{orientação} = (B_x - A_x) \cdot (C_y - A_y) - (B_y - A_y) \cdot (C_x - A_x)$$

| Resultado     | Significado                            |
|:-------------:|----------------------------------------|
| `> 0`         | C está à **esquerda** da reta A→B      |
| `< 0`         | C está à **direita** da reta A→B       |
| `= 0`         | A, B e C são **colineares**            |

#### `ponto_dentro_triangulo(P, tri)` — Teste de Inclusão

Um ponto P está **dentro** do triângulo ABC (com vértices em sentido anti-horário) se e somente se:

- `orientacao(A, B, P) ≥ 0`
- `orientacao(B, C, P) ≥ 0`
- `orientacao(C, A, P) ≥ 0`

Se qualquer orientação for **negativa** (menor que `-EPS`), o ponto está fora — o método retorna `False` imediatamente (short-circuit).

#### `triangulos_colidem(tri1, tri2)` — Colisão entre Triângulos

Verifica em **duas etapas**:

1. Algum vértice de `tri1` está dentro de `tri2`?
2. Algum vértice de `tri2` está dentro de `tri1`?

Se qualquer teste for verdadeiro, há colisão.

---

### 3. Inserção de Obstáculos — Spatial Hashing

#### Grid de Particionamento Espacial

O mapa é dividido em **células** de tamanho `LADO_TRIANGULO * 2`. Cada célula contém uma lista dos triângulos que a ocupam. Isso evita comparar um novo triângulo com **todos** os existentes — apenas os triângulos nas mesmas células são testados.

#### `obter_celulas(tri)` — Mapeamento para o Grid

Calcula o **bounding box** do triângulo (min/max de X e Y) e retorna todas as células do grid que esse retângulo cobre:

```
célula_x = floor(coordenada_x / tamanho_célula)
célula_y = floor(coordenada_y / tamanho_célula)
```

#### `adicionar_obstaculos_aleatorios(qtd, lado)` — Inserção com Retry

Para cada triângulo a inserir:

1. Calcula **margens de segurança** para garantir que o triângulo fique dentro do mapa:
   - `margem_x = lado/2` — meia largura do triângulo
   - `margem_y_topo = lado / √3` — distância do centro à ponta superior
   - `margem_y_base = lado / (2√3)` — distância do centro à base

2. Sorteia coordenadas `(cx, cy)` respeitando as margens

3. Gera o triângulo e verifica colisão apenas contra vizinhos do grid

4. Se não houver colisão, insere o triângulo na lista de obstáculos e registra-o no grid

5. Se houver colisão, tenta novamente (até `LARGURA * 2` tentativas por triângulo)

---

### 4. Visualização — `plotar_mapa()`

Usa **Matplotlib** para renderizar o mapa:

- Triângulos preenchidos em **vermelho** (`alpha=0.5`) com borda preta
- Ponto **azul** em `(0, 0)` — ponto de início
- Ponto **verde** em `(LARGURA, ALTURA)` — ponto de destino
- Título com contagem de obstáculos inseridos e colisões detectadas
- Grade habilitada para referência visual

---

## Execução

```bash
python Mapa_de_visibilidade.py
```

**Saída:**
- Uma janela Matplotlib com o mapa gerado
- No console: quantidade de colisões detectadas e obstáculos inseridos

---

## Dependências

- `matplotlib` — Plotagem do mapa
- `numpy` — Cálculos matemáticos (`sqrt`)
- `random` — Geração de posições aleatórias
