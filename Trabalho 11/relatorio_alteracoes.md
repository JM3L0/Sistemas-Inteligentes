# Relatório de Alterações — CNN Adaptada para Fashion-MNIST

> **Arquivo original:** `cnn_architecture_training (1).ipynb`  
> **Arquivo gerado:** `cnn_architecture_training_fashionmnist.ipynb`  
> **Atividade:** Treinar e Avaliar a arquitetura CNN adaptada para outro conjunto de dados

---

## 1. Dataset Escolhido: Fashion-MNIST

O **Fashion-MNIST** (Zalando Research, 2017) foi escolhido como dataset alvo por três razões principais:

| Critério | Justificativa |
|---|---|
| **Compatibilidade estrutural** | Imagens 28×28 px em escala de cinza — mesma estrutura do MNIST. Nenhuma alteração no `input_shape` foi necessária. |
| **Mesmo número de classes** | 10 categorias → camada de saída com `softmax` de 10 unidades permanece idêntica. |
| **Complexidade adequada** | Padrões visuais de vestuário são mais desafiadores que dígitos, tornando o exercício mais rico didaticamente. |

**Classes do Fashion-MNIST:**

```
0: Camiseta/Top    1: Calça       2: Pullover   3: Vestido   4: Casaco
5: Sandália        6: Camisa      7: Tênis      8: Bolsa     9: Bota
```

---

## 2. O que foi mantido (sem alteração)

- **`input_shape = (28, 28, 1)`** — Fashion-MNIST tem o mesmo formato de imagem do MNIST.
- **Camada de saída:** `Dense(10, activation='softmax')` — mesmo número de classes (10).
- **Estrutura geral da rede:** Conv → Pool → Conv → Pool → Flatten → Dropout → Dense → Saída.
- **`Dropout(0.5)`** — taxa de regularização mantida.
- **`validation_split = 0.1`** — 10% dos dados de treino reservados para validação.
- **`loss = 'categorical_crossentropy'`** — mesma função de perda para classificação multi-classe.
- **Callbacks `ModelCheckpoint` e `EarlyStopping`** — mantidos para boas práticas de treino.

---

## 3. O que foi alterado e por quê

### 3.1 Filtros das Camadas Convolucionais

| Camada | MNIST original | Fashion-MNIST adaptado | Motivo |
|---|---|---|---|
| `conv_1` | 32 filtros | **64 filtros** | Padrões de vestuário têm texturas e bordas mais variadas que dígitos; mais filtros geram mais mapas de características distintos na extração inicial |
| `conv_2` | 64 filtros | **128 filtros** | A segunda camada combina padrões da primeira; dobrando os filtros a rede consegue representar combinações mais complexas de características |

### 3.2 Unidades da Camada Densa

| Camada | MNIST original | Fashion-MNIST adaptado | Motivo |
|---|---|---|---|
| `dense_1` | 128 unidades | **256 unidades** | Maior espaço de representação latente para discriminar 10 classes visualmente ambíguas (ex.: camiseta vs. camisa vs. pullover são muito parecidos entre si) |

### 3.3 Otimizador

| | MNIST original | Fashion-MNIST adaptado | Motivo |
|---|---|---|---|
| **Otimizador** | `SGD` | **`Adam`** | O Adam usa taxas de aprendizado adaptativas por parâmetro, levando a convergência mais rápida e estável. Em um ambiente **sem GPU**, isso é crucial para reduzir o tempo de treinamento sem sacrificar qualidade. O SGD simples precisa de muito mais épocas para convergir em datasets mais complexos. |

### 3.4 Número de Épocas

| | MNIST original | Fashion-MNIST adaptado | Motivo |
|---|---|---|---|
| **Épocas** | 5 | **15** | O Fashion-MNIST é mais difícil; 5 épocas são insuficientes para convergência. O `EarlyStopping` garante que o treino pare automaticamente se não houver melhora, evitando overfitting e desperdício de tempo de CPU. |

### 3.5 Batch Size

| | MNIST original | Fashion-MNIST adaptado | Motivo |
|---|---|---|---|
| **Batch size** | 50 | **64** | Valor como potência de 2 melhora a eficiência das operações matriciais em CPU (melhor uso de cache e cores). Levemente maior para acelerar cada época. |

### 3.6 EarlyStopping patience

| | MNIST original | Fashion-MNIST adaptado | Motivo |
|---|---|---|---|
| **patience** | 15 | **5** | Em um ambiente **sem GPU**, cada época tem custo de tempo significativo. Uma patience de 15 poderia resultar em horas de treino sem melhora real. Com patience=5, o treino para após 5 épocas sem progresso, mantendo o tempo total razoável. |

### 3.7 Arquivo de Checkpoint

| | MNIST original | Fashion-MNIST adaptado | Motivo |
|---|---|---|---|
| **Arquivo salvo** | `best_model.h5` | **`best_model_fashion.h5`** | Arquivo separado para não sobrescrever o modelo original do MNIST, permitindo comparação posterior. |

---

## 4. Novas Variáveis Criadas

Para não conflitar com as variáveis do modelo MNIST no mesmo notebook, todas as variáveis da seção Fashion-MNIST receberam sufixo ou nome diferente:

| Variável | Tipo | Descrição |
|---|---|---|
| `class_names` | `list` | Nomes das 10 classes do Fashion-MNIST |
| `X_train_f`, `X_test_f` | `ndarray` | Dados de imagem normalizados e reshapeados |
| `y_train_f`, `y_test_f` | `ndarray` | Rótulos one-hot encoded |
| `classifier_fashion` | `Sequential` | Modelo CNN adaptado |
| `epochs_f` | `int` | Número máximo de épocas (15) |
| `batch_size_f` | `int` | Tamanho do batch (64) |
| `val_split_f` | `float` | Fração de validação (0.1) |
| `checkpoint_f` | `Callback` | Salva o melhor modelo em `best_model_fashion.h5` |
| `earlystop_f` | `Callback` | Para o treino após patience=5 épocas sem melhora |
| `history_fashion` | `History` | Histórico de métricas por época |
| `best_model_fashion` | `Model` | Melhor modelo carregado para avaliação |
| `score_f` | `list` | `[loss, accuracy]` no conjunto de teste |

---

## 5. Comparação de Resultados

| Métrica | MNIST (original) | Fashion-MNIST (adaptado) |
|---|---|---|
| Dataset | Dígitos escritos à mão | Peças de vestuário e acessórios |
| Dimensão das imagens | 28×28 cinza | 28×28 cinza |
| Nº de classes | 10 | 10 |
| Otimizador | SGD | Adam |
| Épocas treinadas | 5 | até 15 (com EarlyStopping) |
| Batch size | 50 | 64 |
| Test Loss | 0.0415 | *(preencher após execução)* |
| Test Accuracy | **98.68%** | *(esperado: ~88–92%)* |

> **Nota:** A acurácia esperada no Fashion-MNIST (~88–92%) é menor que no MNIST (~98%) por design.
> O dataset é genuinamente mais difícil — classes como "camiseta", "camisa" e "pullover" têm aparência
> visual muito similar, tornando a discriminação mais desafiadora mesmo para redes bem treinadas.
> Essa diferença de desempenho é o principal aprendizado da atividade.

---

## 6. Como Executar

1. Abra `cnn_architecture_training_fashionmnist.ipynb` no Jupyter Notebook ou VS Code
2. Execute as células **sequencialmente** do início ao fim
3. A seção da atividade começa na célula com o título:  
   `## Atividade — Treinar e Avaliar a CNN Adaptada para Fashion-MNIST`
4. Após a execução completa, preencha a tabela da seção **A.6** com os valores reais obtidos
