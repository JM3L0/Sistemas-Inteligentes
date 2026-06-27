# Contextualização Detalhada do Projeto: Classificação de Gliomas (TCGA)

Este documento centraliza todas as informações do projeto da disciplina de Sistemas Inteligentes, desde os requisitos acadêmicos até as decisões arquiteturais de software e os resultados experimentais. O objetivo é fornecer uma base sólida para a redação do artigo científico e preparação da apresentação oral.

---

## 1. Visão Geral e Requisitos Acadêmicos
O projeto final exige a aplicação prática de técnicas de Sistemas Inteligentes a um problema real, utilizando um conjunto de dados referenciado por um artigo de alto impacto (Qualis A, >2018).
*   **Disciplina:** Sistemas Inteligentes
*   **Entrega:** 01-02 de Julho.
*   **Entregáveis e Pontuação:**
    1.  **Artigo Acadêmico (4.0 pts):** Formato padrão SBC (Sociedade Brasileira de Computação), com limite de 8 páginas.
    2.  **Apresentação Oral (4.0 pts):** 20 minutos de exposição seguidos de 5 minutos de arguição.
    3.  **Repositório Público (2.0 pts):** Código-fonte e artefatos disponíveis no GitHub.

---

## 2. O Problema de Pesquisa e a Base de Dados
O trabalho foca na área da neuro-oncologia, especificamente na classificação de gliomas, um dos tipos mais comuns e agressivos de tumores cerebrais primários.
*   **Dataset:** *TCGA Glioma Dataset* (obtido via UCI Machine Learning Repository, id=759).
*   **Objetivo:** Classificação binária para distinguir entre:
    *   **LGG (Lower-Grade Glioma):** Grau mais baixo, associado a maior sobrevida.
    *   **GBM (Glioblastoma Multiforme):** Grau mais alto, altamente agressivo.
*   **Características dos Dados (839 instâncias e 23 atributos):**
    *   *1 Contínuo:* Idade no diagnóstico (`Age_at_diagnosis`).
    *   *2 Categóricos:* Gênero e Raça.
    *   *20 Binários (Mutações Moleculares):* Presença ou ausência de mutações genéticas críticas (ex: IDH1, TP53, ATRX, PTEN, EGFR, etc.).

---

## 3. Justificativa e Posicionamento (Em Relação à Literatura)
A literatura moderna sobre classificação de gliomas, cujo expoente é o estudo de **Tasci et al. (2022)** (que originou este *dataset*), tem focado na criação de modelos *Ensemble* complexos (votação hierárquica) combinados com funis exaustivos de seleção de características (WOE, RFE, RF, LASSO).

**Nosso Posicionamento:**
Este projeto adota uma filosofia diferente, priorizando a **Interpretabilidade Clínica**. Modelos "caixa-preta" altamente complexos são difíceis de justificar no ambiente médico. 
1.  **Simplicidade:** Avaliamos modelos tradicionais (*KNN, MLP, BernoulliNB*) que se adequam naturalmente a um *dataset* composto em sua vasta maioria por atributos binários.
2.  **Transparência via Análise de Erros:** Ao invés de buscar frações de melhora na acurácia, o projeto foca em "abrir o modelo" na Etapa 6, investigando *onde* o algoritmo erra. Procuramos correlações entre os erros preditivos e o perfil molecular do tumor (especialmente a mutação IDH1).

---

## 4. Pipeline de Desenvolvimento e Metodologia
O código (`Trabalho_Final.ipynb`) foi estruturado em 7 etapas rigorosas:

1.  **Pré-processamento Criterioso:**
    *   `StandardScaler`: Apenas para a Idade, normalizando a escala.
    *   `OneHotEncoder`: Para variáveis demográficas (Gênero e Raça).
    *   `Passthrough`: **Decisão arquitetural crucial** — as mutações moleculares foram deixadas inalteradas (0 e 1) para preservar o significado biológico exato.
2.  **Validação Robusta:**
    *   Divisão Hold-Out clássica de 70% Treino e 30% Teste.
    *   Aplicação de Validação Cruzada Estratificada com 5 Folds (*5-Fold Stratified CV*) usando `GridSearchCV` para encontrar os melhores hiperparâmetros sem vazamento de dados (*data leakage*).
3.  **Modelos Escolhidos:**
    *   **K-Nearest Neighbors (KNN):** Rastreia tumores "vizinhos" com perfil genético similar.
    *   **Multilayer Perceptron (MLP):** Avalia a capacidade de uma rede neural simples mapear interações não lineares entre os genes.
    *   **Bernoulli Naive Bayes (NB):** O modelo matematicamente mais propício para lidar com os 20 atributos booleanos de mutação, trabalhando com probabilidades condicionais puras.

---

## 5. Resultados Obtidos
Os modelos foram avaliados em uma suíte completa de métricas.

| Métrica | KNN | MLP | BernoulliNB |
| :--- | :--- | :--- | :--- |
| **Acurácia** | **0.8730** | 0.8016 | 0.8651 |
| **Precisão** | 0.8246 | **0.8333** | 0.8000 |
| **Recall** | 0.8868 | 0.6604 | **0.9057** |
| **F1-Score** | **0.8545** | 0.7368 | 0.8496 |
| **Kappa** | **0.7421** | 0.5810 | 0.7281 |
| **AUC-ROC** | 0.9038 | 0.8874 | **0.9083** |

**Conclusão de Desempenho:**
*   O **KNN** foi o campeão global, atingindo métricas (87.3% de acurácia) equivalentes aos *ensembles* altamente complexos da literatura (Tasci et al. obteve ~87.6%).
*   O **BernoulliNB** provou ser excepcional na identificação da classe positiva (GBM), obtendo o melhor Recall (90.57%) e a melhor curva AUC-ROC (0.9083), validando a hipótese de que algoritmos probabilísticos adequados a dados booleanos performam incrivelmente bem neste problema.

---

## 6. Análise de Erros e Impacto Biológico
A parte mais rica do projeto. A análise das matrizes de confusão demonstrou comportamentos vitais:

| Modelo | Total Erros | Falsos Positivos | Falsos Negativos | Erros IDH1 Wildtype |
| :--- | :--- | :--- | :--- | :--- |
| **KNN** | 32 | 20 | 12 | 23 |
| **MLP** | 50 | 14 | **36** | 40 |
| **BernoulliNB** | 34 | 24 | **10** | 25 |

*   **Risco Clínico dos Falsos Negativos:** O **MLP** apresentou um Recall baixíssimo (66%) e 36 Falsos Negativos (classificou o grave GBM como o leve LGG). Na clínica, um Falso Negativo é catastrófico, atrasando o tratamento de um câncer letal. O **BernoulliNB** (apenas 10 FNs) seria a escolha clínica mais segura.
*   **O Peso da Mutação IDH1:** Foi descoberto que a vasta maioria dos erros de classificação ocorrem quando o tumor possui **IDH1 Wildtype (valor 0)**. Dos 32 erros do KNN, 23 (71,9%) tinham esse perfil. Isso indica que tumores sem essa mutação (wildtype) são muito mais confusos e atípicos, enganando os algoritmos de aprendizado de máquina.

---

## 7. Próximos Passos (Ação)
Com o contexto e os resultados finalizados, o fluxo de trabalho deve ser:
1.  **Artigo (LaTeX SBC):** Redigir Introdução, Trabalhos Relacionados (já mapeados), Metodologia (detalhando o pipeline), Resultados (inserindo as tabelas acima) e Conclusão (focada na descoberta sobre o IDH1 e o desempenho do KNN/NB).
2.  **Slides:** Criar os recursos visuais focando em 3 pilares: O Problema -> Pipeline/Metodologia -> A "Descoberta" da Análise de Erros.
3.  **Deploy:** Organizar os arquivos e publicá-los no GitHub.
