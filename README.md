###### PT

# Aplicação de algoritmos de ML e análise exploratória no Pima Indians Diabetes Dataset

JOHNNY CLEITON  |  JOSS TIMOTEO  |  RAPHAEL FEITOSA

Este trabalho, desenvolvido no contexto da pós-graduação em *Ciência de Dados e Analytics*, investiga o uso do *Pima Indians Diabetes Dataset* como base para estudo aplicado em saúde. A pesquisa contempla mapeamento de stakeholders, inventário de dados e revisão de trabalhos relacionados, além de práticas fundamentais como pré-processamento, análise exploratória e visualização de dados. Por fim, são aplicados algoritmos de *machine learning*, com destaque para *K-Means* e *Árvore de Decisão*, visando a extração de padrões relevantes e a geração de *insights* que possam contribuir para a compreensão do diabetes.

## BACKGROUND

O *Pima Indians Diabetes Dataset* é um conjunto de dados público amplamente utilizado em aprendizado de máquina para problemas de classificação binária (diabéticos vs não-diabéticos). Ele contém informações de 768 mulheres Pima índias, povo indígena do sudoeste dos EUA, com 21 anos ou mais. Esse grupo tem sido foco de diversos estudos médicos devido à alta prevalência de obesidade e diabetes tipo 2 (não insulino-dependente) entre seus membros.



### Mapeamento de Stakeholders

No contexto do *Pima Indians Diabetes Dataset* e de um trabalho em Ciência de Dados, o mapeamento de *stakeholders* identifica as partes interessadas ou impactadas pelo estudo, ou seja, os “atores” mais relevantes para o projeto, como pacientes, profissionais de saúde e pesquisadores, entre outros.

<div align="center">
  
| ![Matriz de Influência e Interesse](/assets/matriz-influencia-interesse.png) |
|:--:|
| *Matriz de Influência e Interesse* |

</div>

### Inventário de Dados

Esse levantamento permite organizar e documentar todas as fontes de dados disponíveis, incluindo o *Pima Indians Diabetes Dataset*, estatísticas de saúde pública, registros populacionais e estudos acadêmicos relacionados ao diabetes.

- [Diabetes Health Indicators Dataset - BRFSS](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- [Diabetes around the world in 2024 - IDF](https://idf.org/about-diabetes/diabetes-facts-figures/)


### Trabalhos Relacionados

A análise de trabalhos relacionados permite contextualizar o estudo dentro da literatura existente, identificando abordagens, metodologias e resultados já aplicados ao *Pima Indians Diabetes Dataset* e a outros conjuntos de dados sobre diabetes.

- [Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques](https://pubmed.ncbi.nlm.nih.gov/31538566/)
- [A Comparative Study of Diabetes Detection Using The Pima Indian Diabetes Database](https://www.researchgate.net/publication/374730950_A_COMPARATIVE_STUDY_OF_DIABETES_DETECTION_USING_THE_PIMA_INDIAN_DIABETES_DATABASE)


## PREPARAÇÃO

### Dicionário de Dados

<div align="center">

| **Variável**             | **Descrição**                                         | **Tipo de Dado**       | **Valores Possíveis**       |
|---------------------------|-------------------------------------------------------|------------------------|-----------------------------|
| Pregnancies               | Número de vezes que a paciente esteve grávida         | Inteiro                | ≥ 0                         |
| Glucose                   | Concentração de glicose no plasma (teste de tolerância) | Numérico (contínuo)    | ≥ 0 (0 = ausente)           |
| BloodPressure             | Pressão arterial diastólica                           | Numérico (contínuo)    | ≥ 0 (0 = ausente)           |
| SkinThickness             | Espessura da dobra cutânea do tríceps                 | Numérico (contínuo)    | ≥ 0 (0 = ausente)           |
| Insulin                   | Nível de insulina sérica em 2 horas                   | Numérico (contínuo)    | ≥ 0 (0 = ausente)           |
| BMI                       | Índice de Massa Corporal (peso/altura²)               | Numérico (decimal)     | ≥ 0 (0 = ausente)           |
| DiabetesPedigreeFunction  | Função de pedigree de diabetes (risco genético familiar) | Numérico (decimal)   | ~0 a >2                     |
| Age                       | Idade da paciente                                     | Inteiro                | ≥ 0                         |
| Outcome                   | Diagnóstico de diabetes                               | Categórico (binário)   | {0 = não, 1 = sim}          |

</div>

### Atributos Principais

Para o projeto, foram selecionados os atributos principais do dataset: *Glucose* (glicose), *BMI* (índice de massa corporal), *Age* (idade), *DiabetesPedigreeFunction* (histórico familiar) e *Outcome* (resultado). Clinicamente, essas variáveis são fundamentais para o diagnóstico do diabetes, e sua escolha também considera o senso comum, já que, por exemplo, ao observar uma pessoa idosa e com sobrepeso, intuitivamente associamos a maior risco de problemas de saúde.

<div align="center">

| Atributo | Motivação |
|----------|-----------|
| Glucose | Principal indicador de diabetes; valores altos estão fortemente associados à doença. |
| BMI (Índice de Massa Corporal) | Obesidade é um fator de risco bem estabelecido para diabetes. |
| Age (Idade) | A probabilidade de diabetes aumenta com a idade. |
| DiabetesPedigreeFunction (Histórico familiar) | Mede predisposição genética; ajuda a capturar risco familiar. |
| Outcome (Resultado) | Representa o diagnóstico de diabetes (0 = não, 1 = sim). |

</div>

### Pré-processamento de Dados

Para esta e as demais etapas, utilizamos Python com bibliotecas de Ciência de Dados (pandas, numpy, matplotlib, plotly e scikit-learn). O primeiro passo do pré-processamento dos dados foi realizar a redução dos atributos para considerar apenas aqueles que foram selecionados como principais para o projeto. Para a etapa de limpeza, foi verificado que os atributos *"Glucose (glicose)"* e *"BMI (Índice de Massa Corporal)"* estavam preenchidos com valor zero em algumas linhas, o que não faz sentido dentro do contexto clínico. Para isso, os valores 0 foram substituídos por *NaN*, permitindo posteriormente a remoção de todas as linhas com valores ausentes do dataset. Com isso o dataset inicial com 768 linhas e 9 colunas passou a ter 752 linhas e 5 colunas.

```python
# substitui zeros inválidos por NaN
df[["Glucose", "BMI"]] = df[["Glucose", "BMI"]].replace(0, np.nan)

# remove linhas com qualquer valor nulo nas variáveis selecionadas
df = df.dropna()
```

Ainda no pré-processamento, foi realizada uma etapa essencial: a transformação das variáveis numéricas, garantindo que todos os atributos estivessem padronizados na mesma escala. Isso porque, enquanto a idade pode variar até cerca de 80 anos, o DiabetesPedigreeFunction (histórico familiar) possui valores entre 0 e 2. Sem essa padronização, os atributos não seriam tratados de forma equilibrada no modelo. Nessa tarefa foi utilizada a classe ```StandardScaler``` (do scikit-learn) que serve para padronizar estatísticamente as variáveis numéricas, ou seja, usa da fórmula *Z-score* (ou escore-padrão) transformando os dados para que tenham média 0 e desvio padrão 1.

```python
# seleciona do DataFrame apenas as variáveis independentes que se deseja escalar
X = df[["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]]

X_scaled = scaler.fit_transform(X)
# fit: calcula para cada coluna a média (𝜇) e o desvio padrão (σ) a partir dos dados fornecidos
# transform: aplica a padronização usando a fórmula abaixo para cada valor
```

$$
z = \frac{x - \mu}{\sigma}
$$


## ANÁLISE DESCRITIVA

Na etapa de análise descritiva, os dados são explorados e resumidos para compreender suas características principais, identificar padrões e detectar possíveis inconsistências. Essa fase inclui estatísticas básicas, distribuição de valores, correlações entre variáveis e visualizações gráficas, servindo como base para decisões posteriores no projeto.

### Histograma

<div align="center">
  
| ![Histograma](/assets/histograma.png) |
|:--:|
| *Histograma de distribuição sobreposta (stacked/overlapped histogram)* |

</div>

Esse gráfico mostra a distribuição dos valores de glicose no conjunto de dados, separados de acordo com o diagnóstico de diabetes (Outcome).

- A cor vermelha representa pessoas sem diabetes (Outcome = 0), cuja maior concentração de valores está em torno de 90 a 110 mg/dL, região considerada mais próxima da normalidade.
- A cor azul representa pessoas com diabetes (Outcome = 1), que se concentram mais em valores acima de 120 mg/dL, indicando níveis elevados de glicose.

Em resumo, o gráfico evidencia que valores mais altos de glicose estão fortemente associados ao diagnóstico positivo de diabetes, enquanto valores mais baixos predominam entre os indivíduos sem a doença.



### Gráfico de Dispersão

<div align="center">
  
| ![Gráfico de Dispersão](/assets/scatterplot.png) |
|:--:|
| *Gráfico de Dispersão (scatter plot)* |

</div>

O que ele mostra:
- Eixo X (horizontal): valores de BMI (índice de massa corporal).
- Eixo Y (vertical): valores de Glicose.
- Cor dos pontos (cmap): variável categórica Outcome (0 = azul escuro, 1 = amarelo).

O gráfico mostra que pessoas sem diabetes (azul escuro) concentram-se em níveis mais baixos de glicose, mesmo com variações de IMC, enquanto aquelas com diabetes (amarelo) apresentam valores mais elevados de glicose, independentemente do IMC, indicando que a glicose é o principal fator diferenciador entre os grupos, embora o IMC também contribua para o risco.



### Boxplot

<div align="center">
  
| ![Boxplot](/assets/boxplot.png) |
|:--:|
| *Boxplot* |

</div>

O que se observa:
- Para quem não tem diabetes (Outcome = 0, vermelho), a mediana da idade está em torno de 27 anos, com a maioria concentrada entre aproximadamente 22 e 36 anos. Há vários outliers em idades mais altas.
- Para quem tem diabetes (Outcome = 1, azul), a mediana é mais alta, por volta de 36 anos, e a faixa interquartílica (25% a 75%) vai de cerca de 28 a 45 anos, mostrando que indivíduos com diabetes tendem a ser mais velhos.
- A dispersão é maior entre os diagnosticados com diabetes, mas fica evidente que a idade elevada está mais associada à presença da doença.

Em resumo, o gráfico indica que a idade é um fator relevante no risco de diabetes, com maior prevalência em pessoas mais velhas.



## MODELAGEM

Nesta etapa de modelagem, serão aplicados diferentes algoritmos de **Machine Learning** ao **Pima Indians Diabetes Dataset** com o intuito de gerar insights sobre os padrões presentes nos dados. Serão utilizados dois métodos, um supervisionado (Árvore de Decisão) e outro não-supervisionado (K-Means).

### Árvore de Decisão (Decision Tree)

A **árvore de decisão** é um algoritmo de **aprendizado supervisionado** usado para classificação e regressão. Ela divide os dados em ramos com base nas variáveis mais relevantes, formando uma estrutura em forma de árvore. Cada divisão representa uma decisão, e as folhas finais indicam o resultado ou a previsão do modelo.

#### Configuração da Árvore de Decisão

Para o código, foi utilizada a biblioteca ```Scikit-learn```, que fornece o módulo ```DecisionTreeClassifier```, responsável pela construção, treinamento e avaliação da Árvore de Decisão. O modelo é treinado com parte dos dados (treino) e avaliado em dados novos (teste). A função ```train_test_split``` divide os dados de forma aleatória, mantendo a proporção das classes com o parâmetro ```stratify```, destinando 30% para teste e 70% para treino. O parâmetro ```random_state=42``` garante que a divisão seja sempre a mesma, sendo 42 apenas um número de referência.

```python
# criação do modelo de árvore de decisão
clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_leaf=20,
    random_state=42
)
```


- `criterion="gini"` → a árvore usa o **índice de Gini** (ou, alternativamente, "entropy") para medir a impureza do nó.  
- `max_depth=3` → limita a árvore a 3 níveis, evitando complexidade excessiva.  
- `min_samples_leaf=20` → cada folha precisa ter pelo menos 20 exemplos, prevenindo overfitting.

#### Índice de Gini

O **índice de Gini** mede o quão misturado está um nó; e é calculado como:

$$
Gini = 1 - \sum_{i=1}^{n} p_i^2
$$

onde $$p_i$$ é a proporção de elementos da classe $$i$$ no nó. Quanto mais próximo de 0, mais puro é o nó; quanto mais próximo de 1, mais misturado estão os dados.

#### Visualização da Árvore

<div align="center">
  
| ![Árvore de Decisão (Decision Tree)](/assets/arvore-de-decisao.png) |
|:--:|
| *Árvore de Decisão (Decision Tree)* |

</div>

- `feature <= valor` → condição de decisão
- `gini` → medida de impureza (0 = puro, 0.5 = misto)
- `samples` → quantos exemplos chegaram até esse nó
- `value = [a, b]` → número de amostras das classes [Não Diabético, Diabético]
- `class = ...` → decisão da árvore naquele nó

A árvore de decisão divide os dados em etapas para separar os grupos “Diabético” e “Não Diabético”. Ela começa avaliando o nível de glicose, o fator mais determinante, e depois refina a decisão com idade, IMC e histórico familiar. Cada nó mostra como o modelo reduz a incerteza até chegar às folhas, onde estão as previsões finais. Os tons alaranjados representam não diabéticos e os tons azulados, diabéticos — quanto mais escura a cor, mais certa é a classificação. Além disso, a idade se mostrou um divisor claro, pois nenhum indivíduo com menos de "27,5" anos foi classificado como diabético, deixando todo esse ramo da árvore composto apenas por nós com resultados “Não Diabético”.

| Padrão | Interpretação |
|------------|----------------|
| **Alta glicose** | É o fator mais decisivo → indica fortemente diabetes. |
| **Baixa glicose + baixo BMI + jovem** | Quase sempre não diabético. |
| **Histórico familiar alto (Pedigree alto)** | Aumenta bastante a chance de diabetes mesmo com glicose não tão alta. |

### K-Means

O K-means é um algoritmo de aprendizado não supervisionado usado para agrupar dados semelhantes em K grupos (clusters). Ele funciona atribuindo cada ponto ao centro mais próximo e ajustando esses centros iterativamente até que os grupos fiquem estáveis. É considerado o algoritmo de clusterização mais simples e amplamente utilizado em **Machine Learning**, sendo ideal para descobrir padrões ou segmentos ocultos em conjuntos de dados sem rótulos.

Para o código, foi utilizada a biblioteca ```Scikit-learn```, que fornece o módulo ```KMeans```, responsável pela realização da clusterização não supervisionada. Diferente dos modelos de classificação, o K-Means não utiliza rótulos (classes) durante o treinamento, pois seu objetivo é agrupar os dados com base em similaridades entre as variáveis. O modelo foi configurado para criar dois agrupamentos ```(n_clusters=2)```, e o parâmetro ```random_state=42``` foi definido para assegurar a reprodutibilidade dos resultados. Após o ajuste do modelo, cada amostra recebeu um rótulo de cluster, permitindo a visualização e análise dos grupos formados.

#### Visualização do Resultado

Para os agrupamentos, escolheu-se K = 2 porque o dataset naturalmente possui duas categorias (diabético e não diabético), e o objetivo é verificar se o K-Means consegue reproduzir essa separação de forma não supervisionada. Abaixo, é mostrado o gráfico de dispersão em função de duas variáveis importantes: glicose e índice de massa corporal.

<div align="center">
  
| ![Gráfico de Dispersão do K-Means](/assets/k-means-scatterplot.png) |
|:--:|
| *K-Means Scatterplot com K = 2* |

</div>


O gráfico revela a formação de dois grupos principais:
- Cluster 1 (roxo): agrupa indivíduos com valores mais baixos de glicose e BMI moderado, indicando um perfil mais próximo de pacientes sem diabetes ou com menor risco metabólico.
- Cluster 2 (amarelo): agrupa indivíduos com níveis de glicose mais elevados e, em geral, maior índice de massa corporal, o que pode estar associado a um maior risco de diabetes.

#### Validação

Para avaliar se o número de clusters definido (K = 2) é adequado, foi aplicada a técnica de Índice da Silhueta (Silhouette Score).

<div align="center">
  
| ![Índice da Silhueta (Silhouette Score](/assets/silhouette-score-kmeans.png) |
|:--:|
| *Índice da Silhueta (Silhouette Score)* |

</div>

Interpretação do gráfico:
- O eixo X (embaixo) é o número de clusters (K) testado, nesse caso indo de 2 até 8.
- O eixo Y esquerdo mostra o *Silhouette Score* que é a linha azul.
- A linha azul representa o *Silhouette Score* para cada número de clusters (K), o valor mostra o quão bem os pontos estão agrupados, pois quanto maior o *Silhouette Score*, melhor o agrupamento.
- A linha verde ao fundo mostra o tempo que o algoritmo demorou para treinar *(fit time)*, ou seja, é apenas um indicador de desempenho computacional e não influencia na escolha do melhor K.

**Resultado:** o gráfico mostra que o maior valor de *Silhouette Score* ocorre em K = 2, com pontuação 0.496. Depois desse ponto o *score* cai bastante, o que significa que os agrupamentos ficaram menos bem definidos à medida que aumentamos o número de clusters. Em resumo, o melhor K é aquele onde o *Silhouette Score* é máximo, porque significa que os grupos estão mais bem separados e coesos internamente. Nesse caso K = 2 realmente é o ideal.

---

## Informação Adicional

O código completo pode ser conferido através do Google Colab neste repositório.


