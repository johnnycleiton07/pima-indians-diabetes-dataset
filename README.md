###### PT

# Aplicação de algoritmos de ML e análise exploratória no Pima Indians Diabetes Database

JOHNNY CLEITON  |  JOSS TIMOTEO  |  RAPHAEL FEITOSA

Este trabalho, desenvolvido no contexto da pós-graduação em *Ciência de Dados e Analytics*, investiga o uso do *Pima Indians Diabetes Database* como base para estudo aplicado em saúde. A pesquisa contempla mapeamento de stakeholders, inventário de dados e revisão de trabalhos relacionados, além de práticas fundamentais como pré-processamento, análise exploratória e visualização de dados. Por fim, são aplicados algoritmos de *machine learning*, com destaque para K-Means e Árvore de Decisão, visando a extração de padrões relevantes e a geração de *insights* que possam contribuir para a compreensão do diabetes.

## BACKGROUND

O *Pima Indians Diabetes Database* é um conjunto de dados público amplamente utilizado em aprendizado de máquina para problemas de classificação binária (diabéticos vs não-diabéticos). Ele contém informações de 768 mulheres Pima índias, povo indígena do sudoeste dos EUA, com 21 anos ou mais. Esse grupo tem sido foco de diversos estudos médicos devido à alta prevalência de obesidade e diabetes tipo 2 (não insulino-dependente) entre seus membros.

<div align="center">
  
| ![Pima Indians Diabetes Database](/assets/povo-pima.png) |
|:--:|
| *Os Pima também são conhecidos como Akimel O'odham, que significa "povo do rio"* |

</div>

### Mapeamento de Stakeholders

No contexto do Pima Indians Diabetes Database e de um trabalho em Ciência de Dados, o mapeamento de stakeholders identifica as partes interessadas ou impactadas pelo estudo, ou seja, os “atores” mais relevantes para o projeto, como pacientes, profissionais de saúde e pesquisadores, entre outros.

<div align="center">
  
| ![Matriz de Influência e Interesse](/assets/matriz-influencia-interesse.png) |
|:--:|
| *Matriz de Influência e Interesse* |

</div>

### Inventário de Dados

Esse levantamento permite organizar e documentar todas as fontes de dados disponíveis, incluindo o *Pima Indians Diabetes Database*, estatísticas de saúde pública, registros populacionais e estudos acadêmicos relacionados ao diabetes.

- [Diabetes Health Indicators Dataset - BRFSS](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- [Diabetes around the world in 2024 - IDF](https://idf.org/about-diabetes/diabetes-facts-figures/)


### Trabalhos Relacionados

A análise de trabalhos relacionados permite contextualizar o estudo dentro da literatura existente, identificando abordagens, metodologias e resultados já aplicados ao *Pima Indians Diabetes Database* e a outros conjuntos de dados sobre diabetes.

- [Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques](https://pubmed.ncbi.nlm.nih.gov/31538566/)
- [A Comparative Study of Diabetes Detection Using The Pima Indian Diabetes Database](https://www.researchgate.net/publication/374730950_A_COMPARATIVE_STUDY_OF_DIABETES_DETECTION_USING_THE_PIMA_INDIAN_DIABETES_DATABASE)


## PREPARAÇÃO

### Dicionário de Dados e Atributos Principais

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


