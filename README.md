# Projeto: predição de risco de acidente vascular cerebral (AVC)

O acidente vascular cerebral (AVC) é uma condição médica grave que ocorre quando há interrupção ou redução do fluxo sanguíneo para uma parte do cérebro, resultando em danos às células cerebrais. Representa uma das principais causas de morte no mundo e pode ter consequências graves, tanto físicas quanto cognitivas. Diante de sua alta incidência e impacto, a previsão do risco de desenvolvimento de AVC torna-se essencial para a adoção de medidas preventivas eficazes. Identificar precocemente indivíduos em maior risco permite intervenções clínicas e mudanças no estilo de vida que podem reduzir a ocorrência de novos casos e melhorar a qualidade de vida da população.

O conjunto de dados utilizado neste projeto foi construído com base em literatura médica, consultas a especialistas e modelagem estatística. As distribuições e relações de características foram inspiradas em observações clínicas do mundo real, garantindo validade médica. Nele há 70.000 registros e 18 características relacionadas a fatores de risco para desfecho de acidente vascular cerebral.

Fonte dos dados: [Kaggle](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset)

[Notebooks desenvolvidos no projeto](./notebooks/)

## Resultados principais

Além da realização de [análise exploratória](./notebooks/projeto_risco_avc_02_analise_exploratoria.ipynb), foram criados modelos de machine learning de [classificação](./notebooks/projeto_risco_avc_03_classificacao.ipynb) e de [regressão](./notebooks/projeto_risco_avc_04_regressao.ipynb). Ambos mostram a porcentagem de chance de AVC, porém obtidas de formas diferentes. O modelo de classificação exibe essa porcentagem como probabilidade de pertencer à classe "Sim" para risco de AVC e os modelos de regressão mostram diretamente essa porcentagem a partir do treinamento com a porcentagem de risco presente no dataset.

Foram testados modelos lineares, de árvore e de distância entre os pontos. Tanto para classificação quanto para regressão o melhor modelo foi o LightGBM. Adicionalmente, apenas para estudo, também foi criado um modelo de regressão com deep learning, que apresenta um acerto similar ao LightGBM.

Os resultados obtidos neste projeto possuem apenas finalidade de aprendizado sobre criação de modelos de machine learning.

Os modelos estão [disponíveis aqui](./modelos/).

## Organização do projeto

```
├── LICENSE            <- Licença de código aberto.
├── README.md          <- Instruções e detalhes do projeto.
|
├── dados              <- Arquivos de fontes de dados para o projeto.
|
├── modelos            <- Arquivos de modelos de ML criados no projeto.
|
├── notebooks          <- Cadernos Jupyter. A numeração indica a ordem em que as etapas foram executadas.
│
|   └──src             <- Dados centralizados organizados para uso neste projeto.
|      │
|      ├── __init__.py      <- Torna um módulo Python.
|      ├── clf_graphics.py  <- Variáveis e funções para gráficos utilizadas no projeto de classificação.
|      ├── clf_models.py    <- Variáveis e funções para modelos de ML utilizadas no projeto de classificação.
|      ├── config.py        <- Configurações básicas de caminhos de arquivos do projeto.
|      ├── reg_graphics.py  <- Variáveis e funções para gráficos utilizadas no projeto de regressão.
|      ├── reg_models.py    <- Variáveis e funções para modelos de ML utilizadas no projeto de regressão.
|
```
