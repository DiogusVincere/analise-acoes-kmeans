# Análise de ações com K-means

Trabalho somativo de IA e Big Data (dezembro de 2024): análise exploratória e agrupamento
não supervisionado de um conjunto de ações fictícias com K-means, no notebook `codigo.ipynb`.

## O que foi feito
1. **Carga e exploração** — `dadosacoes.csv` (100 linhas, 4 colunas: nome da ação, preço, quantidade de cotas e valor de mercado em bilhões); `head()`, `info()`, `describe()` e checagem de nulos; boxplots (seaborn) do preço e do valor de mercado por ação.
2. **Pré-processamento** — `pd.get_dummies(drop_first=True)` na coluna de nome da ação (gera 6 colunas booleanas para 7 ações); `fillna` pela média e `dropna` (não havia nulos).
3. **K-means** — um primeiro ajuste com 4 grupos direto nos valores brutos; depois padronização com `StandardScaler` e K-means com 4, 5 e 8 grupos (`random_state=42`), usando preço, cotas e valor de mercado.
4. **Escolha de k** — gráfico do cotovelo (inércia para k de 1 a 8) e silhueta para k = 4 (score médio e histograma dos valores por ponto).
5. **Visualização** — dispersão 2D (preço × valor de mercado, colorida por grupo) e 3D (preço × cotas × valor de mercado), além da 3D do primeiro ajuste.
6. **Questão 5** — resposta sobre a vantagem do aprendizado não supervisionado, em célula markdown no fim do notebook.

## Resultados principais (lidos das saídas do notebook)
- Dados: 100 ações de 7 nomes (GOOGLE 25, VALE3 15, SUZB3 14, ELEKTRO 13, APPLE 12, AMAZON 11, PETR4 10). Preço médio R$ 136,14 (de 15 a 253); cotas em média 52 (de 2 a 97); valor de mercado médio 2.433,7 bilhões (de 52 a 4.993).
- K-means sem padronizar (k = 4): os grupos se separam quase só pelo valor de mercado — médias de 4.264, 3.137, 2.101 e 554 bilhões — enquanto preço e cotas ficam parecidos entre os grupos (preço médio entre 102 e 153).
- Silhueta para k = 4 nos dados padronizados: **0,303**, ou seja, grupos pouco separados.
- O cotovelo é calculado para k de 1 a 8; os rótulos de 5 e 8 grupos ficam em `Cluster_5` e `Cluster_8`, mas o notebook não compara métricas entre os valores de k.

## Observações honestas
- O conjunto é sintético e pequeno (100 linhas): a mesma ação aparece com preços e valores de mercado muito diferentes (SUZB3 a R$ 91, 165 e 56), o que não acontece em dados reais. Os grupos refletem o sorteio dos dados, não o mercado.
- O primeiro K-means roda **sem padronizar**: o valor de mercado está em milhares e o preço em centenas, então a distância é dominada pelo valor de mercado e os grupos daquele ajuste são faixas de valor de mercado. O ajuste seguinte, com `StandardScaler`, corrige isso; a silhueta e as visualizações 2D/3D finais usam o padronizado.
- Silhueta de 0,30 é fraca, e o k = 4 vem do enunciado, não de uma escolha pelos gráficos.
- Os dummies do nome da ação são criados, mas não entram no K-means (as features são só as três colunas numéricas) — o que é adequado, só não está dito no notebook.
- `KMeans` sem `n_init` explícito gera `FutureWarning` no scikit-learn 1.2/1.3; em versões novas o padrão já é `n_init='auto'`.
- `fillna`/`dropna` não mudam nada (não há nulos), e a numeração das questões no notebook (6) difere do enunciado (5, repetido).

## Como rodar
```bash
pip install -r requirements.txt
jupyter notebook codigo.ipynb
```
Rode as células em ordem; o CSV precisa estar na mesma pasta do notebook.

## Enunciado original

Você foi contratado pela empresa S&M Data Analytics para realizar a análise de um determinado
conjunto de dados de uma companhia de investimentos.
Nesse conjunto de dados há diversas informações sobre os ativos de investimentos Nome da ação,
preço da ação, qtde de cotas, valor de mercado da empresa
Sua tarefa consiste em realizar análise exploratória e implementar um algoritmo de aprendizado não
supervisionado que consiste em agrupar as ações com características semelhantes. Para essa tarefa são
necessários realizar as seguintes etapas do processo de descoberta do conhecimento:

1 - Preparação dos dados
Realize a importação de um conjunto de dados que contenha as seguintes colunas:
Nome ação: nome dação
Preço ação: Preço para compra da ação.
Qtde de cotas: Quantidade de cotas disponíveis da empresa.
Valor de mercado da empresa: Valor de mercado da empresa

2- Exploração dos dados
Explore os dados para entender a distribuição do preço da ação, qtde de cotas, valor de mercado
Plotar box plot do preço da ação para cada ativo para verificar outliers (Utilizar biblioteca seaborn)
a) Plotar box plot do valor de mercado da empresa para cada ativo para verificar outliers(Utilizar
biblioteca seaborn - sns.boxplot)
b) Utilizar df.info() para ver informações do conjunto de dados
c) Utilizar df.describe para ver informações do conjunto de dados

3 - Pré processamento dos dados
Realize qualquer pré-processamento necessário, como lidar com valores ausentes, codicar
variáveis categóricas, etc.
Utilizar o pdget_dummies(nome_dataframe,columns=[‘nome da coluna’],drop_rst=True) para
transformar a coluna de variáveis categóricas para True ou false para aplicar o algoritmo K-means
4 - Utilizar o algoritmo K-means com n_clusters = 4 para agrupar as ações com base nas
características :preço da ação, quantidade de cotas e valor de mercado,
a) Alterar o número de clusters para 5 e verificar o que acontece com os grupos formados
b) Alterar o número de clusters para 8 e verificar o que acontece com os grupos formados
c) Plotar o gráfico do cotovelo para n variando de 1 a 8 clusters
d) Plotar o gráfico da silhueta
5 - Plotar visualização dos clusters formados
a) Visualização 2d dos clusters formados
b) Visualização 3d dos clusters formados
5) Qual a maior vantagem do aprendizado não supervisionado diante o aprendizado supervisionado ?

## Licença

MIT — ver [LICENSE](LICENSE).
