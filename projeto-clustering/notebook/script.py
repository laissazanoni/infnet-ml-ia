
# # Projeto da Disciplina Algoritmos Não-Supervisionados para Clusterização
#
# - **Nome:** Laíssa Medeiros
# - **Data:** 20/11/2022
# - **Professor:** Luiz Frias
#
# ## Contexto
# A HELP International é uma ONG humanitária internacional que está empenhada em combater a pobreza e fornecer ajuda e ​​serviços básicos à população de países subdesenvolvidos, durante o período de desastres e calamidades naturais.
# A ONG já conseguiu arrecadar cerca de US$ 10 milhões e agora o CEO precisa decidir como usar esse dinheiro de forma estratégica e eficaz. Portanto, você precisa sugerir os países nos quais o CEO precisa se concentrar e ajudá-lo a tomar a melhor decisão.
#
# ## Objetivo
# Utilizar os algoritmos de KMeans e Clusterização Hierárquica para agrupar os países utilizando os fatores socioeconômicos e de saúde que determinam o desenvolvimento geral dos países. Interpretar os resultados e identificar as semelhanças e diferenças entre os dois modelos.
#
# ## Dataset
# Foi utilizado o dataset [Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data?select=data-dictionary.csv) do Kaggle, no qual cada linha do dataset representa um país e existem 9 features que representam indicadores socioeconômicos e de saúde que determinam o desenvolvimento geral de cada país.
#
# - **country:** nome do país
# - **child_mort:** morte de crianças menores de 5 anos por 1000 nascidos vivos
# - **exports:** exportações de bens e serviços per capita. Dado como % da idade do PIB per capita
# - **health:** gasto total com saúde per capita. Dado como % de idade do PIB per capita
# - **imports:** importações de bens e serviços per capita. Dado como % de idade do PIB per capita
# - **Income:** renda líquida por pessoa
# - **Inflation:**  medida da taxa de crescimento anual do PIB total
# - **life_expec:** número médio de anos que um recém-nascido viveria se os padrões de mortalidade atuais permanecessem os mesmos
# - **total_fer:** número de filhos que nasceriam de cada mulher se as taxas atuais de fertilidade por idade permanecessem as mesmas
# - **gdpp:** o PIB per capita. Calculado como o PIB total dividido pela população total


# # 0. Imports


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.core.display import HTML
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch


# ## 0.1. Helper Functions


def distribution(data):

    cols = ['child_mort', 'exports', 'health', 'imports', 'income',
            'inflation', 'life_expec', 'total_fer', 'gdpp']

    df = data[cols].copy()

    # tendencia central - media, mediana
    t1 = pd.DataFrame(df.mean()).T
    t2 = pd.DataFrame(df.median()).T

    # dispersion - std, min, max, range, skew, kurtosis
    d1 = pd.DataFrame(df.std()).T
    d2 = pd.DataFrame(df.min()).T
    d3 = pd.DataFrame(df.max()).T
    d4 = pd.DataFrame(df.apply(lambda x: x.max() - x.min())).T
    d5 = pd.DataFrame(df.skew()).T
    d6 = pd.DataFrame(df.kurtosis()).T

    # concat
    m1 = pd.DataFrame()
    m1 = pd.concat([d2, d3, d4, t1, t2, d1, d5, d6]).T.reset_index()
    m1.columns = ['attributes', 'min', 'max', 'range',
                  'media', 'mediana', 'std', 'skew', 'kurtosis']
    display(m1)


def plot_inertia(df, kmin=1, kmax=10, figsize=(8, 4)):

    _range = range(kmin, kmax)
    inertias = []
    for k in _range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        inertias.append(kmeans.inertia_)

    plt.rcParams['figure.figsize'] = (20, 5)
    plt.plot(_range, inertias, 'yx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.show()


def plot_cluster_points(df, labels, title='', ax=None):

    pca = PCA(2)
    pca_data = pd.DataFrame(pca.fit_transform(df), columns=['PC1', 'PC2'])
    pca_data['cluster'] = pd.Categorical(labels)
    sns.scatterplot(x="PC1", y="PC2", hue="cluster",
                    data=pca_data, palette='deep', ax=ax).set(title=title)


# ## 0.2. Loading Data


df_raw = pd.read_csv('../dataset/country-data.csv', low_memory=True)
df_raw.shape


df_raw.head()


# # 1. Data Description


df1 = df_raw.copy()


# ## 1.1 Data Dimensions


print('Number of rows: {}'.format(df1.shape[0]))
print('Number of columns: {}'.format(df1.shape[1]))


# Existe um total de 167 países no dataset.


# ## 1.2. NA Check


df1.isna().sum()


# ## 1.3. Data Type


df1.dtypes


# income
df1['income'] = df1['income'].astype(np.float64)

# gdpp
df1['gdpp'] = df1['gdpp'].astype(np.float64)


df1.dtypes


# ## 1.4. Check Balanced Data


df1['country'].value_counts(normalize=True)


# ## 1.5 Descriptive Statistics


num_attributes = df1.select_dtypes(include=['int64', 'float64'])
cat_attributes = df1.select_dtypes(
    exclude=['int64', 'float64', 'datetime64[ns]'])


distribution(num_attributes)


# - **Min**: menor valor do conjunto de dados
# - **Max**: maior valor do conjunto de dados
# - **Range**: diferença entre o maior e o menor valor do conjutno de dados
# - **Média**: resume o conjunto de dados em um ponto central
# - **Mediana**: representa o número do meio de uma lista ordenada
# - **Desvio Padrao**: estimativa de dispersão ou variabilidade. Mede a dispersão dos valores do conjunto de dados em torno da média
# - **Skewness**: medida de assimetria da distruição de dados, em relação a uma distribuição normal
#   - Mediana > Média: Deslocamento para a esquerda = Skewness Negativa
#   - Mediana < Média: Deslocamento para a direita = Skewness Positiva
# - **Kurtosis**: evidencia as caudas longas, probabilidade de acontecer valor extremos, seja mínimos ou máximos
#
#


# # 2. Exploratory Data Analysis (EDA)


df2 = df1.copy()


# Distribuições

plt.rcParams['figure.figsize'] = (30, 10)
num_attributes.hist(bins=25)

# Nenhuma feature possui distribuição normal


# Outliers

n = 1
plt.figure(figsize=(30, 20))
for col in num_attributes.columns:
    plt.subplot(5, 3, n)
    n = n + 1
    sns.boxplot(x=col, data=df2)


# Todas as features apresentam outliers


plt.rcParams['figure.figsize'] = (30, 10)

correlation = num_attributes.corr(method='pearson')
sns.heatmap(correlation, annot=True)


# Observamos uma correlação positiva entre as features gdpp e income
# Observamos uma correlação negativa entre as features child_mort e life_expect; total_fer e life_expect


g = sns.PairGrid(num_attributes, vars=num_attributes.columns)
g.map_diag(sns.kdeplot, lw=3)
g.map_offdiag(plt.scatter)
g.add_legend()


# # 3. Data Preparation


df3 = df2.copy()


# Regra para normalização dos dados:
# 1. **Standard Scaler**: features que possuem distribuiçåo normal e não possuem outlier
# 2. **Robust Scaler**: features que possuem distribuiçåo normal e possuem outlier
# 3. **Min Max Scaler**: features que não possuem distribuição normal


mms = MinMaxScaler()

# child_mort
df3['child_mort'] = mms.fit_transform(df3[['child_mort']].values)

# exports
df3['exports'] = mms.fit_transform(df3[['exports']].values)

# health
df3['health'] = mms.fit_transform(df3[['health']].values)

# imports
df3['imports'] = mms.fit_transform(df3[['imports']].values)

# income
df3['income'] = mms.fit_transform(df3[['income']].values)

# inflation
df3['inflation'] = mms.fit_transform(df3[['inflation']].values)

# life_expec
df3['life_expec'] = mms.fit_transform(df3[['life_expec']].values)

# total_fer
df3['total_fer'] = mms.fit_transform(df3[['total_fer']].values)

# gdpp
df3['gdpp'] = mms.fit_transform(df3[['gdpp']].values)


df3.head()


# # 5. Machine Learning Model - Clustering


df4 = df3.copy()
df4 = df4.set_index('country')


# ## 5.1 K-Means


# ### 5.1.1 Finding the number of clusters


plot_inertia(df4)


# ### 5.1.2 Clustering


model_kmeans = KMeans(n_clusters=3, random_state=1234)
k_fit = model_kmeans.fit(df4)


# centróide de cada cluster
k_fit.cluster_centers_


# cluster encontrado para cada ponto do dataset
clusters = k_fit.labels_
clusters


# ### 5.1.3 Cluster Analysis


df_result = df2.copy()
df_result['kmeans_cluster'] = clusters
df_result.head()


# visualização dos clusters
plot_cluster_points(df4, model_kmeans.labels_, title='K-Means')


# distribuição dos clusters
i = 1
for i in range(0, 3):
    print("Distribuição das features do Cluster " + str(i))
    distribution(df_result.loc[df_result['kmeans_cluster'] == i])
    i = i + 1


plt.figure(figsize=(20, 20))

cols = ['child_mort', 'exports', 'health', 'imports',
        'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

for i in range(len(cols)):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=df_result['kmeans_cluster'], y=df_result[cols[i]])
    plt.title(cols[i])


# o país que melhor representa o seu agrupamento, será o país mais próximo ao centróide de seu cluster

df_aux = df3.copy()
df_aux['kmeans_cluster'] = clusters

cols = ['child_mort', 'exports', 'health', 'imports',
        'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']
i = 1
for i in range(0, 3):

    df_cluster = df_aux[df_aux['kmeans_cluster'] == i]
    df_aux2 = df_cluster[cols].copy()

    df_aux2['distance'] = df_aux2.apply(
        lambda x: np.linalg.norm(x - k_fit.cluster_centers_[0]), axis=1)

    print("País que melhor representa o Cluster " + str(i) + ": " +
          df_cluster.loc[df_aux2['distance'].idxmin(), 'country'])


# A partir das distribuições das features para cada cluster, pode-se classificar os 3 grupos de países em:
# - **Cluster 0**: Emergentes
# - **Cluster 1**: Subdesenvolvidos
# - **Cluster 2**: Desenvolvidos


df_result['kmeans_class'] = ""
df_result.loc[df_result['kmeans_cluster']
              == 2, 'kmeans_class'] = 'Desenvolvido'
df_result.loc[df_result['kmeans_cluster'] ==
              1, 'kmeans_class'] = 'Subdesenvolvido'
df_result.loc[df_result['kmeans_cluster'] == 0, 'kmeans_class'] = 'Emergente'


fig = px.choropleth(df_result[['country', 'kmeans_class']],
                    locationmode='country names',
                    locations='country',
                    color=df_result['kmeans_class'],
                    color_discrete_map={
                        'Desenvolvido': 'blue', 'Emergente': 'goldenrod', 'Subdesenvolvido': 'red'},
                    title='KMeans Clustered Countries'
                    )

fig.update_geos(fitbounds="locations",
                visible=False
                )

fig.update_layout(mapbox_style="carto-positron",
                  height=600,
                  width=1000,
                  margin={"r": 0, "t": 0, "l": 0, "b": 0},
                  title_x=0.02,
                  title_y=0.98,
                  legend_title_text='Country',
                  geo=dict(
                      showframe=False,
                      showcoastlines=True,
                      projection_type='equirectangular',
                  ),
                  )

fig.show()


# ## 5.2 Hierarchical Clustering


# ### 5.2.1 Dendogram


df5 = df2.copy()
df5 = df5.set_index('country')


plt.figure(figsize=(30, 10))
plt.grid(False)

dendrogram = sch.dendrogram(sch.linkage(df5, method='ward'), labels=df5.index)

plt.title('Dendrogram')
plt.ylabel('Euclidean Distance')


# ### 5.2.2 Clustering


model_agg = AgglomerativeClustering(n_clusters=3)
model_agg.fit(df4)


df_result['hier_cluster'] = model_agg.labels_
df_result.head()


# ### 5.2.3 Cluster Analysis


# visualização dos clusters
plot_cluster_points(df4, model_agg.labels_, title='Hierarchical Clustering')


plt.figure(figsize=(20, 20))

cols = ['child_mort', 'exports', 'health', 'imports',
        'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

for i in range(len(cols)):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=df_result['hier_cluster'], y=df_result[cols[i]])
    plt.title(cols[i])


# A partir das distribuições das features para cada cluster, pode-se classificar os 3 grupos de países em:
# - **Cluster 0**: Desenvolvido
# - **Cluster 1**: Subdesenvolvidos
# - **Cluster 2**: Emergente


df_result['hier_class'] = ""
df_result.loc[df_result['hier_cluster'] == 0, 'hier_class'] = 'Desenvolvido'
df_result.loc[df_result['hier_cluster'] == 1, 'hier_class'] = 'Subdesenvolvido'
df_result.loc[df_result['hier_cluster'] == 2, 'hier_class'] = 'Emergente'


# # 6. Results


# ## 6.1 K-Means vs Hierarchical Clustering


plt.subplot(1, 2, 1)
plot_cluster_points(df4, model_kmeans.labels_, title='K-Means')

plt.subplot(1, 2, 2)
plot_cluster_points(df4, model_agg.labels_, title='Hierarchical Clustering')


fig = px.choropleth(df_result[['country', 'kmeans_class']],
                    locationmode='country names',
                    locations='country',
                    color=df_result['kmeans_class'],
                    color_discrete_map={
                        'Desenvolvido': 'blue', 'Emergente': 'goldenrod', 'Subdesenvolvido': 'red'},
                    title='KMeans Clustered Countries'
                    )

fig.update_geos(fitbounds="locations",
                visible=False
                )

fig.update_layout(mapbox_style="carto-positron",
                  height=600,
                  width=1000,
                  margin={"r": 0, "t": 0, "l": 0, "b": 0},
                  title_x=0.02,
                  title_y=0.98,
                  legend_title_text='Country',
                  geo=dict(
                      showframe=False,
                      showcoastlines=True,
                      projection_type='equirectangular',
                  ),
                  )

fig.show()


fig = px.choropleth(df_result[['country', 'hier_class']],
                    locationmode='country names',
                    locations='country',
                    color=df_result['hier_class'],
                    color_discrete_map={
                        'Desenvolvido': 'blue', 'Emergente': 'goldenrod', 'Subdesenvolvido': 'red'},
                    title='Hierarchical Clustered Countries'
                    )

fig.update_geos(fitbounds="locations",
                visible=False
                )

fig.update_layout(mapbox_style="carto-positron",
                  height=600,
                  width=1000,
                  margin={"r": 0, "t": 0, "l": 0, "b": 0},
                  title_x=0.02,
                  title_y=0.98,
                  legend_title_text='Country',
                  geo=dict(
                      showframe=False,
                      showcoastlines=True,
                      projection_type='equirectangular',
                  ),
                  )

fig.show()


plt.figure(figsize=(20, 60))

cols = ['child_mort', 'exports', 'health', 'imports',
        'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

i = 0
for k in range(len(cols)):
    plt.subplot(9, 2, i + 1)
    sns.boxplot(x=df_result['kmeans_class'], y=df_result[cols[k]])
    plt.title(cols[k])
    plt.subplot(9, 2, i + 2)
    sns.boxplot(x=df_result['hier_class'], y=df_result[cols[k]])
    plt.title(cols[k])
    i = i + 2


# K-Means: pontos médios de cada cluster

df_cluster = df_result.copy()
df_cluster['country'] = df_cluster.index

df_cluster = df_cluster[['country', 'kmeans_cluster']
                        ].groupby('kmeans_cluster').count().reset_index()

df_avg = df_result[['child_mort', 'exports', 'health', 'imports', 'income', 'inflation',
                    'life_expec', 'total_fer', 'gdpp', 'kmeans_cluster', 'kmeans_class']].groupby('kmeans_class').mean().reset_index()

df_cluster = pd.merge(df_cluster, df_avg, how='inner', on='kmeans_cluster')

df_cluster.sort_values(by=['kmeans_class'])


# Clusterização Hierárquica: pontos médios de cada cluster

df_cluster = df_result.copy()
df_cluster['country'] = df_cluster.index

df_cluster = df_cluster[['country', 'hier_cluster']
                        ].groupby('hier_cluster').count().reset_index()

df_avg = df_result[['child_mort', 'exports', 'health', 'imports', 'income', 'inflation',
                    'life_expec', 'total_fer', 'gdpp', 'hier_cluster', 'hier_class']].groupby('hier_class').mean().reset_index()

df_cluster = pd.merge(df_cluster, df_avg, how='inner', on='hier_cluster')

df_cluster.sort_values(by=['hier_class'])


# Conclusões:
# - De acordo com o K-Measn existem 46 países que mais precisam de ajuda. E de acordo com a Clusterização Hierárquica, existem 41 países que mais precisam de ajuda.
# - O algoritmo do K-Means identificou mais países subdesenvolvidos e menos países Emergentes do que a Clusterização Hierárquica. Mas a quantidade de países desenvovidos encontrado por cada algoritmo foi bem similar, apesar de não serem exatamente os mesmos países.
# - Analisando os primeiros gráficos dessa seção, para o algoritmo da Clusterização Hierárquica parece exister uma zona de confusão entre os países emergentes e subdesenvolvidos, e uma zona de confusão entres os países emergentes e desenvolvidos. O algoritmo do K-Means conseguiu encontrar uma separação melhor entre esses grupos.
# - Comparando os clusters de ambos algoritmos, as features que apresentaram maior divergência nas distribuições, foram: health e imports:
#   - Health: a distribuição dessa feature apresentou diferença principalmente para os clusters dos países desenvolvidos e subdesenvolvidos
#   - Imports: a distribuição dessa feature apresentou diferença principalmente para os clusters dos países desenvolvidos
#
#
