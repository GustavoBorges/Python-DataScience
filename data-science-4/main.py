#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
countries.dtypes # Listando o tipo das colunas


# In[6]:


countries.head()


# In[7]:


#Removendo espaçamento no lado esquerdo e direito do valor
countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()


# In[8]:


dataframe_aux = pd.DataFrame({'Coluna': countries.columns,
                             'tipo': countries.dtypes
}) #Criando dataframe auxiliar


# In[9]:


dataframe_aux.head() #carregando dados


# In[10]:


dataframe_aux = dataframe_aux.drop(['Country', 'Region']) #Excluindo duas colunas texto do dataframe_aux


# In[11]:


print(dataframe_aux)


# In[12]:


colunas_categoricas = list(dataframe_aux[dataframe_aux['tipo'] == 'object']['Coluna']) #Pegando o nome das colunas com o tipo 'object'


# In[13]:


countries[colunas_categoricas] = countries[colunas_categoricas].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))


# In[14]:


countries[colunas_categoricas] = countries[colunas_categoricas].astype('float')


# In[15]:


countries.head()


# In[16]:


print(dataframe_aux)


# In[17]:


countries.dtypes


# In[18]:


dataframe_aux = pd.DataFrame({'Coluna' : countries.columns,
                              'Tipo' : countries.dtypes
                            })


# In[19]:


dataframe_aux


# In[20]:


coluna_float = list(dataframe_aux[dataframe_aux['Tipo'] == 'float64']['Coluna'])
coluna_int = list(dataframe_aux[dataframe_aux['Tipo'] == 'int64']['Coluna'])


# In[21]:


dataframe_mediaspadronizacao = pd.DataFrame()


# In[22]:


for coluna in coluna_float:
    dataframe_mediaspadronizacao[coluna] = countries[coluna].mean()
    #dataframe_mediaspadronizacao[coluna + _padronizacao] = (countries[coluna] - media_population) / countries[coluna].std()


# In[23]:


dataframe_mediaspadronizacao.head()


# In[24]:


media_population = countries['Population'].mean()
media_population = (countries['Population'] - media_population) / countries['Population'].std()
media_area = countries['Area'].mean()
media_area = (countries['Area'] - media_area) / countries['Area'].std()
media_gdp = countries['GDP'].mean()
media_gdp = (countries['GDP'] - media_gdp) / countries['GDP'].std()


# In[25]:


def Resultado_1():
    regiao = list(countries['Region'].sort_values().unique())
    return regiao


# In[26]:


def Resultado_2():
    X = np.array(countries['Pop_density']).reshape(-1, 1)
    est = KBinsDiscretizer(n_bins = 10, encode = 'ordinal',  strategy = 'quantile') 
    est.fit(X)
    Xt = est.transform(X)
    count = 0
    p = np.percentile(Xt, 90)
    for dados in Xt:
        if (dados > p):
           count = count + 1
    return count


# In[27]:


def Resultado_3():
    quant_regiao = len(countries['Region'].unique())
    quant_clima = len(countries['Climate'].unique())
    soma = quant_regiao + quant_clima
    return soma


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[28]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return Resultado_1()
    pass


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[29]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return Resultado_2()
    pass


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[30]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return Resultado_3()
    pass


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[31]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[32]:


X = countries['Arable']


# In[33]:


#pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline(steps=[
("imputer", SimpleImputer(strategy="median")),
("minmax_scaler", StandardScaler()),
])


# In[34]:


#fit data
num_pipeline.fit_transform(countries.drop(['Country','Region'],axis = 1))


# In[35]:


#len(test_country[2:])
def Resultado_4():
    columns = countries.drop(['Country','Region'],axis = 1).keys()
    resposta = num_pipeline.transform(np.array(test_country[2:]).reshape(1,-1))[0][columns == 'Arable'][0]
    resposta = round(float(resposta), 3)
    return resposta


# In[36]:


sns.boxplot(countries['Net_migration'], orient="vertical")


# In[37]:


countries['Net_migration'].describe()


# In[38]:


q_3 = countries['Net_migration'].quantile(0.75)
q_1 = countries['Net_migration'].quantile(0.25)
iqr = q_3 - q_1


# In[39]:


valor_outliers_abaixo = q_1 - 1.5 * iqr
valor_outliers_acima = q_3 + 1.5 * iqr
outliers_abaixo = countries[countries['Net_migration'] < valor_outliers_abaixo]['Net_migration'].count()
outliers_acima = countries[countries['Net_migration'] > valor_outliers_acima]['Net_migration'].count()


# In[40]:


def Resultado_5():
    resp = tuple((int(outliers_abaixo), int(outliers_acima), False))
    return resp
#Não deveria remover, pois existem métodos mais eficazes para este tipo de análise, como o teste de hipotese.
#Outra informação importante é que, a remoção dos outliers depende do negócio.


# In[41]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return Resultado_4()
    pass


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[46]:


def q5():
    # Retorne aqui o resultado da questão 4.
    return Resultado_5()
    pass


# # Respondendo as questões 6 e 7

# In[47]:


from sklearn.datasets import fetch_20newsgroups
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[48]:


newsgroup.data


# In[49]:


vetorizar = CountVectorizer()
X = vetorizar.fit_transform(newsgroup.data)


# In[56]:


def Resultado_6():
    #text = 'phone'
    words_idx = [vetorizar.vocabulary_.get(f"{word.lower()}") for word in
                    [u"phone"]]
    counts = pd.DataFrame(X[:, words_idx].toarray(), columns=np.array(vetorizar.get_feature_names())[words_idx]).sum()
    counts = int(counts[0])
    return counts


# In[58]:


def Resultado_7():
    tfidf = TfidfVectorizer()
    tfidf.fit(newsgroup.data)
    text = 'phone'
    newsgroups_tfidf = tfidf.transform(newsgroup.data)
    words_idx = sorted([vetorizar.vocabulary_.get(f"{word.lower()}") for word in
                        [text]])
    resp = pd.DataFrame(newsgroups_tfidf[:, words_idx].toarray(), columns=np.array(vetorizar.get_feature_names())[words_idx])
    resp = float(round(resp['phone'].sum(), 3))
    return resp


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[223]:


def q6():
    # Retorne aqui o resultado da questão 4.
    return Resultado_6()
    pass


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[119]:


def q7():
    # Retorne aqui o resultado da questão 4.
    return Resultado_7()
    pass

