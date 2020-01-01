#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
import statistics

#from loguru import logger


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


fifa = pd.read_csv("fifa.csv")


# In[4]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[64]:


# Sua análise começa aqui.
fifa.head() #Carregando o dataframe


# In[149]:


fifa = fifa.dropna()


# In[154]:


fifa = fifa-fifa.mean() #Normalizando base


# In[69]:


features = fifa.columns #Variavel recebendo o nome das colunas


# In[95]:


X = fifa.loc[:, features].values #X = Os registros das colunas "features"


# In[71]:


def Resultado_1(): 
    pca = PCA().fit(X)
    evr = pca.explained_variance_ratio_
    evr = float(round(evr[0], 3))
    return evr


# In[47]:


def Resultado_2():
    pca = PCA(n_components= 0.95)
    principalComponents = pca.fit_transform(X)
    n_componentes = int(principalComponents.shape[1])
    return n_componentes


# In[62]:


def Resultado_4():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression

    X = fifa.drop('Overall',axis = 1)
    y = fifa['Overall']
    keys = fifa.drop('Overall',axis = 1).keys() #pegando o nome das colunas

    model = LinearRegression() #Selecionando o modelo de regressão
    selector = RFE(model, 5, step=1) #Aplicando o RFE com o modelo selecionado, total de colunas no modelo final e eliminando uma a uma.
    selector = selector.fit(X, y)
    important = keys[selector.support_] #Pegando as colunas mais importantes
    important = list(important)  #lista com as mais importantes
    return important


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[101]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return Resultado_1()
    pass


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[102]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return Resultado_2()
    pass


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[144]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[161]:


def Resultado_3():
    pca = PCA(n_components= 2)
    pca = pca.fit(fifa)
    cons = pca.transform(np.reshape(x, (1, -1)))
    cons = tuple(cons[0])
    cons = tuple((round(cons[0], 3), round(cons[1], 3)))
    return cons


# In[162]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return Resultado_3()
    pass


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[164]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return Resultado_4()
    pass

