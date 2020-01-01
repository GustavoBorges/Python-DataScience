#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[115]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import random as random


# In[116]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[117]:


athletes = pd.read_csv("athletes.csv")


# In[143]:


def get_sample(df, col_name, n=3000, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[144]:


# Sua análise começa aqui.
athletes.shape #Verificando quantidade de linha e colunas do dataframe


# In[145]:


athletes.describe() #Verificando as informações de media, moda...


# In[146]:


athletes.head() #Lendo as 5 primeiras linhas do dataframe


# ## Criando as soluções das questões

# In[202]:


def Resultado_1(): 
    amostra = get_sample(athletes, 'height')
    stat, p = sct.shapiro(amostra)
    alpha = 0.05
    if  p > alpha:
        return True
    else:
        return False


# In[123]:


def Resultado_2():    
    amostra = get_sample(athletes, 'height')
    stat, p = sct.jarque_bera(amostra)
    alpha = 0.05
    if  p > alpha:
        return True
    else:
        return False


# In[124]:


def Resultado_3(): 
    amostra = get_sample(athletes, 'weight')
    stat, p = sct.normaltest(amostra)
    alpha = 0.05
    if  p > alpha:
        return True
    else:
        return False


# In[125]:


def Resultado_4():
    amostra = get_sample(athletes, 'weight')
    amostra_log = np.log(amostra)
    stat, p = sct.normaltest(amostra_log)
    alpha = 0.05
    if  p > alpha:
        return True
    else:
        return False


# In[239]:


bra = athletes.where(athletes.nationality == 'BRA')
bra = bra.height.dropna()
usa = athletes.where(athletes.nationality == 'USA')
usa = usa.height.dropna()
can = athletes.where(athletes.nationality == 'CAN')
can = can.height.dropna()


# In[260]:


def Resultado_5():
    teste_hipotese, p = sct.ttest_ind(bra, usa, equal_var=False)
    alpha = 0.05
    if  p > alpha:
        return True
    else:
        return False


# In[261]:


def Resultado_6():
    teste_hipotese, p = sct.ttest_ind(bra, can, equal_var=False)
    alpha = 0.05
    if  p > alpha:
        return True
    else:
        return False


# In[262]:


def Resultado_7():
    teste_hipotese, p = sct.ttest_ind(usa, can, equal_var=False)
    p = float(round(p, 8))
    return p


# ## Gráficos

# In[266]:


#amostra.plot.hist(bins=25)
#plt.show()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[267]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return Resultado_1()
    pass


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[268]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return Resultado_2()
    pass


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[269]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return Resultado_3()
    pass


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[270]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return Resultado_4()
    pass


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[271]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return Resultado_5()
    pass


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[272]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return Resultado_6()
    pass


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[273]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return Resultado_7()
    pass


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
