#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[124]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[125]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[126]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[127]:


# Sua análise da parte 1 começa aqui.
dataframe.head()


# In[128]:


#Respondendo a questão 1
def Resultado_1():
    q1_norm = dataframe.normal.quantile(0.25)
    q2_norm = dataframe.normal.quantile(0.50)
    q3_norm = dataframe.normal.quantile(0.75)
    q1_binom = dataframe.binomial.quantile(0.25)
    q2_binom = dataframe.binomial.quantile(0.50)
    q3_binom = dataframe.binomial.quantile(0.75)
    difQuartis = (round(q1_norm - q1_binom, 3), round(q2_norm - q2_binom, 3), round(q3_norm - q3_binom, 3))
    return difQuartis


# In[182]:


#Respondendo a questão 2
def Resultado_2():
    media = dataframe.normal.mean()
    std = dataframe.normal.std()
    norm = dataframe.normal
    low = media - std
    up = media + std
    result_cdf = ECDF(norm)
    result_cdf = result_cdf([low, up])
    resp = result_cdf[1] - result_cdf[0]
    resp = round(resp, 3)
    return resp


# In[184]:


#Respondendo a questão 3
def Resultado_3():
    m_norm = dataframe.normal.mean()
    m_binom = dataframe.binomial.mean()
    v_norm = dataframe.normal.var()
    v_binom = dataframe.binomial.var()
    difMedVar = (round(m_binom - m_norm, 3), round(v_binom - v_norm, 3))
    return difMedVar


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[88]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return Resultado_1()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[133]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return Resultado_2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[90]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return Resultado_3()


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[91]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[92]:


# Sua análise da parte 2 começa aqui.
stars.head(20)


# In[93]:


stars.shape


# In[94]:


#Filtrando apenas registros com o valor 0 na coluna target
filtro_target = stars.where(stars.target == 0)
filtro_target = filtro_target.query('target.notnull()', engine='python')


# In[95]:


#Padronizando os valores com score z
media = filtro_target.mean_profile.mean()
std = filtro_target.mean_profile.std()
mean_profile = (filtro_target.mean_profile - media) / std


# In[96]:


filtro_target['false_pulsar_mean_profile_standardized'] = mean_profile


# In[97]:


filtro_target.shape


# In[189]:


def Resultado_4():
    media = 0
    std = 1
    q_1 = sct.norm.ppf(0.80, loc=media, scale=std)
    q_2 = sct.norm.ppf(0.90, loc=media, scale=std)
    q_3 = sct.norm.ppf(0.95, loc=media, scale=std)
    media_false_pulsar = filtro_target.false_pulsar_mean_profile_standardized.mean()
    std_false_pulsar = filtro_target.false_pulsar_mean_profile_standardized.std()
    ecdf = ECDF(filtro_target.false_pulsar_mean_profile_standardized)
    resp = (round(ecdf(q_1), 3), round(ecdf(q_2), 3), round(ecdf(q_3), 3))
    return resp


# In[193]:


def Resultado_5(): 
    q_1 = filtro_target.false_pulsar_mean_profile_standardized.quantile(0.25)
    q_2 = filtro_target.false_pulsar_mean_profile_standardized.quantile(0.50)
    q_3 = filtro_target.false_pulsar_mean_profile_standardized.quantile(0.75)
    ppf_1 = sct.norm.ppf(0.25)
    ppf_2 = sct.norm.ppf(0.50)
    ppf_3 = sct.norm.ppf(0.75)
    resp = (round(q_1 - ppf_1, 3), round(q_2 - ppf_2, 3), round(q_3 - ppf_3, 3))
    resp = tuple(resp)
    return resp


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[195]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return Resultado_4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[196]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return Resultado_5()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
