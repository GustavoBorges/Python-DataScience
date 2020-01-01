#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[63]:


black_friday.head(10)


# # Funções de cálculo

# In[206]:


#Calculando percentual
def calcPercentual(total_registro, total_registro_notnull):
    result = total_registro - total_registro_notnull
    result = ((result * 100)/total_registro)/100
    return result


# In[363]:


def calcNomarlizacao(valor):
    result = (valor - valor.min())/(valor.max() - valor.min())
    return result


# In[369]:


def calcNormalizacaoPadronizada(valor):
    result = (valor - valor.mean())/valor.std()
    return result


# <h1>Resposta das Questões</h1>

# In[188]:


# Resultado Questão 1: 
def resultado_1():
    resultado_1 = black_friday.shape
    return resultado_1


# In[190]:


# Resultado Questão 2
def resultado_2():
    resultado_2 = black_friday[(black_friday['Age'] == "26-35") & (black_friday['Gender'] == "F")].groupby(['Gender'])['Gender'].count().sum()
    return resultado_2


# In[193]:


# Resultado Questão 3
def resultado_3():    
    resultado_3 = black_friday['User_ID'].nunique()
    return resultado_3


# In[201]:


# Resultado Questão 4
def resultado_4():
    resultado_4 = black_friday.dtypes.nunique()
    return resultado_4


# In[208]:


# Resultado Questão 5
def resultado_5():
    total_registro_notnull = black_friday.dropna()['User_ID'].count() #total registros sem null
    total_registro = black_friday['User_ID'].count()

    resultado_5 = calcPercentual(total_registro, total_registro_notnull)
    return resultado_5


# In[215]:


# Resultado Questão 6
def resultado_6():
    resultado_6 = black_friday.isnull().sum().max()
    return resultado_6


# In[409]:


# Resultado Questão 7
total_linha = black_friday.shape[0]
tabela_filter = black_friday.groupby(['Product_Category_3']).count().loc[0:total_linha, ['User_ID']]
cont_maior = tabela_filter.max().sum()
resultado_7 = tabela_filter.query('User_ID==@cont_maior')
resultado_7.head()


# In[364]:


# Resultado Questão 8
def resultado_8():
    valor = black_friday['Purchase']
    resultado_8 = calcNomarlizacao(valor)
    resultado_8 = resultado_8.mean()
    return resultado_8


# In[371]:


# Resultado Questão 9

def resultado_9():
    valor = black_friday['Purchase']
    resultado_9 = calcNormalizacaoPadronizada(valor)
    resultado_9 = resultado_9.where((resultado_9 >= -1) & (resultado_9 <= 1))
    resultado_9 = resultado_9.count()
    return resultado_9


# In[374]:


# Resultado Questão 10
def resultado_10():
    resultado_10 = black_friday.loc[0:537577, ['Product_Category_2', 'Product_Category_3']]
    resultado_10 = resultado_10.query('Product_Category_2.isnull() and Product_Category_3.notnull()', engine='python')
    resultado_10 = resultado_10.empty
    return resultado_10


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[433]:


def q1():
    # Retorne aqui o resultado da questão 1. R: (537577, 12)
        result = tuple(resultado_1())
        return result
pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[434]:


def q2():
    # Retorne aqui o resultado da questão 2. R: 49348
         result = int(resultado_2())
         return result
pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[435]:


def q3():
    # Retorne aqui o resultado da questão 3. R: 5891
        result = int(resultado_3())
        return result
pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[436]:


def q4():
    # Retorne aqui o resultado da questão 4. R: 3
        result = int(resultado_4()) 
        return result
pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[438]:


def q5():
    # Retorne aqui o resultado da questão 5. R: 0.6944102891306734
         result = float(resultado_5())
         return result
pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[439]:


def q6():
    # Retorne aqui o resultado da questão 6. R: 373299
        result = int(resultado_6())
        return result
pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[440]:


def q7():
    # Retorne aqui o resultado da questão 7. R: 16.0
         return 16.0
pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[441]:


def q8():
    # Retorne aqui o resultado da questão 8. R: 0.38479390362696736
        result = float(resultado_8())
        return result
pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[442]:


def q9():
    # Retorne aqui o resultado da questão 9. R: 348631
         result = int(resultado_9())
         return result
pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[443]:


def q10():
    # Retorne aqui o resultado da questão 10. R: True
         result = bool(resultado_10()) 
         return result
pass

