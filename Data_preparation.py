import pandas as pd
import numpy as np

data_x = pd.read_csv('Data_X.csv')
data_y = pd.read_csv('Data_Y.csv')
data_new_x = pd.read_csv('DataNew_X.csv')

"""
# Avant suppression des lignes ayant des valeurs nulles
print(data_x)
print(data_new_x)
# Recherche des valeurs nulles
print(data_x.isnull())
print(data_new_x.isnull())
# Suppression des lignes ayant des valeurs nulles
data_x.dropna(inplace=True)
data_new_x.dropna(inplace=True)
print(data_x)
print(data_new_x)


# On remplace les valeurs manquantes par des 0.0

#data_x = data_x.fillna(0.0)
print(data_x.isnull())
print(data_x.loc[:,'DE_RAIN'])
df.loc[:,'SalePrice']

"""

data_x = data_x.fillna(data_x.mean())
print(data_x)

#%%
