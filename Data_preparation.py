import pandas as pd
import numpy as np

data_x = pd.read_csv('Data_X.csv')
data_y = pd.read_csv('Data_Y.csv')
data_new_x = pd.read_csv('DataNew_X.csv')

# Avant suppression des lignes ayant des valeurs nulles
print(data_x)
# Recherche des valeurs nulles
print(data_x.isnull())
# Suppression des lignes ayant des valeurs nulles
data_x.dropna(inplace=True)
print(data_x)