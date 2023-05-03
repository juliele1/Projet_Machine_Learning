import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# remplacement des valeurs null par la moyenne de la colonne
for col in data_x.columns[data_x.isnull().any()]:
    data_x[col].fillna(data_x[col].mean(), inplace=True)
print(data_x)

# fusion de data_x et data_y
merged_X_Y = pd.merge(data_x, data_y, on='ID', how='inner')

# triage des données
merged_X_Y = merged_X_Y.sort_values(by=['DAY_ID', 'COUNTRY'])
print(merged_X_Y)

# création de deux datas pour la France et l'Allemagne
data_FR = merged_X_Y.loc[merged_X_Y['COUNTRY'] == 'FR']
data_DE = merged_X_Y.loc[merged_X_Y['COUNTRY'] == 'DE']

# triage des deux datas avec leurs valeurs correspondantes
data_FR = data_FR.filter(regex='FR|GAS_RET|COAL_RET|TARGET', axis=1)
print(data_FR)

data_DE = data_DE.filter(regex='DE|GAS_RET|COAL_RET|TARGET', axis=1)
print(data_DE)

# affiche le type de chaque colonnes et si elles sont nulles ou non
data_FR.info()
data_DE.info()

# affiche la distribution, la plage de valeurs et la signification de chaque colonne
data_FR.describe()
data_DE.describe()

DATA_FR = sns.load_dataset("Data_FR")

#hist = data_FR['FR_CONSUMPTION'].hist()
#print(plt.savefig("pandas_hist_01.png", bbox_inches='tight', dpi=100))
#sns.histplot(data=DATA_FR, x='FR_CONSUMPTION')
#plt.show()


plt.figure(figsize=(19,23))
for i, j in enumerate(data_FR.describe().columns):
    plt.subplot(4, 2, i+1)
    sns.distplot(x=data_FR[j])
    plt.xlabel(j)
    plt.title('{} Distribution'.format(j))
    # plt.subplots_adjust(wspace=.2, hspace=.5)
    plt.tight_layout()
plt.show()
