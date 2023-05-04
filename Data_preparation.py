import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


data_x = pd.read_csv('Data_X.csv')
data_y = pd.read_csv('Data_Y.csv')
data_new_x = pd.read_csv('DataNew_X.csv')

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

# affiche le type de chaque colonne et si elles sont nulles ou non
data_FR.info()
data_DE.info()

# affiche la distribution, la plage de valeurs et la signification de chaque colonne
data_FR.describe()
data_DE.describe()

# création des histogrammes
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(8, 35))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

for i, ax in enumerate(axes.flatten()):
    if i < len(data_FR.describe().columns):
        j = data_FR.describe().columns[i]
        sns.histplot(data_FR[j], ax=ax)
        ax.axvline(data_FR[j].mean(), color='y', linestyle='-', label='Mean')
        ax.axvline(data_FR[j].median(), color='g', linestyle='-', label='Median')
        ax.set_xlabel(j)
        ax.set_title(f'{j} Distribution')
        ax.legend()

plt.tight_layout()
plt.show(block=True)

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(8, 30))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

for i, ax in enumerate(axes.flatten()):
    if i < len(data_DE.describe().columns):
        j = data_DE.describe().columns[i]
        sns.histplot(data_DE[j], ax=ax)
        ax.axvline(data_DE[j].mean(), color='y', linestyle='-', label='Mean')
        ax.axvline(data_DE[j].median(), color='g', linestyle='-', label='Median')
        ax.set_xlabel(j)
        ax.set_title(f'{j} Distribution')
        ax.legend()

plt.tight_layout()
plt.show(block=True)

# création de diagrammes en boîtes
plt.figure(figsize=(8, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

for i, j in enumerate(data_FR.describe().columns):
    plt.subplot(5, 4, i+1)
    sns.boxplot(x=data_FR[j])
    plt.title('{} Boxplot'.format(j))

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

for i, j in enumerate(data_DE.describe().columns):
    plt.subplot(5, 4, i+1)
    sns.boxplot(x=data_DE[j])
    plt.title('{} Boxplot'.format(j))

plt.tight_layout()
plt.show()


# création de diagrammes de dispersion
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(12, 54))
for i, j in enumerate(data_FR.describe().columns):
    axes.flat[i].set_xlabel(j)
    sns.scatterplot(x=data_FR[j], y=data_FR.TARGET, ax=axes.flat[i])
    axes.flat[i].set_title(f'{j} Distribution')

plt.tight_layout()
plt.show(block=True)

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 40))
for i, j in enumerate(data_DE.describe().columns):
    axes.flat[i].set_xlabel(j)
    sns.scatterplot(x=data_DE[j], y=data_DE.TARGET, ax=axes.flat[i])
    axes.flat[i].set_title(f'{j} Distribution')

plt.tight_layout()
plt.show(block=True)

# Calculer la matrice de corrélation
correlation_metrics_1 = data_FR.corr()

correlation_metrics_2 = data_DE.corr()

# Créer la heatmap avec une taille plus grande, une police plus petite et une palette de couleurs adaptée
fig1 = plt.figure(figsize=(40, 12))
sns.heatmap(correlation_metrics_1, square=True, annot=True, annot_kws={"size": 8}, vmax=1, vmin=-1, cmap='coolwarm')
plt.title('Correlation Between Variables', size=14)
plt.show()

fig2 = plt.figure(figsize=(40, 12))
sns.heatmap(correlation_metrics_2, square=True, annot=True, annot_kws={"size": 8}, vmax=1, vmin=-1, cmap='coolwarm')
plt.title('Correlation Between Variables', size=14)
plt.show()
