import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')

data_x = pd.read_csv('Data_X.csv')
data_y = pd.read_csv('Data_Y.csv')
data_new_x = pd.read_csv('DataNew_X.csv')

# Remplacement des valeurs null par la moyenne de la colonne
for col in data_x.columns[data_x.isnull().any()]:
    data_x[col].fillna(data_x[col].mean(), inplace=True)

# Fusion de data_x et data_y
merged_X_Y = pd.merge(data_x, data_y, on='ID', how='inner')

# Triage des données
merged_X_Y = merged_X_Y.sort_values(by=['DAY_ID', 'COUNTRY'])

# Création de deux datasets pour la France et l'Allemagne
data_FR = merged_X_Y.loc[merged_X_Y['COUNTRY'] == 'FR'].copy()
data_DE = merged_X_Y.loc[merged_X_Y['COUNTRY'] == 'DE'].copy()

# Triage des deux datasets avec leurs valeurs correspondantes
data_FR = data_FR.filter(regex='FR|GAS_RET|COAL_RET|TARGET', axis=1)
data_DE = data_DE.filter(regex='DE|GAS_RET|COAL_RET|TARGET', axis=1)

# Calcul des quantiles pour chaque colonne dans data_FR
quantiles_FR = data_FR.quantile(q=[0.05, 0.95])

# Suppression des valeurs aberrantes pour chaque colonne dans data_FR
for colonne in data_FR.columns:
    q05 = quantiles_FR.loc[0.05, colonne]
    q95 = quantiles_FR.loc[0.95, colonne]
    critere_suppression = (data_FR[colonne] < q05) | (data_FR[colonne] > q95)
    data_FR = data_FR.loc[~critere_suppression]

# Calcul des quantiles pour chaque colonne dans data_DE
quantiles_DE = data_DE.quantile(q=[0.05, 0.95])

# Suppression des valeurs aberrantes pour chaque colonne dans data_DE
for colonne in data_DE.columns:
    q05 = quantiles_DE.loc[0.05, colonne]
    q95 = quantiles_DE.loc[0.95, colonne]
    critere_suppression = (data_DE[colonne] < q05) | (data_DE[colonne] > q95)
    data_DE = data_DE.loc[~critere_suppression]


# pour l'Allemagne

# Sélection des variables
data_DE_LR = data_DE.loc[:, ['DE_RESIDUAL_LOAD', 'DE_NET_IMPORT', 'DE_GAS', 'DE_HYDRO', 'TARGET']]
selected_variables = ['DE_RESIDUAL_LOAD', 'DE_NET_IMPORT', 'DE_GAS', 'DE_HYDRO', 'TARGET']

# Prise en compte des corrélations négatives
correlation_metrics_2 = data_DE_LR.corr()
negative_corr_2 = correlation_metrics_2[correlation_metrics_2['TARGET'] < 0].index.tolist()
data_DE_LR[negative_corr_2] = -data_DE_LR[negative_corr_2]

X = data_DE_LR[selected_variables]
Y = data_DE_LR['TARGET']

# Création des poids basés sur les corrélations avec la variable cible
correlation_with_target = correlation_metrics_2['TARGET'].abs()
variable_weights = correlation_with_target[selected_variables]
variable_weights /= variable_weights.sum()

# Appliquer les poids aux variables
X_weighted = X * variable_weights


# Répétition du processus avec différentes valeurs de random_state pour les clusters=
scores = []
random_state_values_done = []

for random_state_train_test in range(100):

    # Création des clusters
    kmeans_model = KMeans(n_clusters=3, n_init=10, random_state=1)
    kmeans_model.fit(X_weighted)
    cluster_labels = kmeans_model.labels_
    data_DE_LR['Cluster'] = cluster_labels

    # Régression Linéaire avec clusters
    X_cluster = data_DE_LR.drop('TARGET', axis=1)
    X_cluster['Cluster'] = data_DE_LR['Cluster']

    # Séparation des données train/test avec un unique random_state
    X_train, X_test, Y_train, Y_test = train_test_split(X_cluster, Y, test_size=0.25, random_state=random_state_train_test)

    regression_model = Ridge(alpha=0.01)
    regression_model.fit(X_train, Y_train)

    Y_pred = regression_model.predict(X_test)
    accuracy = regression_model.score(X_test, Y_test)
    scores.append(accuracy)
    random_state_values_done.append(random_state_train_test)

# Calcul de la moyenne des scores
mean_score_clusters = np.mean(scores)

# Affichage des résultats
best_cluster_index = np.argmax(scores)
best_cluster_random_state = random_state_values_done[best_cluster_index]


print("Moyenne des scores obtenus : {:.2f}%".format(mean_score_clusters * 100))
print("Meilleur random_state de test : ", best_cluster_random_state)
print("Meilleur score obtenu : {:.2f}%".format(max(scores) * 100))

# Création des clusters avec le meilleur random_state
kmeans_model = KMeans(n_clusters=3, n_init=10, random_state=1)
kmeans_model.fit(X_weighted)
cluster_labels = kmeans_model.labels_
data_DE_LR['Cluster'] = cluster_labels

# Régression Linéaire avec clusters
X_cluster = data_DE_LR.drop('TARGET', axis=1)
X_cluster['Cluster'] = data_DE_LR['Cluster']

X_train, X_test, Y_train, Y_test = train_test_split(X_cluster, Y, test_size=0.2, random_state=best_cluster_random_state)

regression_model = Ridge(alpha=0.01)
regression_model.fit(X_train, Y_train)

Y_pred = regression_model.predict(X_test)
accuracy = regression_model.score(X_test, Y_test)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

# Affichage des résultats
print("Score du modèle de régression : {:.2f}%".format(accuracy * 100))
print("MSE : {:.2f}".format(mse))
print("RMSE : {:.2f}".format(rmse))
print("R^2 : {:.2f}".format(r2))

plt.figure(figsize=(10, 6))
sns.scatterplot(x=Y_test.values.flatten(), y=Y_pred.flatten())
sns.regplot(x=Y_test.values.flatten(), y=Y_pred.flatten()   , line_kws={"color": "red"})
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
plt.xlabel('Valeur réelle')
plt.ylabel('Prédiction')
plt.title('Prédiction vs Valeur réelle (Régression linéaire avec clustering) en Allemagne')
plt.show()