import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
import itertools
import warnings
from sklearn import linear_model
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
import graphviz

# Import demographics
demographics_url = f'https://raw.githubusercontent.com/vbtCogSci/features_of_agency/master/data/datasets/properties_1/datasets_csv/demographics.csv'
df_demo = pd.read_csv(demographics_url, sep=',').rename({'Unnamed: 0': 'part_id'}, axis=1)

# Import sanity checks
sanity_checks_url = f'https://raw.githubusercontent.com/vbtCogSci/features_of_agency/master/data/datasets/properties_1/datasets_csv/sanity_checks.csv'
sanity_checks_in = pd.read_csv(sanity_checks_url, sep=',').drop('Unnamed: 0', axis=1)

# Task data
df_task = f'https://raw.githubusercontent.com/vbtCogSci/features_of_agency/master/data/datasets/properties_1/datasets_csv/task_data.csv'
df_task = pd.read_csv(df_task).drop('Unnamed: 0', axis=1)

properties = df_task.columns[4:-4].to_list()
properties_agency = df_task.columns[4:-4].to_list() + ['agency']
properties_full = df_task.columns[4:].to_list()

X = df_task[properties_agency].to_numpy().T 
U_label = properties_agency

df_means = df_task[['word'] + properties].groupby('word').mean()#.sort_values('agency')

X = df_means.to_numpy()
labels = df_means.index.to_list()

num_components = 6
cov = 'full'

gm = GaussianMixture(n_components=num_components, covariance_type=cov).fit(X)

#for i in range(gm.covariances_.shape[0]):
#    print(np.round(np.diag(gm.covariances_[i, :, :])**(1/2), 4))

plt.hist(gm.predict_proba(X).max(axis=1))
plt.show()

a = gm.predict(X)

df_means['assignment'] = a

for w in df_means.index:
    df_task.loc[df_task.word == w, 'cluster'] = df_means.loc[w, 'assignment']