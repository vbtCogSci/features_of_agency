from ast import keyword
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

keyword = "properties_1"

demographics_file_name = 'data_demo_070922.csv'
data_file_names = [f'data_task_{i+1}_070922.csv' for i in range(8)]

#task_raw =  pd.read_csv(f'.\\data\\raw\\{keyword}\\{data_file_name}')
demos_raw = pd.read_csv(f'.\\data\\raw\\{keyword}\\{demographics_file_name}')
data_files_raw = [pd.read_csv(f'.\\data\\raw\\{keyword}\\{file_name}') for file_name in data_file_names]

# PRIVACY: set TRUE if exporting on github (hides prolific id), else FALSE for checks
private = True
if private:
    part_id_key = 'Participant Private ID'
    destination_folder = 'datasets'
else:
    part_id_key = 'Participant Public ID'
    destination_folder = 'datasets_prolific_id'

#### DEMOGRAPHICS LOGIC
demos_columns = ['Sex', 'Sex-quantised', 'Sex-text', 'age', 'academic', 'academic-quantised', 'revenue', 'revenue-quantised', 'politics', 'religiosity']
df_demos = pd.DataFrame(columns=demos_columns)

# Columns with relevant data
question_key = 'Question Key'
response_key = 'Response'

# Collect participant ID
part_ids_demo = demos_raw[part_id_key].dropna().unique()

for ids in part_ids_demo:
    part_id = ids

    # Demos part
    demos_part = demos_raw[demos_raw[part_id_key] == part_id]
    idx = demos_part.index
    cols = demos_part.columns
    # Clean dataframe of unused columns and rows
    demos_data = demos_part[[question_key, response_key]].drop(demos_part.tail(1).index, axis=0).dropna()
    # Collect columns
    new_cols = list(demos_data[question_key])

    # Adjust data format
    demos_idx = demos_data.drop(question_key, axis=1)
    demos_renamed = demos_idx.rename(columns={response_key:part_id})
    demos_T = demos_renamed.transpose()
    demos_T.columns = new_cols

    ##df_demos = demos_T
    df_demos = df_demos.append(demos_T)

# If private only remove education, revenues and politics
if private:
    df_demos.drop(['academic', 'academic-quantised', 'revenue', 'revenue-quantised', 'politics'], axis=1, inplace=True)


## MAIN TASK 

# Property allocation dictionaries
property_allocation_dict = {
    4 : {
        'physicalproperty1':'movement',
        'physicalproperty2': 'energy',
        'physicalproperty3': 'replication',
        'physicalproperty4': 'complexity',
        'behaviouralproperty1': 'learning',
        'behaviouralproperty2': 'reaction',
        'behaviouralproperty3': 'mistakes',
        'behaviouralproperty4': 'communication',
        'behaviouralproperty5': 'variety',
        'behaviouralproperty6': 'monitoring',
        'highlevelproperty1': 'agency',
        'highlevelproperty2': 'goal_setting',
        'highlevelproperty3': 'goal_directedness',
        'highlevelproperty4': 'freewill'
    },
    1 : {
        'physicalproperty1':'energy',
        'physicalproperty2': 'complexity',
        'physicalproperty3': 'movement',
        'physicalproperty4': 'replication',
        'behaviouralproperty1': 'communication',
        'behaviouralproperty2': 'monitoring',
        'behaviouralproperty3': 'reaction',
        'behaviouralproperty4': 'variety',
        'behaviouralproperty5': 'mistakes',
        'behaviouralproperty6': 'learning',
        'highlevelproperty1': 'goal_setting',
        'highlevelproperty2': 'freewill',
        'highlevelproperty3': 'agency',
        'highlevelproperty4': 'goal_directedness'
    },
    3 : {
        'physicalproperty1':'complexity',
        'physicalproperty2': 'movement',
        'physicalproperty3': 'replication',
        'physicalproperty4': 'energy',
        'behaviouralproperty1': 'reaction',
        'behaviouralproperty2': 'variety',
        'behaviouralproperty3': 'learning',
        'behaviouralproperty4': 'communication',
        'behaviouralproperty5': 'monitoring',
        'behaviouralproperty6': 'mistakes',
        'highlevelproperty1': 'freewill',
        'highlevelproperty2': 'agency',
        'highlevelproperty3': 'goal_directedness',
        'highlevelproperty4': 'goal_setting'
    },
    2 : {
        'physicalproperty1':'energy',
        'physicalproperty2': 'replication',
        'physicalproperty3': 'complexity',
        'physicalproperty4': 'movement',
        'behaviouralproperty1': 'monitoring',
        'behaviouralproperty2': 'mistakes',
        'behaviouralproperty3': 'variety',
        'behaviouralproperty4': 'learning',
        'behaviouralproperty5': 'communication',
        'behaviouralproperty6': 'reaction',
        'highlevelproperty1': 'goal_setting',
        'highlevelproperty2': 'goal_directedness',
        'highlevelproperty3': 'agency',
        'highlevelproperty4': 'freewill'
    },
    ###
    8 : {
        'physicalproperty1':'movement',
        'physicalproperty2': 'energy',
        'physicalproperty3': 'replication',
        'physicalproperty4': 'complexity',
        'behaviouralproperty1': 'learning',
        'behaviouralproperty2': 'reaction',
        'behaviouralproperty3': 'mistakes',
        'behaviouralproperty4': 'communication',
        'behaviouralproperty5': 'variety',
        'behaviouralproperty6': 'monitoring',
        'highlevelproperty1': 'agency',
        'highlevelproperty2': 'goal_setting',
        'highlevelproperty3': 'goal_directedness',
        'highlevelproperty4': 'freewill'
    },
    5 : {
        'physicalproperty1':'energy',
        'physicalproperty2': 'complexity',
        'physicalproperty3': 'movement',
        'physicalproperty4': 'replication',
        'behaviouralproperty1': 'communication',
        'behaviouralproperty2': 'monitoring',
        'behaviouralproperty3': 'reaction',
        'behaviouralproperty4': 'variety',
        'behaviouralproperty5': 'mistakes',
        'behaviouralproperty6': 'learning',
        'highlevelproperty1': 'goal_setting',
        'highlevelproperty2': 'freewill',
        'highlevelproperty3': 'agency',
        'highlevelproperty4': 'goal_directedness'
    },
    7 : {
        'physicalproperty1':'complexity',
        'physicalproperty2': 'movement',
        'physicalproperty3': 'replication',
        'physicalproperty4': 'energy',
        'behaviouralproperty1': 'reaction',
        'behaviouralproperty2': 'variety',
        'behaviouralproperty3': 'learning',
        'behaviouralproperty4': 'communication',
        'behaviouralproperty5': 'monitoring',
        'behaviouralproperty6': 'mistakes',
        'highlevelproperty1': 'freewill',
        'highlevelproperty2': 'agency',
        'highlevelproperty3': 'goal_directedness',
        'highlevelproperty4': 'goal_setting'
    },
    6 : {
        'physicalproperty1':'energy',
        'physicalproperty2': 'replication',
        'physicalproperty3': 'complexity',
        'physicalproperty4': 'movement',
        'behaviouralproperty1': 'monitoring',
        'behaviouralproperty2': 'mistakes',
        'behaviouralproperty3': 'variety',
        'behaviouralproperty4': 'learning',
        'behaviouralproperty5': 'communication',
        'behaviouralproperty6': 'reaction',
        'highlevelproperty1': 'goal_setting',
        'highlevelproperty2': 'goal_directedness',
        'highlevelproperty3': 'agency',
        'highlevelproperty4': 'freewill'
    }
}

properties = list(property_allocation_dict[1].values())


def remove_article(my_string, filler='_'):
    split_string = my_string.split(' ')
    if len(split_string) > 1:
        return filler.join(split_string[1:])
    else:
        return my_string

# Task data

if private:
    part_id_key = 'Participant Private ID'
    destination_folder = 'datasets'
else:
    part_id_key = 'Participant Public ID'
    destination_folder = 'datasets_prolific_id'

property_key = 'Zone Name'
zone_type = 'Zone Type'
response_key = 'Response'
word_key = 'word'
reaction_time = 'Reaction Time'
 
cols = [part_id_key, word_key, property_key, response_key, reaction_time, zone_type]


# Task data
if private:
    part_id_key = 'Participant Private ID'
    destination_folder = 'datasets'
else:
    part_id_key = 'Participant Public ID'
    destination_folder = 'datasets_prolific_id'

property_key = 'Zone Name'
zone_type = 'Zone Type'
response_key = 'Response'
word_key = 'word'
reaction_time = 'Reaction Time'
 
cols = [part_id_key, word_key, property_key, response_key, reaction_time, zone_type]

dfs = []

for i, df in enumerate(data_files_raw):
    df_cond = df[cols].replace(property_allocation_dict[i+1])
    df_cond['slider_order'] = i + 1
    df_cond_clean = (df_cond[df_cond[zone_type] == 'response_slider_endValue']).drop([zone_type], axis=1)
    df_cond_clean[word_key] = df_cond_clean[word_key].apply(remove_article)
    df_cond_clean = df_cond_clean.rename({property_key:'property', response_key:'score', reaction_time:'reaction_time', 'Participant Private ID': 'part_id'}, axis=1)
    df_cond_clean['word'] = df_cond_clean['word'].replace({'virtual_assistant,_e.g._Alexa_or_Google_Home': 'virtual_assistant'})
    dfs.append(df_cond_clean)


df_task_lf = pd.concat(dfs, axis=0).reset_index(drop=True)

df_pcols = df_task_lf.copy()
for i, prop in enumerate(properties):
    prop_values = df_pcols[df_pcols.property == prop].score.to_list()
    df_pcols.loc[df_pcols.property == prop, prop] = prop_values


df_task = df_pcols[~df_pcols.movement.isna()].copy()

for i, prop in enumerate(properties):
    prop_values = df_pcols[~df_pcols[prop].isna()][prop].to_list()
    df_task.loc[:, prop] = prop_values
    
df_task = df_task.drop(['property', 'score'], axis=1).reset_index(drop=True)

df_checks = df_task[df_task.word.isin(['human', 'rock', 'hammer'])]



# Save files
df_checks.to_excel(f'.\\data\\{destination_folder}\\{keyword}\\datasets_excel\\sanity_checks.xlsx')
df_demos.to_excel(f'.\\data\\{destination_folder}\\{keyword}\\datasets_excel\\demographics.xlsx')
df_task_lf.to_excel(f'.\\data\\{destination_folder}\\{keyword}\\datasets_excel\\task_data_lf.xlsx')
df_task.to_excel(f'.\\data\\{destination_folder}\\{keyword}\\datasets_excel\\task_data.xlsx')

df_checks.to_csv(f'.\\data\\{destination_folder}\\{keyword}\\datasets_csv\\sanity_checks.csv')
df_demos.to_csv(f'.\\data\\{destination_folder}\\{keyword}\\datasets_csv\\demographics.csv')
df_task_lf.to_csv(f'.\\data\\{destination_folder}\\{keyword}\\datasets_csv\\task_data_lf.csv')
df_task.to_csv(f'.\\data\\{destination_folder}\\{keyword}\\datasets_csv\\task_data.csv')

# Clustering
# Task data
df_task = f'.\\data\\{destination_folder}\\{keyword}\\datasets_csv\\task_data.csv'
df_task = pd.read_csv(df_task).drop('Unnamed: 0', axis=1)

properties = df_task.columns[4:-4].to_list()
properties_agency = df_task.columns[4:-4].to_list() + ['agency']
properties_full = df_task.columns[4:].to_list()

df_means = df_task[['word'] + properties].groupby('word').mean()#.sort_values('agency')
#df_means = df_means.groupby('word').mean()
#print(df_means.groupby('word').mean())
X = df_means.to_numpy()
labels = df_means.index.to_list()

num_components = 6
cov = 'full'

gm = GaussianMixture(n_components=num_components, covariance_type=cov).fit(X)

a = gm.predict(X)

df_means['assignment'] = a

for w in df_means.index:
    df_task.loc[df_task.word == w, 'cluster'] = df_means.loc[w, 'assignment']


df_task.to_excel(f'.\\data\\{destination_folder}\\{keyword}\\datasets_excel\\task_data.xlsx')
df_task.to_csv(f'.\\data\\{destination_folder}\\{keyword}\\datasets_csv\\task_data.csv')