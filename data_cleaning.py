import pandas as pd
import numpy as np

task_raw =  pd.read_csv('.\\data\\raw\\data_exp_70408-v19_task-niae.csv')
demos_raw = pd.read_csv('.\\data\\raw\\data_exp_70408-v19_questionnaire-sgay.csv')

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
part_ids_demo = task_raw[part_id_key].dropna().unique()

for ids in part_ids_demo:
    part_id = ids

    # Demos part
    demos_part = demos_raw[demos_raw[part_id_key] == part_id]
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

#### TASK LOGIC
# Collect participant ID
part_ids = task_raw[part_id_key].dropna().unique()

# Columns with relevant data

## Find variable names (could replace)
zone_key = 'Zone Name'
rating_val = 'agencyjudgement'
sense_val = 'sense' 
math_val = 'mathresponse1'
## Secondary keys
zone_sec_key = 'Zone Type'
rating_last_val = 'response_slider_endValue'
sense_sec_val = 'response_rating_scale_likert'
math_sec_val = 'response_text_entry'

## Find response column
response_key = 'Response'
## Reaction time
reaction_time_key = 'Reaction Time'

## Find words
word_left_key = 'word1'
word_right_key = 'word2'

# Math question col
math_question_key = 'math_question'

# Columns of interest
gen_keys = [zone_key, zone_sec_key, response_key, reaction_time_key]
task_math_keys = [math_question_key]
task_comparison_keys = [word_left_key, word_right_key]

# Formatting adjustments
def remove_article(my_string, filler='_'):
    split_string = my_string.split(' ')
    if len(split_string) > 1:
        return filler.join(split_string[1:])
    else:
        return my_string


agency_datasets = []
checks_columns = ['total_time_min', 'rock_human_score', 'rock_god_score', 'math1_num_resp', 'math2_num_resp', 'math3_num_resp']
df_checks = pd.DataFrame(columns=checks_columns)

for part_id in part_ids:
    #print(part_id)
    task_part = task_raw[task_raw[part_id_key] == part_id]

    # Get full elapsed time in milliseconds
    time_ml = int(task_part['Reaction Time'].dropna().tail(1))
    time_min = time_ml / 1000 / 60
    
    
    task_clean = task_part.drop(task_part.tail(1).index, axis=0)[gen_keys + task_math_keys + task_comparison_keys].rename({reaction_time_key:'reaction_time'}, axis=1)
    
    # Sense data
    sense_data_raw = task_clean[task_clean[zone_key] == sense_val].copy()
    series = (sense_data_raw[response_key] != 'Does not make sense').astype(int)
    sense_data_raw['sense_response'] = series
    sense_data = sense_data_raw.drop(task_math_keys + [zone_key, zone_sec_key, response_key], axis=1).reset_index(drop=True)

    # Agency data
    agency_data_raw_1 = task_clean[task_clean[zone_key] == rating_val]
    agency_data_raw = agency_data_raw_1[agency_data_raw_1[zone_sec_key] == rating_last_val].copy()
    agency_data_raw[response_key] = agency_data_raw[response_key].astype(int)
    agency_data_raw['left_score'] = agency_data_raw[response_key].apply(lambda x: 3 - x if x < 3 else 2 - x)
    agency_data_raw['right_score'] = agency_data_raw['left_score'].apply(lambda x: -x)
    agency_data_raw['left_score_binary'] = agency_data_raw['left_score'].apply(lambda x: 1 if x > 0 else 0)
    agency_data_raw['right_score_binary'] = agency_data_raw['right_score'].apply(lambda x: 1 if x > 0 else 0)
    #agency_data_raw['right_score'] = agency_data_raw[response_key].apply(lambda x: x - 2 if x > 2 else x - 3)
    agency_data = agency_data_raw.drop(task_math_keys + [zone_key, zone_sec_key, response_key], axis=1).reset_index(drop=True)
    
    # Group all data (sense and agency)
    task_data = agency_data.merge(sense_data, how='left')
    
    # Apply article removal
    task_data['word1'] = task_data['word1'].apply(remove_article)
    task_data['word2'] = task_data['word2'].apply(remove_article)
    # Rename words columns to left and right
    task_data = task_data.rename({'word1':'words_left', 'word2':'words_right'}, axis=1)

    # Store data
    agency_datasets.append(task_data)


    ## Checking dataframe
    # Checks dataframe
    ## Math data
    math_data_raw = task_clean[task_clean[zone_key] == math_val].copy()
    math_questions = [
        '9 + 7 + 9 =',
        '3 + 14 =',
        '9 / 3 ='
    ]
    math_attemps = np.zeros(len(math_questions))

    for i, q in enumerate(math_questions):
        math_attemps[i] = len(math_data_raw[math_data_raw[math_question_key] == q])

    # Task completion time
    explanation_time = task_raw[task_raw[zone_key] == 'advancementZone'][reaction_time_key].sum()
    fixation_time = task_raw[task_raw[zone_key] == 'fixation'][reaction_time_key].sum()
    transition_time = task_raw[task_raw[zone_sec_key] == 'timelimit_screen'][reaction_time_key].sum()

    full_time_manual = math_data_raw.reaction_time.sum() + task_data.reaction_time.sum() + explanation_time + fixation_time + transition_time

    rock_score = task_data[task_data.words_right == 'rock']
    rock_human_score = rock_score[rock_score.words_left == 'human'].reset_index(drop=True).loc[0, 'right_score']

    rock_score = task_data[task_data.words_right == 'rock']
    rock_god_score = rock_score[rock_score.words_left == 'god'].reset_index(drop=True).loc[0, 'right_score']

    # Aggregate data
    data = [time_min, rock_human_score, rock_god_score, math_attemps[0], math_attemps[1], math_attemps[2]]

    # Add data to dataframe
    df_checks.loc[part_id] = data


# Create matrices and array
## Score, binary, sense
words = pd.Series(list(task_data.words_left) + list(task_data.words_right)).unique()

aggregate_columns = ['mean_score', 'score_std', 'win_prob', 'win_prob_std', 'sense_prob', 'sense_std']

score_matrix = np.zeros((len(part_ids), len(words), len(words)))
win_matrix = np.zeros((len(part_ids), len(words), len(words)))
sense_matrix = np.zeros((len(part_ids), len(words), len(words)))
aggregate_matrix = np.zeros((len(part_ids), len(words), len(aggregate_columns)))

dfs_scores = []
dfs_wins = []
dfs_sense = []
dfs_aggregate = []

for k, part_id in enumerate(part_ids):
    
    matrix_prep = agency_datasets[k].copy()

    for word in words:
        matrix_prep[word + '_left'] = (matrix_prep.words_left == word).astype(int)
        matrix_prep[word + '_right'] = (matrix_prep.words_right == word).astype(int)
        matrix_prep[word + '_in'] = matrix_prep[word + '_left'] + matrix_prep[word + '_right']

        #matrix_prep[word + '_score'] = matrix_prep.apply(lambda x: x['left_score'] if x['words_left'] == word else x['right_score'])

        matrix_prep.loc[matrix_prep.words_left == word, word + '_score'] = matrix_prep.left_score
        matrix_prep.loc[matrix_prep.words_right == word, word + '_score'] = matrix_prep.right_score

        matrix_prep.loc[matrix_prep.words_left == word, word + '_score_binary'] = matrix_prep.left_score_binary
        matrix_prep.loc[matrix_prep.words_right == word, word + '_score_binary'] = matrix_prep.right_score_binary
        matrix_prep.drop([word + '_left', word + '_right'], axis=1, inplace=True)


    pairwise_scores = pd.DataFrame()
    pairwise_binaries = pd.DataFrame()
    sense_mat = pd.DataFrame()

    for i, word1 in enumerate(words):
        scores_1 = matrix_prep[matrix_prep[word1 + '_in'] == 1]
        for j, word2 in enumerate(words):
            if word1 == word2:
                pairwise_scores.loc[word2, word1] = np.nan
                pairwise_binaries.loc[word2, word1] = np.nan
                sense_mat.loc[word2, word1] = np.nan
                continue
            #print(len(scores_1[scores_1.words_left == word2]))
            if len(scores_1[scores_1.words_left == word2]):
                scores_2 = scores_1[scores_1.words_left == word2].reset_index()
            else:
                scores_2 = scores_1[scores_1.words_right == word2].reset_index()

            pairwise_scores.loc[word2, word1] = scores_2.loc[0, word1 + '_score']
            pairwise_binaries.loc[word2, word1] = scores_2.loc[0, word1 + '_score_binary']
            sense_mat.loc[word2, word1] = scores_2.loc[0, 'sense_response']
 
    # In matrices, columns are the pespective of words while row is the perspective of opponents

    # Create aggregate dataframe to be averaged over participants
    df_aggregate = pd.DataFrame()

    df_aggregate[aggregate_columns[0]] = pairwise_scores.mean()
    df_aggregate[aggregate_columns[1]] = pairwise_scores.std()
    df_aggregate[aggregate_columns[2]] = pairwise_binaries.mean()
    df_aggregate[aggregate_columns[3]] = pairwise_binaries.std()
    df_aggregate[aggregate_columns[4]] = sense_mat.mean()
    df_aggregate[aggregate_columns[5]] = sense_mat.std()

    # Store data in arrays
    score_matrix[k, :, :] = pairwise_scores.to_numpy()
    win_matrix[k, :, :] = pairwise_binaries.to_numpy()
    sense_matrix[k, :, :] = sense_mat.to_numpy()
    aggregate_matrix[k, :, :] = df_aggregate.to_numpy()


    dfs_scores.append(pairwise_scores)
    dfs_wins.append(pairwise_binaries)
    dfs_sense.append(sense_mat)
    dfs_aggregate.append(df_aggregate)

    #print(sense_matrix)
    #if k > 2:
    #    break


# AGGREGATE DATASETS (averaged over participants)
# Generate global datasets
df_score = pd.DataFrame(index=pairwise_binaries.index, columns=pairwise_binaries.columns, data=score_matrix.mean(axis=0))
df_win = pd.DataFrame(index=pairwise_binaries.index, columns=pairwise_binaries.columns, data=win_matrix.mean(axis=0))
df_sense = pd.DataFrame(index=pairwise_binaries.index, columns=pairwise_binaries.columns, data=sense_matrix.mean(axis=0))

final_aggregate = pd.DataFrame(index=pairwise_scores.index, columns=df_aggregate.columns, data=aggregate_matrix.mean(axis=0))#.sort_values('mean_score', ascending=False)

# Save files
df_checks.to_excel('.\\data\\' + destination_folder + '\\datasets_excel\\sanity_checks.xlsx')
df_demos.to_excel('.\\data\\'  + destination_folder + '\\datasets_excel\\demographics.xlsx')
df_score.to_excel('.\\data\\' + destination_folder + '\\datasets_excel\\scores.xlsx')
df_win.to_excel('.\\data\\'  + destination_folder + '\\datasets_excel\\probability_win.xlsx')
df_sense.to_excel('.\\data\\'  + destination_folder + '\\datasets_excel\\sense.xlsx')
final_aggregate.to_excel('.\\data\\' + destination_folder + '\\datasets_excel\\summaries.xlsx')

df_checks.to_csv('.\\data\\'  + destination_folder + '\\datasets_csv\\sanity_checks.csv')
df_demos.to_csv('.\\data\\'  + destination_folder + '\\datasets_csv\\demographics.csv')
df_score.to_csv('.\\data\\' + destination_folder + '\\datasets_csv\\scores.csv')
df_win.to_csv('.\\data\\' + destination_folder + '\\datasets_csv\\probability_win.csv')
df_sense.to_csv('.\\data\\' + destination_folder + '\\datasets_csv\\sense.csv')
final_aggregate.to_csv('.\\data\\' + destination_folder + '\\datasets_csv\\summaries.csv')

# PARTICIPANT-WISE DATASETS
index_2d = pd.MultiIndex.from_product([part_ids, words], names=['part_id', 'word'])

score_dataset = score_matrix.reshape((len(part_ids) * len(words), len(words)))
win_dataset = win_matrix.reshape((len(part_ids) * len(words), len(words)))
sense_dataset = sense_matrix.reshape((len(part_ids) * len(words), len(words)))
aggregate_dataset = aggregate_matrix.reshape((len(part_ids) * len(words), len(aggregate_columns)))

df_score_full = pd.DataFrame(data=score_dataset, index=index_2d, columns=df_score.columns)
df_win_full = pd.DataFrame(data=win_dataset, index=index_2d, columns=df_score.columns)
df_sense_full = pd.DataFrame(data=sense_dataset, index=index_2d, columns=df_score.columns)
df_aggregate_full = pd.DataFrame(data=aggregate_dataset, index=index_2d, columns=aggregate_columns)


df_score_full.to_excel('.\\data\\'  + destination_folder + '\\datasets_excel\\scores_participants.xlsx')
df_win_full.to_excel('.\\data\\'  + destination_folder + '\\datasets_excel\\probability_win_participants.xlsx')
df_sense_full.to_excel('.\\data\\'  + destination_folder + '\\datasets_excel\\sense_participants.xlsx')
df_aggregate_full.to_excel('.\\data\\'  + destination_folder + '\\datasets_excel\\summaries_participants.xlsx')

df_score_full.to_csv('.\\data\\'  + destination_folder + '\\datasets_csv\\scores_participants.csv')
df_win_full.to_csv('.\\data\\'  + destination_folder + '\\datasets_csv\\probability_win_participants.csv')
df_sense_full.to_csv('.\\data\\'  + destination_folder + '\\datasets_csv\\sense_participants.csv')
df_aggregate_full.to_csv('.\\data\\'  + destination_folder + '\\datasets_csv\\summaries_participants.csv')

# Metadata for later processing
pd.Series(words).to_csv('.\\data\\'  + destination_folder + '\\datasets_csv\\word_list.csv', header=False, index=False)
pd.Series(part_ids).to_csv('.\\data\\'  + destination_folder + '\\datasets_csv\\participants_ids.csv', header=False, index=False)

# FULL TRIALS DATASET
trials_index = pd.MultiIndex.from_product([part_ids, task_data.index], names=['part_id', 'words'])

trials_dataset = pd.concat(agency_datasets).values

df_trials = pd.DataFrame(data=trials_dataset, index=trials_index, columns=task_data.columns)

df_trials.to_excel('.\\data\\'  + destination_folder + '\\datasets_excel\\trials_participants.xlsx')
df_trials.to_csv('.\\data\\'  + destination_folder + '\\datasets_csv\\trials_participants.csv')