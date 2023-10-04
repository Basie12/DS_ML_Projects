#
from data_etl import load_file, remove_rows_with_salary_and_duplicates
from data_etl import  inner_join_dataframes,  one_hot_encode_and_combine, get_target_df
from sklearn.utils import shuffle
# define inputs
train_feature_file = 'data/train_features.csv'
train_target_file = 'data/train_salaries.csv'
test_feature_file = 'data/test_features.csv'

#define variables
categorical_vars = ['companyId', 'jobType', 'degree', 'major', 'industry']
numeric_vars = ['yearsExperience', 'milesFromMetropolis']
target_var = 'salary'

#load data
print("Loading data")
feature_df = load_file(train_feature_file)
target_df = load_file(train_target_file)
test_df = load_file(test_feature_file)

#consolidate training data
raw_train_df = inner_join_dataframes(feature_df, target_df, 'jobId')

#clean, shuffle, and reindex training data -- shuffling improves cross-validation accuracy
clean_train_df = shuffle(remove_rows_with_salary_and_duplicates(raw_train_df)).reset_index()

#encode categorical data and get final feature dfs
print("Encoding data")
feature_df = one_hot_encode_and_combine(clean_train_df, categorical_vars)
test_df = one_hot_encode_and_combine(test_df, categorical_vars)

#get target df
target_df = get_target_df(clean_train_df, target_var)