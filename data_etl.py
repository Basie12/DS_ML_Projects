import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_file(file):
    '''loads csv to pd dataframe'''
    return pd.read_csv(file)

# def consolidate_data(df1, df2, key=None, left_index=False, right_index=False):
#     '''perform inner join to return only records that are present in both dataframes'''
#     return pd.merge(left=df1, right=df2, how='inner', on=key, left_index=left_index, right_index=right_index)

def inner_join_dataframes(df1, df2, join_key):
    """
    Perform an inner join on two DataFrames and return only the records that are present in both DataFrames.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    join_key (str): The name of the column to perform the inner join on.

    Returns:
    pd.DataFrame: A new DataFrame containing only the records present in both input DataFrames.
    """

    # Perform an inner join on the specified key
    result_df = pd.merge(df1, df2, on=join_key, how='inner')
    return result_df


# def clean_data(raw_df):
#     '''remove rows that contain salary <= 0 or duplicate job IDs'''
#     clean_df = raw_df.drop_duplicates(subset='jobId')
#     clean_df = clean_df[clean_df.salary>0]
#     return clean_df


def remove_rows_with_salary_and_duplicates(df):
    """
    Remove rows from a DataFrame that have a salary less than or equal to 0 or duplicate job IDs.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A new DataFrame with rows removed based on the specified conditions.
    """

    # Remove rows with salary less than or equal to 0
    df = df[df['salary'] > 0]

    # Remove rows with duplicate job IDs
    df = df.drop_duplicates(subset='jobId', keep='first')

    return df


# def one_hot_encode_feature_df(df, cat_vars=None, num_vars=None):
#     '''performs one-hot encoding on all categorical variables and combines result with continous variables'''
#     cat_df = pd.get_dummies(df[cat_vars])
#     num_df = df[num_vars].apply(pd.to_numeric)
#     return pd.concat([cat_df, num_df], axis=1)#,ignore_index=False)


def one_hot_encode_and_combine(df, categorical_columns):
    """
    Perform one-hot encoding on categorical variables and combine the results with continuous variables.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing both categorical and continuous variables.
    categorical_columns (list): List of column names that are categorical variables.

    Returns:
    pd.DataFrame: A new DataFrame with one-hot encoded categorical variables concatenated with continuous variables.
    """

    # Ensure that the categorical_columns exist in the DataFrame
    missing_cols = set(categorical_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")

    # Create a new DataFrame for one-hot encoded categorical variables
    encoded_categorical = pd.get_dummies(df[categorical_columns], prefix=categorical_columns)

    # Concatenate the one-hot encoded categorical DataFrame with the original continuous DataFrame
    combined_df = pd.concat([encoded_categorical, df.drop(categorical_columns, axis=1)], axis=1)

    return combined_df


def get_target_df(df, target):
    '''returns target dataframe'''
    return df[target]

def train_model(model, feature_df, target_df, num_procs, mean_mse, cv_std):
    neg_mse = cross_val_score(model, feature_df, target_df, cv=2, n_jobs=num_procs, scoring='neg_mean_squared_error')
    mean_mse[model] = -1.0*np.mean(neg_mse)
    cv_std[model] = np.std(neg_mse)

def print_summary(model, mean_mse, cv_std):
    print('\nModel:\n', model)
    print('Average MSE:\n', mean_mse[model])
    print('Standard deviation during CV:\n', cv_std[model])

def save_results(model, mean_mse, predictions, feature_importances):
    '''saves model, model summary, feature importances, and predictions'''
    with open('model.txt', 'w') as file:
        file.write(str(model))
    feature_importances.to_csv('feature_importances.csv') 
    np.savetxt('predictions.csv', predictions, delimiter=',')

print("Functions are good to run")