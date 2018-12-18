import pandas as pd
import numpy as np
import random

def load_data(filepath,
              factor_columns=False,
              nulls_to_negatives=False,
              **kwargs):
    """
    This loads and cleans all the data.
    Returns the final dataframe to apply sampling.py to.
    """
    dataframe = pd.read_csv(filepath, **kwargs)
    value_fixed = clean_data(dataframe)
    if nulls_to_negatives:
        value_fixed = replace_nulls_to_negatives(value_fixed)
        if factor_columns:
            cols = get_factor_columns(value_fixed)
            return value_fixed, cols
        else:
            return value_fixed
    else:
        nulls_removed, nulls = remove_nulls(value_fixed, return_unclean=True)
        if factor_columns:
            cols = get_factor_columns(nulls_removed)
            return nulls_removed, cols
        else:
            return nulls_removed, nulls


def clean_data(df, binary_replace_dict, drop_cols):
    """
    Converts binary string values into integers for splitting.,
    and replaces various null values with np.nan values.
    Finally, it drops uneccesary columns if present.

    """
    # replace Y/N with 1/0
    df = df.copy()
    df = df.replace(replace_dict)
    try:
        df.drop(['X'], axis=1, inplace=True)
    except:
        pass
    for col in drop_cols:
        try:
            df.drop([col], axis=1, inplace=True)
        except:
            pass

    return df

def remove_nulls(df, target_var, col_null_threshold, return_unclean=False):
    """
    Removes null values.

    First, it eliminates columns with > col_null_threshold null values.
    Then of the remaining rows, it drops rows with any null value.

    This leads to only a few rows being dropped in total.

    """
    # subset columns with fewer than col_null_threshold null values
    mask = pd.isnull(df).sum() < col_null_threshold
    clean_cols = mask[mask].index
    unclean_cols = list(set(mask.index).difference(set(clean_cols)))
    unclean_cols = unclean_cols + [target_var]
    subset_df = df.copy()[clean_cols]
    # then remove null rows
    index_mask = pd.isnull(subset_df).sum(axis=1) == 0
    subset_df = subset_df.loc[index_mask, :]
    clean_df = subset_df.reset_index(drop=True)
    if not return_unclean:
        return clean_df
    else:
        return clean_df, df.copy()[unclean_cols].reset_index(drop=True)

def get_factor_columns(df, categorical_threshold, target_var):
    """
    Returns a list of integers. Each integer represents
    the index of a column that is a categorical/factor column. This
    is useful for the CatBoost Algorithm.
    """
    factor_columns = []
    objects = df.head().select_dtypes(include=['object']).columns
    # first, add all object-type columns
    for col in objects:
        factor_columns.append(col)
    # then, add all non-object-type columns with only 1s, 0s, or nulls.
    non_objects = df.select_dtypes(exclude=['object'])
    uniques = non_objects.apply(pd.Series.nunique) < categorical_threshold 
    uniques = list(uniques[uniques].index)
    factor_columns += uniques
    factor_column_indices = map(lambda x: list(df.columns).index(x), factor_columns)
    try:
        target_column_index = list(df.columns).index(target_var)
        factor_column_indices.remove(target_column_index)
    except:
        print("Target column not a factor")
    return factor_column_indices

def replace_nulls_to_negatives(df):
    return df.replace(np.nan, -1)

def load_subset(n=10000000, s=5000000):
    skip = sorted(random.sample(xrange(n),n-s))
    try:
        skip.remove(0)
    except:
        pass
    df = load_data('../data/data.csv',
                   nulls_to_negatives=True,
                   sep=' ',
                   header=0,
                   skiprows=skip)
    return df