import pandas as pd
import numpy as np
import time

from SMOTE import *
import load_and_clean_data

# #load csv already cleaned and with nulls dropped
# df = pd.read_csv('clean_data.csv', sep=',')

def create_datasets(input_df,
                    target_var,
                    num_val_pos,
                    num_val_neg,
                    num_train_pos,
                    num_train_neg,
                    num_smote=None,
                    num_train_sets=5,
                    replace=True,
                    SMOTE_random_sample=None):
    """
    Automatically creates validation and train/test sets for you.
    
    INPUTS
      --------------------------------------------------------------------------
      | TYPE  |  VARIABLE NAME   |  DESCRIPTION                                |
      --------------------------------------------------------------------------
      pd.df   |  df              |  Starting dataframe                         |
      int     |  num_val_pos     |  # pos. values in validation dataframe      |
      int     |  num_val_neg     |  # neg values in validation dataframe       |
      int     |  num_train_pos   |  # pos. values in each train dataframe      |
                                    if == "all", then you use all non-validation
                                    examples for each training set leading to an
                                    oversampling factor of num_train_sets.
      int     |  num_train_neg   |  # neg. values in each train dataframe      |
      int     |  num_train_sets  |  # of training setss desires                |
      --------------------------------------------------------------------------

    RETURNS:
      --------------------------------------------------------------------------
      | TYPE  |  VARIABLE NAME |  DESCRIPTION                                  |
      --------------------------------------------------------------------------
      dict    |  train_pos     |  dict of dataframes of positive training obs  |
      dict    |  train_neg     |  dict of dataframes of negative training obs  |
      pd.df   |  val_pos       |  dataframe of positive testing obs            |
      pd.df   |  val_neg       |  dataframe of negative testing obs            |
      --------------------------------------------------------------------------
    """
    total_positives = (input_df[target_var]).sum()

    # if num_val_pos < 1:
    #     num_val_pos = int(num_val_pos * total_positives)
    
    # if num_train_pos < 1:
    #     num_train_pos = int(num_train_pos * total_positives)

    # if num_val_neg < 1:
    #     num_val_neg = int(num_val_neg * total_positives)
    
    # if num_train_neg < 1:
    #     num_train_neg = int(num_train_neg * total_positives)

    # # After defining num_val_positives, use the rest for training
    # if not num_train_pos:
    #     num_train_pos = total_positives - num_val_pos


    # Get Positive, Negative Indices
    positive_indices = input_df[ input_df[ target_var ] == 1 ].index.tolist()
    num_positives = len(positive_indices)

    negative_indices = input_df[ input_df[ target_var ] == 0 ].index.tolist()
    num_negatives = len(negative_indices)

    # Create validation set
    print("Creating Validation sets...")
    start = time.time()

    val_positive_indices = np.random.choice(positive_indices,
                                            num_val_pos,
                                            replace=False)
    val_pos = input_df.copy().iloc[ val_positive_indices, : ]

    val_negative_indices = np.random.choice(negative_indices,
                                            num_val_neg,
                                            replace=False)
    val_neg = input_df.copy().iloc[ val_negative_indices, : ]

    # Remove Validation Set Elements from Train/Test Set Elements
    positive_indices = list(set(positive_indices)\
                       .difference(set(val_positive_indices)))
    negative_indices = list(set(negative_indices)\
                       .difference(set(val_negative_indices)))
    end = time.time()
    print("Completed in {}s\n".format(round(end - start, 1)))

    ## SMOTE new samples
    remaining_indices = positive_indices + negative_indices
    remaining_df = input_df.iloc[remaining_indices, :].reset_index(drop=True)
    if num_smote is not None and num_smote > 0:
      start = time.time()
      print("SMOTEing synthetic examples...")
      X_pos = remaining_df[remaining_df['target_var'] == 1].drop([target_var], axis=1)
      if SMOTE_random_sample is not None and SMOTE_random_sample > 0:
        X_pos = X_pos.sample(SMOTE_random_sample)
      smoter = SMOTE()
      X_synth = smoter.generate(X_pos,
                                None,
                                num_smote,
                                False,
                                custom_SMOTE.match_columns,
                                custom_SMOTE.smote_columns,
                                )
      y_synth = np.ones(X_synth.shape[0]).reshape(-1, 1)
      synths = pd.DataFrame(np.hstack((X_synth, y_synth)),
                            columns=remaining_df.columns)
      new_df = pd.concat((remaining_df, synths))
      new_df = new_df.convert_objects()
      end = time.time()
      print("Completed in {}s\n".format(round(end - start, 1)))
    else:
      new_df = remaining_df.copy()

    # Get indices of remaining samples
    positive_indices = new_df[ new_df[ target_var ] == 1 ].index.tolist()
    negative_indices = new_df[ new_df[ target_var ] == 0 ].index.tolist()

    # Create Train/Test Set Values
    print("Creating Train/Test sets...")
    start = time.time()

    if num_train_pos == 'all':
      train_positives = np.array(positive_indices)[:, np.newaxis].T
      train_positives = np.repeat(train_positives, num_train_sets, axis=0)

    else:
      train_positives = np.random.choice(positive_indices,
                                   size=(num_train_sets,
                                   num_train_pos))

    train_negatives = np.random.choice(negative_indices,
                                       size=(num_train_sets,
                                       num_train_neg))

    # Return Dataframes
    print("Returning Dataframes...")
    train_pos, train_neg = {}, {}
    for i in range(num_train_sets):
        set_name = "set_{}".format(i + 1)
        train_pos[ set_name ] = new_df.iloc[ train_positives[i] , : ]
        train_neg[ set_name ] = new_df.iloc[ train_negatives[i] , : ]

    end = time.time()
    print("Completed in {}s\n".format(round(end - start, 1)))
    print("Done")
    return train_pos, train_neg, val_pos, val_neg

def get_factor_columns(training_set):
  """
  If in the process of creating a training set the variable type is lost,
  this re-outputs the true factor columns list of indices.
  Input a training set after running create_datasets(),
  i.e.:

    get_factor_columns(train_neg['set_1'])
    
  """
  global_set = set(range(len(train_pos['set_1'].columns)))
  for set_name in train_pos:
    factors = set(load_and_clean_data.get_factor_columns(train_pos[set_name]))
    global_set = global_set.intersection(factors)
  return list(global_set)