from load_and_clean_data import *
from sampling import *
from models import *
from pipeline import *

def main():
    # 1. Load Data
    data, factor_cols = load_and_clean_data.load_data("FILEPATH")
    # data = load_and_clean_data.load_subset(s=5000000)  # loads a 5-million row subset of the data

    # 2. Split the dataframe into training & validation sets
    num_val_pos = 10000  # choose some number of positive samples in your validation set
    num_val_neg = ((1. / 0.006) * num_val_pos)  # as above, for negatives; here, the ratio of +/- is the same as in the original data
    num_train_pos = 50000  # set to "all" if you want to use all non-validation positive samples in each subset
    num_train_neg = 500000  # arbitrary; here set to a ~10-to-1 ratio negative-to-positive
    num_smote = int(0.5 * num_train_pos)  # arbitrary; here set to 50% the number of non-SMOTE samples
    num_train_sets = 10

    # train_pos: dict with keys like 'set_#', where each value is a matrix of
    # observations of positive training examples.
    #
    # train_neg, val_pos, val_neg are like train_pos but for negative training
    # examples, positive validation examples, and negative validation examples.
    train_pos, train_neg, val_pos, val_neg = sampling.create_datasets(input_df=data,
                                                                      num_val_pos=num_val_pos,
                                                                      num_val_neg=num_val_neg,
                                                                      num_train_pos=num_train_pos,
                                                                      num_train_neg=num_train_neg,
                                                                      num_smote=num_smote,
                                                                      num_train_sets=num_train_sets)

    factor_columns = sampling.get_factor_columns(train_pos['set_1'])  # gets the correct factor column indices for use with CatBoost

    # 3. Load a list of models
    m = models.CatBoost()  # set to CatBoost; can extend arbitarily to other models or to list of models
    m.class_weights = [0.5, 1]  # if you wish to redefine the class weights (penalty of wrong classification for label
    cat.factor_column_indices = factor_columns
    models_to_use = {"CatBoost": m}

    # 4. Run the model
    evals = ensemble(models_to_use, train_pos, train_neg, val_pos, val_neg)
    pipe.run()

    # 5. Evaluate the model
    y_hat = pipe.predict_proba(X_val, 'CatBoost')
    return pipe.evaluate(y_val, y_hat, plot=True)

if __name__ == "__main__":
    main()