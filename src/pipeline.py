import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


class Ensemble(object):
    def __init__(self, models, train_pos, train_neg, val_pos, val_neg):
        self.models = models
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.val_pos = val_pos
        self.val_neg = val_neg
        self.trained_models = defaultdict(list)
        self.target_var = 'some_target_str'


    def evaluate(self, y_true, y_hat, plot=False):
        """
        Evaluates the precision and recall of the predictions.
        Plots a precision-recall graph if plot==True
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_hat)
        average_precision = average_precision_score(y_true, y_hat)

        if plot:
            plt.figure()
            plt.step(recall, precision, color='b', alpha=0.2,
                     where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2,
                             color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
                      average_precision))
            plt.show()
        return precision, recall, thresholds

    def predict_proba(self, X, model_name):
        """
        Given trained models, predict outcome of X
        """
        predictions = []
        for model in self.trained_models[model_name]:
            y_hat = model.predict_proba(X)
            predictions.append(y_hat)
        return np.mean(np.vstack(predictions), axis=0)

    def run(self):
        """
        Ensembles the models
        """
        self.evals = {}  # save each model run

        # create X_val, y_va
        self.val_pos[self.target_var] = 1
        self.val_neg[self.target_var] = 0
        combined_set = pd.concat((self.val_pos, self.val_neg)).sample(frac=1)
        self.X_val = combined_set.drop([self.target_var], axis=1).values
        self.y_val = combined_set[self.target_var].values  # create validation set


        # 1. Iterate over each model
        for model_key in self.models:
            print("Trying model: {}".format(model_key))
            self.evals[model_key] = {'predictions': []}
            
            # 2. For each model, train on subset training sets
            num_sets = len(self.train_pos)
            for i, train_key in enumerate(self.train_pos):  # iterate over the training sets
                self.evals[model_key][train_key] = {'model': None,
                                               'y_hat': None
                                               }  # store model and predictions
                print("Now training model #{}/{}...".format(i + 1, num_sets))
                
                set_pos = self.train_pos[train_key].copy()
                set_pos[self.target_var] = 1
                set_neg = self.train_neg[train_key].copy()
                set_neg[self.target_var] = 0
                combined_set = pd.concat((set_pos, set_neg)).sample(frac=1)
                X, y = combined_set.drop([self.target_var], axis=1), combined_set[self.target_var]  # create training set
                X = X.values
                y = y.values
                
                model = self.models[model_key]  # load model
                start = time.time()
                model.fit(X, y)  # train model
                end = time.time()
                print("Done in {}s.\n".format(round(end - start, 2)))
                print("Now predicting model #{}...".format(i + 1))
                start = time.time()
                y_hat = model.predict_proba(self.X_val)  # predict model
                end = time.time()
                print("Done in {}s.\n".format(round(end - start, 2)))

                # add model, y_hat, and metrics to dictionary
                self.evals[model_key][train_key]['model'] = model
                self.evals[model_key][train_key]['y_hat'] = y_hat
                self.evals[model_key]['predictions'].append(y_hat)
                self.trained_models[model_key].append(model)
        
            # 3. Take the average of predictions for all models
            model_predictions = np.mean(\
                                np.vstack(\
                                self.evals[model_key]['predictions']), axis=0)
            
            # 4. Evaluate the performance of the ensemble
            p, r, t = self.evaluate(self.y_val,
                                    model_predictions,
                                    plot=True)
            self.precision, self.recall, self.thresholds = p, r, t