import pandas as pd
import numpy as np
import catboost as cb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class CatBoost():
  """
  This model fits the CatBoost Algorithm, an algorithm developed by
  Yandex that builds on XGBoost, a very popular gradient boosting algorithm.

  It supports an input of the ```factor_column_indices``` parameter,
  which allows the algorithm to take special care of factor/ordinal 
  columns indexed by the parameter.
  """
  def __init__(self):
  # set parameters
    self.iterations=30
    self.learning_rate=0.2
    self.depth=12
    self.loss_function='Logloss'
    self.class_weights=[1.0, 100.0]
    self.random_seed=42
    self.verbose=True
    self.factor_column_indices = []
  
  def fit(self, X, y):
    model = cb.CatBoostClassifier(iterations=self.iterations,
                                  learning_rate=self.learning_rate,
                                  depth=self.depth,
                                  loss_function=self.loss_function,
                                  class_weights=self.class_weights,
                                  random_seed=self.random_seed,
                                  verbose=self.verbose)
    model.fit(X,
              y,
              cat_features=self.factor_column_indices,
              verbose=self.verbose)
    self.model = model

  def predict_proba(self, X_test):
    y_hat = self.model.predict_proba(X_test)
    return y_hat[:, 1]


class NeuralNetwork():
  """
  This model fits a neural network of the given parameters:
      --------------------------------------
      | TYPE  |  VARIABLE NAME   |  VAL    |
      --------------------------------------
      @param  |  batch_size      |  32     |
      @param  |  hidden1_units   |  24     |
      @param  |  hidden2_units   |  6      |
      @param  |  learning_rate   |  0.005  |
      @param  |  n_epochs        |  10     |
      @param  |  class_weights   |  1.0    |
      --------------------------------------

    By default, this network has two dense ReLU activation layers.
    It could be improved with, for example:

        1. Different activation functions
        2. Dropout
        3. More layers

  """
  def __init__(self):
    # 1. Define the Model Parameters
    self.batch_size = 32
    self.hidden1_units = 128
    self.hidden2_units = 24
    self.learning_rate = 0.005
    self.n_epochs = 10

    # construct weights tensor
    # if you want to weight by proportion:
    # ratio = None  # replace with y.mean()
    # class_weights = tf.constant( [ 1.0 - ratio, ratio ] )  # needs to be reshaped to batch size.

    # if not reshaped, use this to assign default equal rating.
    self.class_weights = 1.0

  def fit(self, X, y):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.ERROR)
    if not isinstance(X, np.ndarray):
        X = np.array(X.copy())
    if not isinstance(y, np.ndarray):
        y = np.array(y.copy())
    self.X = X.copy()
    self.y = y.copy()
    self.X = StandardScaler().fit_transform(X)
    if y.ndim <= 1:
        self.y = y[:, np.newaxis]
    self.y = self.y[:, 0]
    new_column = (self.y==0).astype(int)
    self.y = np.vstack((self.y, new_column)).T
    self.output_dimension = self.y.shape[1]
    self.input_dimension = self.X.shape[1]

    layers_custom_model = tf.contrib.learn.Estimator(
                            model_fn=self.layers_custom_model_fn)
    self.model = layers_custom_model
    max_steps = 15000


    self.model.fit(input_fn=lambda: self.input_fn(self.X, self.y),
                   steps=max_steps)

  def predict_proba(self, X_test):
    y_test = np.empty(X_test.shape[0])
    y_test[:] = np.nan
    y_hat = self.model.predict(input_fn=lambda: self.input_fn(X_test,
                                                              y_test),
                               batch_size=None)
    return y_hat

  def input_fn(self, data, labels):
    # 2. Define the data input function
    """
    This is an input function for batches of data. Here's how it works:
        1. Takes in data and labels
        2. Replicates it ("slices")
        3. Turns that into a dictionary
        4. Splits that dictionary into batches
        5. returns the batches dict
        6. returns the labels for each batch
    """
    input_data = tf.constant(data, shape=data.shape, verify_shape=True, dtype=tf.float32)
    input_labels = tf.constant(labels, shape=labels.shape, verify_shape=True, dtype=tf.float32)
    data, label = tf.train.slice_input_producer(
        [input_data, input_labels],
        num_epochs=self.n_epochs)
    dataset_dict = dict(data=data, labels=label)
    batch_dict = tf.train.batch(
        dataset_dict, self.batch_size, allow_smaller_final_batch=True)
    batch_labels = batch_dict.pop('labels')
    return batch_dict, batch_labels

  # 4. Define the Model.
  def layers_custom_model_fn(self,
                             features,
                             targets,
                             mode,
                             params):
    # 1. Configure the model via TensorFlow operations (using tf.layers). Note how
    #    much simpler this is compared to defining the weight matrices and matrix
    #    multiplications by hand.
  

    # Note: this is a two-layer DNN, with custom numbers of hidden units.
    self.hidden_layer1 = tf.layers.dense(inputs=features['data'],
                                         units=self.hidden1_units,
                                         activation=tf.nn.relu)
    self.hidden_layer2 = tf.layers.dense(inputs=self.hidden_layer1,
                                         units=self.hidden2_units,
                                         activation=tf.nn.relu)
    self.output_layer = tf.layers.dense(inputs=self.hidden_layer2,
                                         units=self.output_dimension,
                                         activation=tf.nn.sigmoid)
    
    # 2. Define the loss function for training/evaluation
    # use the softmax x-entropy for the binary classification task.
    self.loss = tf.losses.softmax_cross_entropy(weights=self.class_weights,
                                           onehot_labels=targets,
                                           logits=self.output_layer)
    
    # 3. Define the training operation/optimizer
    # one might experiment with other optimizers.
    train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=self.learning_rate,
            optimizer="SGD")
    
    # 4. Generate predictions
    predictions_dict = {
        "classes":       tf.argmax(input=self.output_layer, axis=1),
        "probabilities": tf.nn.softmax(self.output_layer, name="softmax_tensor"), 
        "logits":        self.output_layer,
    }
    
    # Define eval metric ops.
    accuracy = tf.metrics.accuracy(
        labels=tf.argmax(input=targets, axis=1),
        predictions=tf.argmax(input=self.output_layer, axis=1))
    precision = tf.metrics.precision(
        labels=tf.argmax(input=targets, axis=1),
        predictions=tf.argmax(input=self.output_layer, axis=1))
    recall = tf.metrics.recall(
        labels=tf.argmax(input=targets, axis=1),
        predictions=tf.argmax(input=self.output_layer, axis=1))
    eval_metric_ops = {"accuracy":  accuracy,
                       "precision": precision,
                       "recall": recall
                      }

    # 5. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object
    return tf.contrib.learn.ModelFnOps(mode=mode,
                                       predictions=predictions_dict,
                                       loss=self.loss,
                                       train_op=train_op,
                                       eval_metric_ops=eval_metric_ops)
