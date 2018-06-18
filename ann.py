# -*- coding: utf-8 -*-
"""
Created on Sat May  5 11:44:31 2018

@author: mrclx
"""

import pandas as pd
import numpy as np
import os
import shutil
import tensorflow as tf
import skopt
from skopt.utils import use_named_args

# global variable for the accuracy of the optimal model.
Best_loss = 1e+30
# gloabl variable for default parameters.
Default_parameters = [9e-10, 1, 412, tf.nn.sigmoid, 0.4]
# global DataFrame containing, hyper-parameter values and corresponding loss.
Hparams_log = pd.DataFrame(columns = ["learing_rate",
                                      "num_layers",
                                      "num_nodes",
                                      "activation",
                                      "num_dropout",
                                      "loss",
                                      "avg_loss"])

MODEL_DIR = "C:/Users/mrclx/Google Drive/TWE/6. Semester/Informations- und Kommunikationstechnik/IT 3 - Artificial Intelligence/Prognosemodell/code"

def input_set(features, labels):
    
    # import features and labels.
    features = pd.read_csv(features, index_col = 0)
    labels = pd.read_csv(labels, header = None, index_col = 0)
    
    # reset index.
    features = features.reset_index(drop = True)
    labels = labels.reset_index(drop = True)
    
    # convert features and labels to tensors.
    features_tensor = {k: tf.constant(features[k].values) for k in features}
    labels_tensor = tf.constant(labels.values)
    
    return (features_tensor, labels_tensor)
    
def input_dataset(features, labels):
    
    # get features and labels.
    features, labels = input_set(features, labels)
    
    # covert features and lables to datasets.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    
    # shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(330)
    dataset = dataset.repeat(100)
    dataset = dataset.batch(66)
    
    # create iterator.
    iterator = dataset.make_one_shot_iterator()
    
    return iterator

def feature_cols():
    
    # initialize list of column names. 
    col_names = []
    
    # get dictionary of features.
    features, labels = input_set("_features_norm.csv", "_labels_norm.csv")
    
    for key in features.keys():
        
        # create list of feature column names.
        col_names.append(tf.feature_column.numeric_column(key = key))
        
    return col_names

def input_training():
    
    training_set = input_dataset("_features_norm.csv", "_labels_norm.csv")
    
    return training_set

def input_eval():
    
    training_set = input_dataset("_features_eval_norm.csv", "_labels_eval_norm.csv")
    
    return training_set

def train_ann(training_steps = 330):
    
    with tf.Session() as sess:
        
        # receive data for training.
        for i in range(1, 67):
                
            value = sess.run(fetches = input_training().get_next())
    
        # build DNN model.
        model = tf.estimator.DNNRegressor(feature_columns=feature_cols(),
                                              hidden_units = [10, 10],
                                              model_dir = MODEL_DIR,
                                              activation_fn = tf.nn.sigmoid)
    
        # train model.
        results = model.train(input_fn = lambda: value,
                              max_steps = training_steps)
    
        return results

def hparams_dimensions() :
    
    # Set parameters for hyper-parameter optimization.
    dim_learning_rate = skopt.space.Real(low = 1e-12, high = 1e-4, prior = 'log-uniform',
                         name='learning_rate')
    dim_num_dense_layers  = skopt.space.Integer(low = 1, high = 10, name = 'num_dense_layers')
    dim_num_dense_nodes = skopt.space.Integer(low = 5, high = 512, name = 'num_dense_nodes')
    dim_activation = skopt.space.Categorical(categories = [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh],
                             name='activation')
    dim_dropout = skopt.space.Real(low = 0.2, high = 0.8, name = "num_dropout")
    
    dimensions = [dim_learning_rate, dim_num_dense_layers,
                  dim_num_dense_nodes, dim_activation, dim_dropout]
    
    return dimensions

def hparams_log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, activation, num_dropout):

    # The dir-name for the TensorBoard log-dir.
    s = "./logs/lr_{0:.1e}_layers_{1}_nodes_{2}_{3}_dropout_{4:.1f}/"
    
    if activation == tf.nn.sigmoid:
        act = "sigmoid"
        
    elif activation == tf.nn.tanh:
        act = "tanh"
        
    elif activation == tf.nn.relu:
        act="relu"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       act,
                       num_dropout)

    return log_dir

@use_named_args(dimensions = hparams_dimensions())
def hparams_fitness(learning_rate, num_dense_layers,
            num_dense_nodes, activation, num_dropout):
    """
    Hyper-parameters:
    learning_rate:      Learning-rate of the optimizer.
    num_dense_layers:   Number of dense layers.
    num_dense_nodes:    Number of nodes in each dense layer.
    activation:         Activation function for all layers.
    num_dropout:        Dropout of the optimizer.
    """

    with tf.Session() as sess:
        
        # Print the hyper-parameters.
        print()
        print('learning rate: {0:.1e}'.format(learning_rate))
        print('num_dense_layers:', num_dense_layers)
        print('num_dense_nodes:', num_dense_nodes)
        print('activation:', activation)
        print("num_dropout: {0:.1f}".format(num_dropout))
        print()
    
        # Create list for hidden_units.
        model_size = [num_dense_nodes for layer in range(1, num_dense_layers + 1)]
            
        # Create regression model.
        model = tf.estimator.DNNRegressor(feature_columns = feature_cols(),
                                              hidden_units = model_size,
                                              activation_fn = activation,
                                              dropout = num_dropout)
   
        # Receive data for training.
        for i in range(1, 67):
            value = sess.run(fetches = input_training().get_next())   
            
        # Train the model.
        model.train(input_fn = lambda: value, max_steps = 33000)
        
        # Receive data for evaluation.
        for i in range(1, 6):
            value = sess.run(fetches = input_eval().get_next())
        
        # Evaluate the model.
        eval_results = model.evaluate(input_fn = lambda: value, steps = 5)
        
        # Get the accuracy on the validation set.
        avg_loss = eval_results["average_loss"]

        # Print the loss.
        print("\nAverage Loss: "+ str(avg_loss))
        print()
        
        # Save the model if it improves on the best-found performance.
        # We use the global keyword so we update the variable outside
        # of this function.
        global Best_loss
        
        # Update the best loss value if necessary.
        if avg_loss < Best_loss:
            Best_loss = avg_loss
            
        # Create directory for log files.
        log_dir = hparams_log_dir_name(learning_rate, num_dense_layers,
                                       num_dense_nodes, activation,
                                       num_dropout)
        os.makedirs(log_dir)
            
        # Move log files from temporary folder to log_dir.
        try:
            shutil.move(model.model_dir, log_dir)
            
        except OSError:
            pass
        
        # Update hparams_log.
        global Hparams_log
        new_log = pd.DataFrame({"learing_rate": [learning_rate],
                                "num_layers": [num_dense_layers],
                                "num_nodes": [num_dense_nodes],
                                "activation":[activation],
                                "num_dropout": [num_dropout],
                                "loss":[eval_results["loss"]],
                                "avg_loss": [avg_loss]})
        Hparams_log = pd.concat([Hparams_log, new_log], ignore_index = True)
        
        # Save log on drive.
        Hparams_log.to_csv("./logs/hparams_log.csv")
        
        # Delete the model with these hyper-parameters from memory.
        del model
    
    # Clear the session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    sess.close()
    
    return avg_loss

def hparams_exec():
    
    # Disable warning about AVX2.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Perform hyper-parameter optimization. Minimizes loss.
    search_result = skopt.gp_minimize(func = hparams_fitness,
                                dimensions = hparams_dimensions(),
                                acq_func = 'EI',
                                n_calls = 40,
                                x0 = Default_parameters)
    
    skopt.plot_convergence(search_result)
    

def predict_valuation(predict_x):
   
    mod_dir = "./logs/lr_6e-09_layers_1_nodes_5_tanh_dropout_1/tmp0fo3bfz0/"
        
    # initialize trained model.
    model = tf.estimator.DNNRegressor(feature_columns=feature_cols(),
                                              hidden_units = [5],
                                              activation_fn = tf.nn.tanh,                   
                                              model_dir = mod_dir)
    
    # calculate enterprise value for company with input metrics.  
    predictions = model.predict(input_fn = lambda:
            tf.data.Dataset.from_tensor_slices(predict_x).batch(len(predict_x)))
        
    # print estimated enterprise value.
    template = ('\nValuation is ${0:.2f}.')
        
    for pred in predictions:
        valuation = 1.041918e+11 * pred["predictions"][0] + 6.899872e+10
        print(template.format(valuation))
        
        return valuation
        
def test_prediction():
            
    data = pd.DataFrame(columns = ["dividendyield", "earningsyield", "evtoebit",
                               "evtoebitda", "evtofcff", "evtoinvestedcapital", "evtonopat",
                               "evtoof", "pricetoearnings", "ebitdagrowth", "freecashflow",
                               "revenuegrowth"])

    data = {k: [2 * np.random.random() - 1] for k in data}
    print(data)

    predict_valuation(data)
    
def demo_prediction():
    """Demo with data about Airbus SE. Date: 15-06-18."""
    
    features = ["dividendyield", "earningsyield", "evtoebit",
                "evtoebitda", "evtofcff", "evtoinvestedcapital", "evtonopat",
                "evtoof", "pricetoearnings", "ebitdagrowth", "freecashflow",
                "revenuegrowth"]
    
    # metrics about EADSY. Date: 15-06-18.
    EADSY = np.array([[0.01654], [0.03214], [16.76], [11.16], [43.52], [5.72],
                       [36.41], [19.61], [31.11], [0.9364], [2.1e+9], [0.028]])
    
    print(EADSY)
    
    # mean and std to compute z-score.
    features_mean = np.array([[2.028145e-02], [5.066418e-02], [2.763982e+01],
                              [1.498835e+01], [9.076441e+01], [9.516009e+00],
                              [4.756403e+01], [0.0], [4.888499e+01],
                              [3.240874e-01], [2.601085e+09], [9.792685e-02]])
    features_std = np.array([[2.150512e-02], [4.071772e-02], [5.808228e+01],
                             [9.666688e+00], [2.488084e+02], [5.323648e+01],
                             [1.388038e+02], [19.61], [1.637779e+02],
                             [2.453869e+00], [4.715736e+09], [1.463448e-01]])
    
    # normalization.
    EADSY_normed = (EADSY - features_mean) / features_std
    
    # build dict.
    EADSY_dict = {k: EADSY_normed[features.index(k)] for k in features}
    print(EADSY_dict)
    
    # EADSY market cap. Date: 15-06-18.
    EADSY_marketcap_150618 = 77600000000
    
    ev = predict_valuation(EADSY_dict)
    
    print("Airbus SE Market Cap (15-06-18): ${0:.2f}.".format(EADSY_marketcap_150618))
        
    # Infer recommendation. If market cap is higher than valuation: sell. Else: buy.
    if EADSY_marketcap_150618 >= ev:
        print()
        print("SELL!!!")
        print()
        
    else:
        print()
        print("BUY!!!")
        print()
    
    
