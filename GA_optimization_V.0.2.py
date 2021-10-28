# import tensorflow as tf
# tf.__version__

import numpy as np

import pandas as pd

# pip install geneticalgorithm

from geneticalgorithm import geneticalgorithm as ga

import tensorflow as tf
tf.__version__

import time

def nn_model(batch_size, num_neurons, active_funcs, learning_rate, beta_1, beta_2, epsilon):
    import tensorflow as tf
    # tf.__version__

    import numpy as np

    import pandas as pd
    df1 = pd.read_csv("pricing_final.csv")
    
    df = df1.head(200000)
    # x data
    x_cat_1 = np.array(df.sku) #categorical
    x_cat_2 = np.array(df.order) #categorical
    x_cat_3 = np.array(df.category) #categorical


    x_num_4 = np.array(df.price) #numeric
    x_num_5 = np.array(df.duration) #numeric

    #y data
    y = np.array(df.quantity)

    len(y)

    input_1_cat = tf.keras.layers.Input(shape=(1,),name = 'input_1_cat')
    input_2_cat = tf.keras.layers.Input(shape=(1,),name = 'input_2_cat')
    input_3_cat = tf.keras.layers.Input(shape=(1,),name = 'input_3_cat')

    input_dim_1 = int(max(df.sku)) +1
    input_dim_2 = int(max(df.order)) +1
    input_dim_3 = int(max(df.category)) +1

    embedding_1 = tf.keras.layers.Embedding(input_dim=input_dim_1, output_dim=5, input_length=1,name = 'embedding_1')(input_1_cat)
    embedding_2 = tf.keras.layers.Embedding(input_dim=input_dim_2, output_dim=5, input_length=1,name = 'embedding_2')(input_2_cat)
    embedding_3 = tf.keras.layers.Embedding(input_dim=input_dim_3, output_dim=5, input_length=1,name = 'embedding_3')(input_3_cat)

    #Embedding shape is (None, 1, 3)
    #Need to flatten to make shape compatible with numeric input, which only has two dimensions: (None, 1)

    embedding_flat_1 = tf.keras.layers.Flatten(name='embedding_flat_1')(embedding_1)
    embedding_flat_2 = tf.keras.layers.Flatten(name='embedding_flat_2')(embedding_2)
    embedding_flat_3 = tf.keras.layers.Flatten(name='embedding_flat_3')(embedding_3)


    input_1_num = tf.keras.layers.Input(shape=(1,),name = 'input_1_num')
    input_2_num = tf.keras.layers.Input(shape=(1,),name = 'input_2_num')


    inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat_1,
                                                                         embedding_flat_2,
                                                                         embedding_flat_3,
                                                                         input_1_num,
                                                                         input_2_num])


    hidden1 = tf.keras.layers.Dense(units=num_neurons, activation=active_funcs, name='hidden1')(inputs_concat)
    hidden2 = tf.keras.layers.Dense(units=num_neurons, activation=active_funcs, name= 'hidden2')(hidden1)
    hidden3 = tf.keras.layers.Dense(units=num_neurons, activation=active_funcs, name= 'hidden3')(hidden2)


    outputs = tf.keras.layers.Dense(1, name = 'out')(hidden3)

    model = tf.keras.Model(inputs = [input_1_cat,input_2_cat,input_3_cat,input_1_num,input_2_num], outputs = outputs)

    # model.summary()

    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon))

    model1 = model.fit(x=[x_cat_1,x_cat_2,x_cat_3,x_num_4,x_num_5], y=y, batch_size=batch_size, epochs=5)
    
    losses = np.nan_to_num(model1.history['loss'], nan = 3000.0)
#     where_are_NaNs = isnan(model1.history['loss'])
#     model1.history['loss'][where_are_NaNs] = 3000.0
#     model1.history['loss'][np.isnan(model1.history['loss'])] = 3000.0
#     if model1.history['loss'][-1] is NaN:
#          model1.history['loss'][-1] == 10000
    
    return int(losses[-1])

def params(X):
    
    batch_size_list = [100,200,300,400,500]
    batch_size_dic = {}

    for i in np.arange(1,6):
        batch_size_dic[i] = batch_size_list[i-1]

    num_neurons_list = [2,5,10,15,20]
    num_neurons_dic = {}

    for i in np.arange(1,6):
        num_neurons_dic[i] = num_neurons_list[i-1]

    active_funcs_list = ["sigmoid","elu","relu","tanh","selu"]
    active_funcs_dic = {}

    for i in np.arange(1,6):
        active_funcs_dic[i] = active_funcs_list[i-1]


    learning_rate_list = [0.5,0.1,0.01,0.001,0.0001]
    learning_rate_dic = {}

    for i in np.arange(1,6):
        learning_rate_dic[i] = learning_rate_list[i-1]


    beta_1_list = [0.6,0.7,0.8,0.9,0.99]
    beta_1_dic = {}

    for i in np.arange(1,6):
        beta_1_dic[i] = beta_1_list[i-1]

    beta_2_list = [0.6,0.7,0.8,0.9,0.999]
    beta_2_dic = {}

    for i in np.arange(1,6):
        beta_2_dic[i] = beta_2_list[i-1]

    epsilon_list = [1e-02,1e-03,1e-04,1e-05,1e-06]
    epsilon_dic = {}

    for i in np.arange(1,6):
        epsilon_dic[i] = epsilon_list[i-1]
    
    final_loss = nn_model(batch_size_dic[X[0]],
                          num_neurons_dic[X[1]], 
                          active_funcs_dic[X[2]], 
                          learning_rate_dic[X[3]], 
                          beta_1_dic[X[4]], 
                          beta_2_dic[X[5]], 
                          epsilon_dic[X[6]])
    
    return final_loss
    
from geneticalgorithm import geneticalgorithm as ga

varbound=np.array([[1,5],[1,5],[1,5],[1,5],[1,5],[1,5],[1,5]])
vartype=np.array([['int'],['int'],['int'],['int'],['int'],['int'],['int']])
algorithm_param = {'max_num_iteration': 40,\
                   'population_size':10,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

ga_model=ga(function=params ,dimension=7,variable_type_mixed=vartype,variable_boundaries=varbound,algorithm_parameters=algorithm_param, function_timeout = 60.0)

ga_model.run()