import tensorflow as tf
tf.__version__

import numpy as np

import pandas as pd

df = pd.read_csv("pricing_final.csv")

df.head()

# x data
x_cat_1 = np.array(df.sku) #categorical
x_cat_2 = np.array(df.order) #categorical
x_cat_3 = np.array(df.category) #categorical


x_num_4 = np.array(df.price) #numeric
x_num_5 = np.array(df.duration) #numeric

#y data
y = np.array(df.quantity)


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


hidden1 = tf.keras.layers.Dense(units=2, activation="sigmoid", name='hidden1')(inputs_concat)
hidden2 = tf.keras.layers.Dense(units=2, activation="sigmoid", name= 'hidden2')(hidden1)


outputs = tf.keras.layers.Dense(1, activation = "softmax", name = 'out')(hidden2)

model = tf.keras.Model(inputs = [input_1_cat,input_2_cat,input_3_cat,input_1_num,input_2_num], outputs = outputs)

model.summary()

model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))

model.fit(x=[x_cat_1,x_cat_2,x_cat_3,x_num_4,x_num_5],y=y, batch_size=1, epochs=10)