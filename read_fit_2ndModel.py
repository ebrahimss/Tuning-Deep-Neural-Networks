import tensorflow as tf
tf.__version__

import numpy as np

import pandas as pd

import time

# read data
df = pd.read_csv("pricing_final.csv")

df.head()


# create a list of dfs based on sku :

df_vals = []
s_time = time.time()
for i in df.sku.unique():
    df_vals.append(df[df['sku'] == i])
    
f_time = time.time()
dur = f_time - s_time
dur

# create a dictionary of the data regarding each sku:

df_sku_dic = []
for i in df.sku.unique():
    new_data = {'sku': i,
                'price_mean': df_vals[i]['price'].mean(),
                'price_sum': df_vals[i]['price'].sum(),
                'quantity_mean': df_vals[i]['quantity'].mean(), 
                'quantity_sum': df_vals[i]['quantity'].sum(), 
                'duration_mean': df_vals[i]['duration'].mean(), 
                'duration_sum': df_vals[i]['duration'].sum(), 
                'category': df_vals[i].category.unique()[0], 
                'order_num': len(df_vals[i])}
#     df_sku_dic[i] = new_data
    df_sku_dic.append(new_data)

# convert to new df

df_new = pd.DataFrame(df_sku_dic)

df_new.head()

# x data

x_cat_1 = np.array(df_new.category) #categorical

x_num_1 = np.array(df_new.price_mean) #numeric
x_num_2 = np.array(df_new.price_sum) #numeric
x_num_3 = np.array(df_new.duration_mean) #numeric
x_num_4 = np.array(df_new.duration_sum) #numeric
x_num_5 = np.array(df_new.order_num) #numeric


#y data
y = np.array(df_new.quantity_mean)

len(y)

# input_1_cat = tf.keras.layers.Input(shape=(1,),name = 'input_1_cat')
input_1_cat = tf.keras.layers.Input(shape=(1,),name = 'input_1_cat')

input_dim_1 = int(max(df_new.category))+1

embedding_1 = tf.keras.layers.Embedding(input_dim=input_dim_1, output_dim=3, input_length=1,name = 'embedding_1')(input_1_cat)

#Embedding shape is (None, 1, 3)
#Need to flatten to make shape compatible with numeric input, which only has two dimensions: (None, 1)

embedding_flat_1 = tf.keras.layers.Flatten(name='embedding_flat_1')(embedding_1)

input_1_num = tf.keras.layers.Input(shape=(1,),name = 'input_1_num')
input_2_num = tf.keras.layers.Input(shape=(1,),name = 'input_2_num')
input_3_num = tf.keras.layers.Input(shape=(1,),name = 'input_3_num')
input_4_num = tf.keras.layers.Input(shape=(1,),name = 'input_4_num')
input_5_num = tf.keras.layers.Input(shape=(1,),name = 'input_5_num')


inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([
                                                                     embedding_flat_1,
                                                                     input_1_num,
                                                                     input_2_num, input_3_num,input_4_num,input_5_num])


hidden1 = tf.keras.layers.Dense(units=3, activation="sigmoid", name='hidden1')(inputs_concat)
hidden2 = tf.keras.layers.Dense(units=3, activation="sigmoid", name= 'hidden2')(hidden1)


outputs = tf.keras.layers.Dense(1, name = 'out')(hidden2)

model = tf.keras.Model(inputs = [input_1_cat,input_1_num,input_2_num,input_3_num,input_4_num,input_5_num], outputs = outputs)

model.summary()

#Gloally Decayed L.R
initial_lr = 0.1
decay_steps = 10000
decay_rate = 0.1

learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_lr, decay_steps, decay_rate)








model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = learning_schedule, momentum=0.9, nesterov=True))

model.fit(x=[x_cat_1,x_num_1,x_num_2,x_num_3,x_num_4,x_num_5],y=y, batch_size=100, epochs=10)

