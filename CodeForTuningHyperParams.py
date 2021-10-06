import pandas as pd  
import numpy as np 
import tensorflow as tf 
import mysql.connector
import itertools




#################### load in data 
df=pd.read_csv('pricing_final.csv')
df.columns
df.info()
X=df.drop('quantity',axis=1) 
y=df['quantity']
##### tune through different optimizers 
#1.number nodes 
#2.number of hidden layers 
#3.act functions 
#4.eps 
#5.encoding number 
#6.optimizers 
#7.batch size  
#8.mumber of epochs 


def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}
#need to figure out how to tune the number of layers 
numberNodes1=np.arange(1,3,1)
numberNodes2=np.arange(1,3,1)
numberNodes3=np.arange(1,3,1)
actFuns1=['elu','sigmoid'] 
actFuns2=['elu','sigmoid'] 
actFuns3=['elu','sigmoid'] 
lr=np.arange(0.0001,0.0002,0.0001)
lr
params=pd.DataFrame(expandgrid(numberNodes1,numberNodes2,numberNodes3,actFuns1,actFuns2,actFuns3,lr))
params
#make a col in params to store the loss 
params['loss']=0.0 

#params['loss'].iloc[0]=model.evaluate(valX,valY)
params.columns
folds=np.arange(0,1.25,.25)

for i in range(0,len(params)):
    lossParam_i=[]
    for j in range(0,len(folds)-1):
        #this line is doing this.... 
        buildX=X.iloc[~X.index.isin(range(int(round(len(X.index)*folds[j])),int(round(len(X.index)*folds[j+1]))))] 
        #this line is doing this.... 
        buildY=y.iloc[~y.index.isin(range(int(round(len(X.index)*folds[j])),int(round(len(X.index)*folds[j+1]))))] 

        valX=X.iloc[X.index.isin(range(int(round(len(X.index)*folds[j])),int(round(len(X.index)*folds[j+1]))))]  
        valY=y.iloc[X.index.isin(range(int(round(len(y.index)*folds[j])),int(round(len(X.index)*folds[j+1]))))]  

        inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not include records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
        hidden1 = tf.keras.layers.Dense(units=params['Var1'].iloc[i], activation=params['Var4'].iloc[i], name = 'hidden1')(inputs)
        hidden2 = tf.keras.layers.Dense(units=params['Var2'].iloc[i], activation=params['Var5'].iloc[i], name= 'hidden2')(hidden1)
        hidden3 = tf.keras.layers.Dense(units=params['Var3'].iloc[i], activation=params['Var6'].iloc[i], name= 'hidden3')(hidden2)
        output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

        model = tf.keras.Model(inputs = inputs, outputs = output)

        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = params['Var7'].iloc[i]))

        #Fit model
        model.fit(x=buildX,y=buildY, batch_size=200, epochs=10)

        lossParam_i.append(model.evaluate(valX,valY))
    params['loss'].iloc[i]=np.mean(lossParam_i)

####### make sure to keep all of the results from training and save as a df in the folder 
params.to_csv('paramsForModel.csv')
params.head()


################# do same but make a deeper nn, will take longer as there will be more hyperparameters to tune through 
#get the best parameters 
params[params['loss']==params['loss'].min()]