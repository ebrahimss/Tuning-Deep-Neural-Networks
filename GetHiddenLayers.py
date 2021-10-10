from numpy.lib.function_base import append
import pandas as pd  
import numpy as np  
import tensorflow as tf  
import itertools
import time

df=pd.read_csv('pricing_final.csv')
df.columns
df.info()
X=df.drop('quantity',axis=1) 
y=df['quantity']
#i think order is a quantative var 
len(np.unique(df['sku'])) #74999
len(np.unique(df['category'])) #32 

df.head()

def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

def to_flatten3(my_list, primitives=(int)):
    flatten = []
    for item in my_list:
        if isinstance(item, primitives):
            flatten.append(item)
        else:
            flatten.extend(item)
    return flatten


def getNodeCombinations(numberOfLayers):
    numNodes=[]
    #numberOfLayers=4
    for j in range(numberOfLayers):
        numNodes.append(np.arange(1,3,1))

    paramDF=pd.DataFrame(expandgrid(numNodes[0]))
    for i in range(numberOfLayers-1): #set this to the number of layers minus 1 
        paramDF=pd.DataFrame(expandgrid(paramDF.values.tolist(),numNodes[i]))
    
    for i in range(0,len(paramDF)):
        x=paramDF.loc[i,'Var1'] 
        while(len(x)<numberOfLayers-1):
            x=to_flatten3(x)
        #then do one more time 
        paramDF['Var1'].loc[i]  = to_flatten3(x)
        paramDF['Var1'].loc[i].append(paramDF['Var2'].loc[i])
    paramDF.drop('Var2',inplace=True,axis=1)
    return paramDF




numlayers=3
nodeSizes=getNodeCombinations(numlayers) 
inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') 


for k in range(len(nodeSizes)): 
    layers=[]
    layerNames=[]
#this loop actualy builds each layer defined in the m outer loop 
    for i in range(numlayers):
        if(i==0):
            layerNames.append('hidden'+str(i+1))
            layers.append(tf.keras.layers.Dense(units=nodeSizes.iloc[k][0][i], activation='sigmoid', name = 'hidden1')(inputs))
        else:
            layerNames.append('hidden'+str(i+1))
            layers.append(tf.keras.layers.Dense(units=nodeSizes.iloc[k][0][i], activation='sigmoid', name = layerNames[i])(layers[i-1]))
#now compile model with last hidden layer fed into output 
    output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(layers[len(layers)-1])
    model = tf.keras.Model(inputs = inputs, outputs = output)

    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01))
        
    model.fit(x=X,y=y, batch_size=200, epochs=1)
    model.summary()
