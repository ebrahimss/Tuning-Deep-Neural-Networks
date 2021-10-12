import pandas as pd  
import numpy as np 
import tensorflow as tf 
import itertools
from tqdm import tqdm


################## output should have a linear act function 

#################### load in data 
df1=pd.read_csv('pricing_final.csv')
df1.columns
df1.info()



######################################################################## convert all to floats 





test=df1.iloc[df1.index[df1.index.isin(np.random.choice(df1.index,size=int(round(len(df1)*.2))))]]
df=df1.iloc[df1.index[~df1.index.isin(np.random.choice(df1.index,size=int(round(len(df1)*.2))))]]
df.head()
df.index=range(0,len(df))

X=df.drop('quantity',axis=1) 
y_data=df['quantity']
#i think order is a quantative var 
len(np.unique(df['sku'])) #74999
len(np.unique(df['category'])) #32

len(X) 
len(y_data)
##### tune through different optimizers 
#1.number nodes 
#2.number of hidden layers 
#3.act functions 
#4.eps 
#5.encoding number 
#6.optimizers 
#7.batch size  
#8.number of epochs 
#9.the number of vars in the encodings 


def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}
#need to figure out how to tune the number of layers 
numberNodes1=np.arange(3,5,1)
numberNodes2=np.arange(2,4,1)
numberNodes3=np.arange(1,3,1)

actFuns1=['elu','sigmoid','relu'] 
actFuns2=['elu','sigmoid','relu'] 
actFuns3=['elu','sigmoid','relu']  
lr=np.arange(0.0001,0.0003,0.0001)

#####parameters for optimizers if use adam then just comment these out 
#change different optimizers 
momment=[0,0.9] # take these out and use adam  
nv=[True,False]

params=pd.DataFrame(expandgrid(numberNodes1,numberNodes2,numberNodes3,actFuns1,actFuns2,actFuns3,lr,momment,nv))
len(params)
#make a col in params to store the loss 
params['loss']=0.0 

#params['loss'].iloc[0]=model.evaluate(valX,valY)
params.head()
folds=np.arange(0,1,.75) #this is done by 1 fold cv 
folds



for i in tqdm(range(len(params))):
    lossParam_i=[]
    for j in range(0,len(folds)-1):
        #this line is doing this.... 
        buildX=X.iloc[X.index.isin(range(int(round(len(X.index)*folds[j])),int(round(len(X.index)*folds[j+1]))))] 
        #this line is doing this.... 
        buildY=y_data.iloc[y_data.index.isin(range(int(round(len(X.index)*folds[j])),int(round(len(X.index)*folds[j+1]))))] 
        len(buildX)
        valX=X.iloc[~X.index.isin(range(int(round(len(X.index)*folds[j])),int(round(len(X.index)*folds[j+1]))))]  
        valY=y_data.iloc[~y_data.index.isin(range(int(round(len(X.index)*folds[j])),int(round(len(X.index)*folds[j+1]))))]  
        len(valX)

        #put encoded layers here 
        #add away to create layers, make a for loop 

      

        # x data
        x_cat_1 = np.array(buildX.sku) #categorical
        x_cat_2 = np.array(buildX.order) #categorical
        x_cat_3 = np.array(buildX.category) #categorical


        x_num_4 = np.array(buildX.price) #numeric
        x_num_5 = np.array(buildX.duration) #numeric

        #y data
        y = np.array(buildY)
        


        ########### val data 
        # x data
        valx_cat_1 = np.array(valX.sku) #categorical
        valx_cat_2 = np.array(valX.order) #categorical
        valx_cat_3 = np.array(valX.category) #categorical


        valx_num_4 = np.array(valX.price) #numeric
        valx_num_5 = np.array(valX.duration) #numeric

        #y data
        valy = np.array(valY)

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



        #inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not include records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
        hidden1 = tf.keras.layers.Dense(units=params['Var1'].iloc[i], activation=params['Var4'].iloc[i], name = 'hidden1')(inputs_concat)
        hidden2 = tf.keras.layers.Dense(units=params['Var2'].iloc[i], activation=params['Var5'].iloc[i], name= 'hidden2')(hidden1)
        hidden3 = tf.keras.layers.Dense(units=params['Var3'].iloc[i], activation=params['Var6'].iloc[i], name= 'hidden3')(hidden2)
        output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

        model = tf.keras.Model(inputs = [input_1_cat,input_2_cat,input_3_cat,input_1_num,input_2_num], outputs = output)

      
        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = params['Var7'].iloc[i], momentum=
            np.array(params['Var8'].iloc[i]), nesterov=bool(params['Var9'].iloc[i])))
        
        # model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=params['Var7'].iloc[i], beta_1=0.9, beta_2=0.999, epsilon=1e-07))

        #Fit model
        model.fit(x=[x_cat_1,x_cat_2,x_cat_3,x_num_4,x_num_5],y=y, batch_size=200, epochs=10)

        lossParam_i.append(model.evaluate(x=[valx_cat_1,valx_cat_2,valx_cat_3,valx_num_4,valx_num_5],y=valy))
    params['loss'].iloc[i]=np.mean(lossParam_i)

####### make sure to keep all of the results from training and save as a df in the folder 
params.to_csv('paramsForModel3_MoreActFunslayers.csv')
params.head()
params['loss'].min()


################# do same but make a deeper nn, will take longer as there will be more hyperparameters to tune through 
#get the best parameters 
params[params['loss']==params['loss'].min()]

### keep track of the loss for each number of hidden layers 


########################### load in the other files 

h=pd.read_csv('paramsForModelHenry.csv')
j=pd.read_csv('paramsForModel3Justus.csv')
h['loss'].min() 
j['loss'].min()
h[h['loss']==h['loss'].min()] 



## lowest loss for each parameter combination data  
params[params['loss']==params['loss'].min()] 
h[h['loss']==h['loss'].min()] 
j[j['loss']==j['loss'].min()] 


###################################### params nn 
bestParams=params[params['loss']==params['loss'].min()] 
## fit nn 
# x data
x_cat_1 = np.array(df.sku) #categorical
x_cat_2 = np.array(df.order) #categorical
x_cat_3 = np.array(df.category) #categorical


input_1_cat = tf.keras.layers.Input(shape=(1,),name = 'input_1_cat')
input_2_cat = tf.keras.layers.Input(shape=(1,),name = 'input_2_cat')
input_3_cat = tf.keras.layers.Input(shape=(1,),name = 'input_3_cat')


input_dim_1 = int(max(df.sku)) +1
input_dim_2 = int(max(df.order)) +1
input_dim_3 = int(max(df.category)) +1




x_num_4 = np.array(df.price) #numeric
x_num_5 = np.array(df.duration) #numeric

#y data

y = np.array(df.quantity)

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


bestParams
i=0
#inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not include records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
hidden1 = tf.keras.layers.Dense(units=bestParams['Var1'].iloc[i], activation=bestParams['Var4'].iloc[i], name = 'hidden1')(inputs_concat)
hidden2 = tf.keras.layers.Dense(units=bestParams['Var2'].iloc[i], activation=bestParams['Var5'].iloc[i], name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=bestParams['Var3'].iloc[i], activation=bestParams['Var6'].iloc[i], name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

model = tf.keras.Model(inputs = [input_1_cat,input_2_cat,input_3_cat,input_1_num,input_2_num], outputs = output)

      
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = bestParams['Var7'].iloc[i], momentum=
    np.array(bestParams['Var8'].iloc[i]), nesterov=bool(bestParams['Var9'].iloc[i])))

#Fit model
model.fit(x=[x_cat_1,x_cat_2,x_cat_3,x_num_4,x_num_5],y=y, batch_size=200, epochs=10)
#get vip and pdp on df 




############################# fit model for h 
bestParams=h[h['loss']==h['loss'].min()] 
## fit nn 
# x data
x_cat_1 = np.array(df.sku) #categorical
x_cat_2 = np.array(df.order) #categorical
x_cat_3 = np.array(df.category) #categorical


input_1_cat = tf.keras.layers.Input(shape=(1,),name = 'input_1_cat')
input_2_cat = tf.keras.layers.Input(shape=(1,),name = 'input_2_cat')
input_3_cat = tf.keras.layers.Input(shape=(1,),name = 'input_3_cat')


input_dim_1 = int(max(df.sku)) +1
input_dim_2 = int(max(df.order)) +1
input_dim_3 = int(max(df.category)) +1




x_num_4 = np.array(df.price) #numeric
x_num_5 = np.array(df.duration) #numeric

#y data

y = np.array(df.quantity)

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


bestParams
i=0
#inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not include records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
hidden1 = tf.keras.layers.Dense(units=bestParams['Var1'].iloc[i], activation=bestParams['Var4'].iloc[i], name = 'hidden1')(inputs_concat)
hidden2 = tf.keras.layers.Dense(units=bestParams['Var2'].iloc[i], activation=bestParams['Var5'].iloc[i], name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=bestParams['Var3'].iloc[i], activation=bestParams['Var6'].iloc[i], name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

model = tf.keras.Model(inputs = [input_1_cat,input_2_cat,input_3_cat,input_1_num,input_2_num], outputs = output)

      
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = bestParams['Var7'].iloc[i], momentum=
    np.array(bestParams['Var8'].iloc[i]), nesterov=bool(bestParams['Var9'].iloc[i])))

#Fit model
model.fit(x=[x_cat_1,x_cat_2,x_cat_3,x_num_4,x_num_5],y=y, batch_size=200, epochs=10)

################### fit a lm model predicting the loss from the parameters, combine j and h 
#222 at 10:12 
