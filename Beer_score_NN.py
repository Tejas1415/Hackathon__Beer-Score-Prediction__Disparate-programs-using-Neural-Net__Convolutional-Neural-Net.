# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:43:22 2018

@author: Tejas_Krishna_Reddy

Beer Score Prediction using deep_neural net
Two methods: 1. General model fit method
             2. KerasRegressor
Plotting and analysing acc, val_acc, loss, val_loss

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import cross_validation, svm, preprocessing

df = pd.read_csv('Beer Train Data Set.csv')
df.drop(['Food Paring', 'Glassware Used'], 1, inplace= True)
df.drop(['Beer Name'], 1, inplace=True)

df['Ratings'] = pd.to_numeric(df.Ratings)
df = df[(df.Ratings != 0)]
df.drop(['Ratings'], 1, inplace=True)

# Now in the ABV column, fill the missing values with the mean of all the values.
df['ABV'].fillna(df['ABV'].mean(), inplace=True)

Label = preprocessing.LabelEncoder()
Label.fit(df['Style Name'])
# Now the label is formed, See the elements in the label and thentransform to integers
list(Label.classes_)
df['Style Name']=Label.transform(df['Style Name'])
df = df.dropna(axis=0)

y= np.array(df['Brewing Company'], dtype = int)
ans=list()
for i in y:
    a = np.binary_repr(i,14)
    ans.append(a)
#print(ans)
df1 = pd.DataFrame(ans)
df1['ans'] = pd.DataFrame(ans)

for i in range(14):
    df['B'+ str(i)] = df1['ans'].str[i]

df.drop(['Brewing Company'], 1, inplace=True)

y1 = np.array(df['Style Name'], dtype = int)
ans1=list()
for i1 in y1:
    a1 = np.binary_repr(i1,7)
    ans1.append(a1)
df1['ans1'] = pd.DataFrame(ans1)

for i1 in range(7):
    df['BS1' + str(i1)] = df1['ans1'].str[i1]

# remove original style name now
df.drop(['Style Name'], 1, inplace = True)
df = df.dropna(axis=0)

c = df.columns[df.dtypes.eq(object)]
df[c] = df[c].apply(pd.to_numeric, errors='coerce', axis=0)

scaler = QuantileTransformer()
#df3 = scaler.fit_transform(df)

X5 = np.array(df.drop(['Score'], 1))
y5 = np.array(df['Score'])

X3 = scaler.fit_transform(pd.DataFrame(X5))
#y3 = scaler.fit_transform(pd.DataFrame(y5))
y3 = 1.2- np.log(y5)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X5, y5, test_size=0.20)

####################### Data Munging Compleated.######################

from keras import layers, models 
from keras import regularizers, optimizers
from keras.wrappers.scikit_learn import KerasRegressor

#39.52 - 80,46,17 # input - no of features, output =1 regression, middle=mean of ip,op
#39.61 - without any droputs/regulirizers/optimizers
#losses- poisson, logcosh, mean_absolute_percentage_error,
#39.75 - with batch size =512 and optimised usings graphs
 
def build_model():
    model= models.Sequential()
    model.add(layers.Dense(16, activation = 'sigmoid', input_shape = (X_train.shape[1],)))
    #model.add(layers.Dropout(0.1))
    model.add(layers.Dense(12, activation = 'sigmoid'))
    #model.add(layers.Dense(3, activation = 'sigmoid'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(2, activation = 'sigmoid'))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    #sgd = optimizers.SGD(lr=0.0001, nesterov=True
    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics=['acc'])
    return model

# train the final model - Without SGD, regularizers its 0.394   
model = build_model()
history = model.fit(X_train, y_train, epochs=12, batch_size=512, validation_data= (X_test, y_test))

#=model.fit_generator()
test_mse, test_mae = model.evaluate(X_test, y_test)
print(test_mae)
prediction1 = model.predict(X_test)

score = mean_squared_error(prediction1, y_test)
print(score)
RMSE = np.sqrt(score)                # Calculated score based on the evaluation scheme of Hackathon
sig = 1/ (1+math.exp(-RMSE))
Final_Score = 1 - sig
print(Final_Score)

#### Using Keras Regressor(0.396 - nb=10, epoch =5, batch_size=20), 

clf2 = KerasRegressor(build_fn= build_model, nb_epoch=1500, epochs= 20, batch_size =32)
clf2.fit(X_train, y_train)

prediction2 = clf2.predict(X_test)
prediction2= np.exp(1.2-prediction2)
score = mean_squared_error(prediction2, y_test)
print(score)
RMSE = np.sqrt(score)                # Calculated score based on the evaluation scheme of Hackathon
sig = 1/ (1+math.exp(-RMSE))
Final_Score = 1 - sig
print(Final_Score)

# Plotting
#history.history
# dont consider the first 3 epochs. 

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss= history.history['val_loss']

#Plotting results for loss
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'g', label = 'validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# plotting for accuracy
plt.clf
plt.plot(epochs, acc, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'r', label = 'validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show 

# Number of epochs + determining if the model is overfitting - through the graphs
# Also since, training and validation loss/acc almost overlap, no need to optimise it through regularizers, optimizers, dropouts etc. 
