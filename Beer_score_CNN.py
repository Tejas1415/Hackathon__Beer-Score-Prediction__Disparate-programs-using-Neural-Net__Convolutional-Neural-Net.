# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 01:42:14 2018

@author: Tejas_2

Beer Score CNN try
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

# Data Munging Compleated.######################

from keras import layers, models

def build_model():
    
    model = models.Sequential()
# 32 - Next layer depth, no of layers that should be present in the next layer, each layer acts as unique filter.
# (28, 28, 1) - height width and number of channels in the input image
    model.add(layers.Conv1D(filters = 32, kernel_size = (5), strides=5,
                            padding = 'valid', activation = 'relu',
                            input_shape = (24, X_train.shape[1])))
# max pooling reduces the dimenion of the imput height and width each time almost to half or half+1
    model.add(layers.MaxPooling1D(pool_size=24))
    model.add(layers.Conv1D(filters = 64, kernel_size = 3,strides=1, padding = 'valid', activation = 'relu'))
    model.add(layers.MaxPooling1D(pool_size=24))
    model.add(layers.Conv1D(filters = 64, kernel_size = 3, strides=1, padding = 'valid', activation = 'relu'))
# Now flatten the layers, and then create dense layers for the output
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'sigmoid'))
    model.add(layers.Dense(10, activation = 'sigmoid'))
    model.add(layers.Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics=['acc'])
    return model

# model.summary() - to check the number of parameters
X_train = X_train.reshape(109724, 24, 1)
y_train = y_train.reshape(109724, 1, 1)

X_train = X_train.reshape(109724, 1, 24)
y_train = y_train.reshape(109724, 1, 1)

model = build_model()
history = model.fit(X_train, y_train, epochs=5, batch_size=512)

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











