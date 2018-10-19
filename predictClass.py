
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from operator import add
from sklearn import tree
from sklearn import preprocessing


# In[24]:


probeA = pd.read_csv("../probeA.csv")
probeB = pd.read_csv("../probeB.csv")
#dataframe for probe A and B
probeA = probeA.loc[:, probeA.columns != 'tna'] #tna values irrelevant
probeB = probeB.loc[:, probeB.columns != 'tna']


# In[26]:


def changeTheData(dom):
    df = dom
    count = 0
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        for i, row in df.iterrows():
            dataPoint = df.loc[i,col]
            if (abs(dataPoint-mean)/std > 3): #if entry doesn't belong in this column
                for j in df.columns: #iterate through this row
                    if (abs(dataPoint-df[j].mean())/df[j].std() < 3): #if entry belongs in new column
                        df.set_value(i, col, df.loc[i,j]) #swap them
                        df.set_value(i, j, dataPoint)
                        
    return df

#swap the entries in the wrong columns
probeA[['c1','c2','c3']] = changeTheData(probeA[['c1','c2','c3']])
probeA[['n1','n2','n3']] = changeTheData(probeA[['n1','n2','n3']])
probeA[['m1','m2','m3']] = changeTheData(probeA[['m1','m2','m3']])
probeA[['p1','p2','p3']] = changeTheData(probeA[['p1','p2','p3']])

probeB[['c1','c2','c3']] = changeTheData(probeB[['c1','c2','c3']])
probeB[['n1','n2','n3']] = changeTheData(probeB[['n1','n2','n3']])
probeB[['m1','m2','m3']] = changeTheData(probeB[['m1','m2','m3']])
probeB[['p1','p2','p3']] = changeTheData(probeB[['p1','p2','p3']])


# In[39]:


X = probeA.loc[:, probeA.columns!='class'] #domain
weights = [0.05181453,  0.06691289,  0.12364071,  0.10153385 , 0.06235098 , 0.09507808,
  0.04968982 , 0.08436919  ,0.16247953 , 0.1170296 ,  0.0419041 ,  0.04319672] #array of weights
X = X*weights
y = probeA.loc[:, probeA.columns == 'class'] #targets
model = KNeighborsClassifier(n_neighbors = 71, weights= 'distance', p = 2) #knn model
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X) #scale probeA data
probeB = scaler.fit_transform(probeB) #scale probeB data
model.fit(X, y)
predictions = model.predict_proba(probeB)[:, 1]
np.savetxt('classB.csv', predictions)

