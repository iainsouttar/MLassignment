
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import preprocessing


# In[56]:


probeA = pd.read_csv("../probeA.csv")
probeB = pd.read_csv("../probeB.csv")
#dataframe for probe A and B
probeA = probeA.loc[:, probeA.columns != 'class'] #tna values irrelevant
probeB = probeB.loc[:, probeB.columns != 'class']


# In[25]:


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


# In[53]:


X = probeA.loc[:, probeA.columns!='tna'] #domain
y = probeA.loc[:, probeA.columns =='tna'] #target
poly = preprocessing.PolynomialFeatures(degree = 2) #feature expansion
lasso = linear_model.Lasso(alpha = 0.001)
scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X) #scale domain
probeBscaled = scaler.fit_transform(probeB) #scale probeB data
Xexpanded = poly.fit_transform(Xscaled) #feature expansion on data
probeBpoly = poly.fit_transform(probeBscaled) 
lasso.fit(Xexpanded, y) #fit model
predictions = lasso.predict(probeBpoly)
np.savetxt('tnaB.csv', predictions)

