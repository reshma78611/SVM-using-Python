# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 22:51:57 2020

@author: HP
"""
import numpy as np
import pandas as pd
import seaborn as sns

letters=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/SVM/letters.csv')
letters.head()
letters.describe()
letters.columns

sns.boxplot(x='lettr',y='x-box',data=letters,palette='hls')
sns.boxplot(x='y-box',y='lettr',data=letters,palette='hls')
#sns.pairplot(data=letters)

from sklearn.model_selection import train_test_split
train,test=train_test_split(letters,test_size=0.3,random_state=0)
columns=list(letters.columns)
predictors=columns[1:17]
target=columns[0]

#SVM Classification using Kernels: linear,poly,rbf

from sklearn.svm import SVC

#kernel=linear
model_linear=SVC(kernel='linear')
model_linear.fit(train[predictors],train[target])
train_pred_linear=model_linear.predict(train[predictors])
test_pred_linear=model_linear.predict(test[predictors])
train_lin_acc=np.mean(train_pred_linear==train[target])
test_lin_acc=np.mean(test_pred_linear==test[target])
train_lin_acc#0.87
test_lin_acc#0.86


#kernel=poly
model_poly=SVC(kernel='poly')
model_poly.fit(train[predictors],train[target])
train_pred_poly=model_poly.predict(train[predictors])
test_pred_poly=model_poly.predict(test[predictors])
train_poly_acc=np.mean(train_pred_poly==train[target])
test_poly_acc=np.mean(test_pred_poly==test[target])
train_poly_acc#0.97
test_poly_acc#0.94

#kernel=rbf
model_rbf=SVC(kernel='rbf')
model_rbf.fit(train[predictors],train[target])
train_pred_rbf=model_rbf.predict(train[predictors])
test_pred_rbf=model_rbf.predict(test[predictors])
train_rbf_acc=np.mean(train_pred_rbf==train[target])
test_rbf_acc=np.mean(test_pred_rbf==test[target])
train_rbf_acc#0.933
test_rbf_acc#0.92





