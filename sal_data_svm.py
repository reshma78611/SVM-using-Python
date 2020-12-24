# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 00:12:01 2020

@author: HP
"""
import numpy as np
import pandas as pd
import seaborn as sns

sal_train=pd.read_csv('C:/Users/HP/Desktop/assignments submission/SVM/SalaryData_Train(1).csv')
sal_test=pd.read_csv('C:/Users/HP/Desktop/assignments submission/SVM/SalaryData_Test(1).csv')
sal_train.columns
sal_test.columns
string_col=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
for i in string_col:
    sal_train[i]=label_encoder.fit_transform(sal_train[i])
    sal_test[i]=label_encoder.fit_transform(sal_test[i])
    
#columns=list(sal_train.columns)
#train_x=sal_train[columns[0:13]]
#train_y=sal_train[columns[13]]
#test_x=sal_test[columns[0:13]]
#test_y=sal_test[columns[13]]

train_x=sal_train.iloc[0:500,0:13]
train_y=sal_train.iloc[0:500,13]
test_x=sal_test.iloc[0:300,0:13]
test_y=sal_test.iloc[0:300,13]


#SVM Classification using kernels: linear,poly,rbf
from sklearn.svm import SVC

#kernel=linear
model_linear=SVC(kernel='linear')
model_linear.fit(train_x,train_y)
train_pred_lin=model_linear.predict(train_x)
test_pred_lin=model_linear.predict(test_x)
train_lin_acc=np.mean(train_pred_lin==train_y)
test_lin_acc=np.mean(test_pred_lin==test_y)
train_lin_acc#81.8
test_lin_acc#81.6

#kernel=poly
model_poly=SVC(kernel='poly')
model_poly.fit(train_x,train_y)
train_pred_poly=model_poly.predict(train_x)
test_pred_poly=model_poly.predict(test_x)
train_poly_acc=np.mean(train_pred_poly==train_y)
test_poly_acc=np.mean(test_pred_poly==test_y)
train_poly_acc#81.2
test_poly_acc#80.33

#kernel=rbf
model_rbf=SVC(kernel='rbf')
model_rbf.fit(train_x,train_y)
train_pred_rbf=model_rbf.predict(train_x)
test_pred_rbf=model_rbf.predict(test_x)
train_rbf_acc=np.mean(train_pred_rbf==train_y)
test_rbf_acc=np.mean(test_pred_rbf==test_y)
train_rbf_acc#81.2
test_rbf_acc#80.33
