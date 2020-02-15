# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 07:05:22 2020

@author: sagar
"""

import numpy as np
import matplotlib as mat
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[: ,: -1].values
Y = dataset.iloc[:, 4].values
# normalized the value
from sklearn.preprocessing import LabelEncoder , OneHotEncoder 
labelencoder_X= LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
#remove dummy variable
X=X[:, 1:]
# splittinf the data into test and training
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y ,test_size =0.2 , random_state =0)
# Creating the model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
# predict the values
Y_predict = regressor.predict(X_test)
# backword elimination
# adding b0 =1 as a matrix
import statsmodels.api as sm
X = np.append(arr = np.ones([50,1]).astype(int), values =X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
#eleminate the x2 gretest p-value
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
#eleminate the x2 gretest p-value
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
#eleminate the x2 gretest p-value
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
#eleminate the x2 gretest p-value
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
#
#
#
# Creating the model 
from sklearn.linear_model import LinearRegression
regressor_M = LinearRegression()
regressor_M.fit(X_opt, Y_train)
# predict the values
Y_predict_M = regressor_M.predict(X_test)