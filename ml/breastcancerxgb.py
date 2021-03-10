# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 16:21:22 2021

@author: hp
"""

import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('haberman.csv')

#Renaming the column names for ease
df.columns = ['age_of_op', 'year_of_op', 'axillary_nodes','Survival']

#Replacing class label 2 with 0 as 0 better denotes Fatal surgery
a = {2: 0}
df.replace(a, inplace=True)

df.drop('year_of_op', inplace=True, axis=1)

X=df.iloc[:, 0:2].values
Y=df.iloc[:, 2].values

#Splitting the data
X_train, X_test, Y_train, Y_test=tts(X, Y, test_size=0.2, random_state=42)

#Using XGBoost algorithm to improve upon the existing baseline score
param_grid={"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}
grid = GridSearchCV(xgb.XGBClassifier(), param_grid, refit = True, scoring='recall') 
grid.fit(X_train, Y_train)
print(grid.best_estimator_)
print(grid.score(X_test, Y_test))

#93% accuracy achieved
classifier = xgb.XGBClassifier(colsample_bytree=0.3, gamma=0.0, learning_rate=0.05, max_depth=4,min_child_weight=7)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

#Recall Score used to measure efficacy as it is more important to minimise false negatives
recall_score(Y_test, y_pred)

#############################################################################


#Recall Score used to measure efficacy as it is more important to minimise false negatives
forest=RandomForestClassifier(n_estimators=100)
forest.fit(X_train, Y_train)
y_pred=forest.predict(X_test)

recall_score(Y_test,y_pred)

joblib.dump(forest, 'b3.pkl')













