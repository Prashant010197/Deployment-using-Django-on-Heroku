
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('Salary_Data.csv')

X = df.iloc[:, 0].values
Y = df.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = tts(np.array(X).reshape(-1, 1), Y, random_state=42, test_size=0.2)

classifier = LinearRegression()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

joblib.dump(classifier, 'linreg.pkl')


