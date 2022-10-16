#implemting multiple non-linear regression in CO2 Emission_Canada dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np


df = pd.read_csv('CO2 Emissions_Canada.csv')
df.info()
print(df.head(5))
print(df.describe())


# define Dependent and Independent Features

X2 = df[['Fuel_Consumption_Comb']]
X3 = df[['Fuel_Consumption_Comb_mpg']]
y = df[['CO2_Emissions']]

poly_features = PolynomialFeatures(degree = 2)
X = poly_features.fit_transform(X2,X3)

# divide train and test set

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state = 42)

# define model

regr = LinearRegression()

# fitting Train set

regr.fit(X_train,y_train)

print("Coefficients : ",regr.coef_)
print("Intercept : ",regr.intercept_)
print("Singular : ",regr.singular_)
print("Rank : ",regr.rank_)

# predicting on test set (validation)

test_y_cap = regr.predict(X_test)
#np.mean(np.absolute(test_y_cap - y_test))

msq = np.mean(np.power((test_y_cap - y_test),2),axis=0)

print("Mean Square Error is {0}".format(msq))

R2_score = r2_score(test_y_cap, y_test)

print("R2 Score is {0}".format(R2_score))






