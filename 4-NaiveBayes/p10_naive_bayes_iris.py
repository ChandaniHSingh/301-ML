#implemting Naive Bayes Classifier in iris dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix


# load dataset and classify feature and target

df = pd.read_csv('iris.csv')

X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df[['species']]

# dataset split into train and test

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

# define model

model = GaussianNB()

# Train
y_train = np.array(y_train).reshape(-1)
model.fit(X_train,y_train)

# test
y_pred = model.predict(X_test)

#print("%d : %d",(X_test.shape[0],(y_test != y_pred).sum()))

print(accuracy_score(y_test,y_pred,normalize=True,sample_weight = None))
print(confusion_matrix(y_test,y_pred))


