# P14. implemting ANN on iris dataset

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# load dataset and classify feature and target

dataset = datasets.load_iris()

X = dataset.data
y = dataset.target

# dataset split into train and test

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

# define model

clf = MLPClassifier(solver='sgd', alpha=0.00001, hidden_layer_sizes=(5,), max_iter=200)

# Train
#y_train = np.array(y_train).reshape(-1)
clf.fit(X_train,y_train)

# test
y_pred_test = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

print("Target Names of Test : ", dataset.target_names[y_pred_test])
print("Test Accuracy Score : ", accuracy_score(y_test,y_pred_test))
print("Train Accuracy Score : ", accuracy_score(y_train,y_pred_train))
print(confusion_matrix(y_test,y_pred_test))

print("Weights0 : ", clf.coefs_[0])
print("Weights1 : ", clf.coefs_[1])




