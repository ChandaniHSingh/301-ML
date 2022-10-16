#3. Implementation of KNN on Breast Cancer Dataset

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load Dataset
df = pd.read_csv('BreastCancerDataset.csv')

print(df.info())
print(df.head(5))
print(df.describe())

# Decide Dependent & Independent Attributes
X = df[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']]
y = df[['diagnosis']]


# Split Train & Test Dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

print("Training Features : ")
print(X_train)
print("Training Labels : ")
print(y_train)
print("Testing Features : ")
print(X_test)
print("Testing Labels : ")
print(y_test)


# fit Train set
model = KNeighborsClassifier(n_neighbors = 12,weights = 'distance',metric='euclidean')



#================================Warning========
model.fit(X_train,y_train)

# predict
dataClass = model.predict(X_test)

#accuracy_score
print("Accuracy_score : ")
print(accuracy_score(y_test,dataClass,normalize=True,sample_weight=None))

#confusion_matrix
print("Confusion_matrix : ")
print(confusion_matrix(y_test,dataClass))

