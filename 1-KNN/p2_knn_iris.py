# implementing kNN algorithm on IRIS Dataset

import sklearn as sk
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


dataset = datasets.load_iris()

print("Dataset : ")
print(dataset)


X = dataset.data
y = dataset.target

#spliting dataset in 2 group
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state = 42)

print("Training Features : ")
print(X_train)
print("Training Labels : ")
print(y_train)
print("Testing Features : ")
print(X_test)
print("Testing Labels : ")
print(y_test)

# model defining
model = KNeighborsClassifier(n_neighbors = 12,weights = 'distance',metric='euclidean')

#model training (fit)
model.fit(X_train,y_train)

# model testing (predict)
dataClass = model.predict(X_test)

#print(dataset.target_names[dataClass])

#accuracy_score
print("Accuracy_score : ")
print(accuracy_score(y_test,dataClass,normalize=True,sample_weight=None))

#confusion_matrix
print("Confusion_matrix : ")
print(confusion_matrix(y_test,dataClass,labels=[0,1,2]))
