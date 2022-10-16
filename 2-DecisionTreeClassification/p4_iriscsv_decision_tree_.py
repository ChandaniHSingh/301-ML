#implementing Decision-Tree on IRIS CSV

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


design = "="*50;

df = pd.read_csv('iris.csv')

print(design)
print("df.info : \n")

df.info()

print(design)
print("df.head : \n")

print(df.head(5))

print(design)
print("df.describe : \n")

print(df.describe())


#define depentdent and independent attribute

X = df[['sepal_length' , 'sepal_width' , 'petal_length' ,'petal_width']]
y = df[['species']]

#spliting into train and test set

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.33,random_state = 42)

# fitting train set
trclf = DecisionTreeClassifier()
trclf.fit(X_train,y_train)

# predict test set
dataClass = trclf.predict(X_test)


#print(dataClass)
print(design)

print("Score : ",trclf.score(X_test,y_test))
tree.plot_tree(trclf)
plt.show()

