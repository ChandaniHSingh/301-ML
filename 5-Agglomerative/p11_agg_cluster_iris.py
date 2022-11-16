# P11. Program to perform clustering on Iris Dataset

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import homogeneity_score
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage

# load dataset

iris = datasets.load_iris()
X = iris.data
y = iris.target


# model defining

#Cluster = AgglomerativeClustering(n_clusters=3,linkage="single")
#Cluster = AgglomerativeClustering(n_clusters=3,linkage="complete")
Cluster = AgglomerativeClustering(n_clusters=3,linkage="average")
#Cluster = AgglomerativeClustering(n_clusters=3,linkage="ward")

#model training (fit) 
# model testing (predict)

pred_y=Cluster.fit_predict(X)

plt.scatter(X[pred_y == 0,0],X[pred_y == 0,1],s=100,c="red", label="Iris-Sentosa")
plt.scatter(X[pred_y == 1,0],X[pred_y == 1,1],s=100,c="blue", label="Iris-Versicolour")
plt.scatter(X[pred_y == 2,0],X[pred_y == 2,1],s=100,c="green", label="Iris-Verginica")

plt.legend()
plt.show()

print("Score=",homogeneity_score(y,pred_y))


print('......Program Ends.........')

'''
X1 = list(X)
y1 = list(y)
data = list(zip(X1, y1))
print(data)
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.show()
'''

