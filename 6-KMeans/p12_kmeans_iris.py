# P12. Program of KMeans Clustering on Iris Dataset

from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score

# load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
err=[]
for i in range(1,11):
     est = KMeans(n_clusters=i, n_init=25, init="k-means++", random_state=0)
     Tpred_y=est.fit_predict(X)
     err.append(est.inertia_)

plt.plot(range(1,11), err)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Error')
plt.show()

# define model
Fest = KMeans(n_clusters=3, n_init=25, init="k-means++", random_state=0,tol=1e-06)

# training and testing
pred_y=Fest.fit_predict(X)

plt.scatter(X[pred_y == 0,0],X[pred_y == 0,1],s=100,c="red", label="Iris-Sentosa")
plt.scatter(X[pred_y == 1,0],X[pred_y == 1,1],s=100,c="blue", label="Iris-Versicolour")
plt.scatter(X[pred_y == 2,0],X[pred_y == 2,1],s=100,c="green", label="Iris-Verginica")
plt.scatter(Fest.cluster_centers_[:,0], Fest.cluster_centers_[:,1],s=100,c="yellow", label="Centroids")

plt.legend()
plt.show()

print("Score=",homogeneity_score(y,pred_y))


print('......Program Ends.........')



