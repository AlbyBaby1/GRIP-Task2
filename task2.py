import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn import datasets

#load the data set
iris_data=datasets.load_iris()

iris=pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
iris.head()
iris.describe()

#normalization function
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#normalize the numerical data
df_norm=norm_fun(iris.iloc[:,1:5])
df_norm

y=iris.iloc[:, [0, 1, 2]].values

#scree plot or elbow curve
TWSS=[]
k=list(range(2,11))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(y)
    TWSS.append(kmeans.inertia_)
TWSS

#elbow curve 
plt.plot(k,TWSS,'ro-');plt.title("Elbow method");plt.xlabel('no of clusters');plt.ylabel('Total within SS')   
plt.show()

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(y)


#visualizing clusters
plt.scatter(y[y_kmeans == 0, 0], y[y_kmeans== 0, 1],s = 100,c='red', label = 'Iris-setosa')
plt.scatter(y[y_kmeans == 1, 0], y[y_kmeans == 1, 1],s = 100,c = 'blue', label = 'Iris-versicolour')
plt.scatter(y[y_kmeans== 2, 0], y[y_kmeans == 2, 1],s = 100, c = 'green', label = 'Iris-virginica')


plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 100, c = 'yellow', label = 'Centroids')


