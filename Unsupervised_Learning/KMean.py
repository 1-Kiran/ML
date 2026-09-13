import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs,make_moons
from sklearn.preprocessing import StandardScaler

# x, y_true = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=42)
# df=pd.DataFrame(x,columns=['feature1','feature2'])
# scaler=StandardScaler()
# x_scaled=scaler.fit_transform(df)

# inertia=[]
# k_range=range(1,11)
# for k in k_range:
#     kmeans=KMeans(n_clusters=k,random_state=42)
#     kmeans.fit(x_scaled)
#     inertia.append(kmeans.inertia_)
# kmeans_final=KMeans(n_clusters=3,random_state=42)
# cluster_labels=kmeans_final.fit_predict(x_scaled)
# df['cluster']=cluster_labels
# sns.scatterplot(x=df['feature1'], y=df['feature2'], hue=df['cluster'], palette='viridis')
# plt.show()

from sklearn.cluster import DBSCAN
x,y_true=make_moons(n_samples=500,noise=0.05,random_state=42)
df=pd.DataFrame(x,columns=['feature1','feature2'])
scaler=StandardScaler()
x_scaled=scaler.fit_transform(df)

kmeans=KMeans(n_clusters=2,random_state=42)
kmeans_labels=kmeans.fit_predict(x_scaled)
df['kmeans_cluster']=kmeans_labels
sns.scatterplot(x=df['feature1'],y=df['feature2'],hue=df['kmeans_cluster'],palette='tab10')
plt.show()

dbscan=DBSCAN(eps=0.3,min_samples=5)
dbscan_labels=dbscan.fit_predict(x_scaled)

df['dbscan_cluster']=dbscan_labels
sns.scatterplot(x=df['feature1'],y=df['feature2'],hue=df['dbscan_cluster'],palette='tab10')
plt.show()