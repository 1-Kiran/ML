import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

x,y=make_blobs(n_features=5,n_samples=500,cluster_std=1.5,random_state=42)
df=pd.DataFrame(x,columns=['feature1','feature2','feature3','feature4','feature5'])
pca=PCA(n_components=2)
df_model=pca.fit_transform(df)
df_pca=pd.DataFrame(df_model,columns=['pca1','pca2'])
df_pca['label']=y
sns.scatterplot(x=df_pca['pca1'],y=df_pca['pca2'],hue=df_pca['label'],palette='Set2')
plt.show()