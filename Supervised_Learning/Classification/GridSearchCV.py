# iris

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df=sns.load_dataset('iris')
# print(df['species'].unique())

# model=KNeighborsClassifier(n_neighbors=5)
model=SVC(C=20,kernel='linear',gamma='auto')
print(df.columns)
gscv=GridSearchCV(model,{
    'C':[1,10,20,30],
    'kernel':['rbf','linear']
},cv=5,return_train_score=False)

x=df.drop('species',axis=1)
y=df['species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
model.fit(x_train,y_train)
score=model.score(x_test,y_test)
gscv.fit(x_train,y_train)
pre=gscv.predict(x_test)

# result=pd.DataFrame(gscv.cv_results_)
# print(result[['param_C','param_kernel','mean_test_score']])

# # m=pd.DataFrame([[2.1, 1.5, 6.4,0.1]],columns=x.columns)
# # y_pre=model.predict(m)