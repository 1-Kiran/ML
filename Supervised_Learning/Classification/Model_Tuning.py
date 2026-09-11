"""
1. Model Tuning
2. Hyperparameter
3. Cross-Validation
4. GridSearchCV

"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings("ignore")

df=sns.load_dataset("titanic")
df.drop(columns=['deck', 'alive', 'class', 'who', 'adult_male', 'embark_town','fare'],inplace=True)


df['embarked'].fillna('S',inplace=True)
col=['embarked','sex']
# print(df.head())

df['age'].fillna(df['age'].mean(),inplace=True)
df=pd.get_dummies(df,columns=col,drop_first=True)
df=df.astype(int)

scaler=StandardScaler()
std_col=['pclass','age', 'sibsp']

df[std_col]=scaler.fit_transform(df[std_col])

x=df.drop(columns='survived',axis=1)
y=df['survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(max_depth=5,criterion="entropy",splitter="random") # This give more accurary then other parameters
# criterion: Literal['gini', 'entropy', 'log_loss'] = "gini",
# splitter: Literal['best', 'random'] = "best",

# model=SVC(kernel='rbf') #Cross-Validation:  0.8293892411022534

# model=KNeighborsClassifier(n_neighbors=5,weights='uniform') #Cross-Validation:  0.8159814198732033
search=GridSearchCV(model)

model.fit(x_train,y_train)
y_pre=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pre)
confusion_mat=confusion_matrix(y_test,y_pre)
classification_rep=classification_report(y_test,y_pre)

print("Cross-Validation: ",cross_val_score(model,x,y,cv=5,scoring='accuracy').mean())

# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", confusion_mat)
# print("classification Report:\n", classification_rep)