import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
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

model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
y_pre=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pre)
confusion_mat=confusion_matrix(y_test,y_pre)
classification_rep=classification_report(y_test,y_pre)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
print("classification Report:\n", classification_rep)


"""
# Output :-

Accuracy: 0.8156424581005587
Confusion Matrix:
 [[95 10]
 [23 51]]
classification Report:
               precision    recall  f1-score   support

           0       0.81      0.90      0.85       105
           1       0.84      0.69      0.76        74

    accuracy                           0.82       179
   macro avg       0.82      0.80      0.80       179
weighted avg       0.82      0.82      0.81       179

"""