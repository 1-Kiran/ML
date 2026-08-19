import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv("ford.csv")
col=["model","transmission","fuelType"]
X=df.drop(columns=["price"],axis=1)
y=df["price"]

#one-hot encode
x_one_encode=pd.get_dummies(X,columns=col,drop_first=True)
x_one_encode=x_one_encode.astype(int)

#Label encode
encode=LabelEncoder()
xlabel=X.copy()
for i in col:
    xlabel[i]=encode.fit_transform(xlabel[i])
cols=["year","mileage","tax","mpg","engineSize"]
all_col=["model","year","transmission","mileage","fuelType","tax","mpg","engineSize"]

#standard deviation
scalar=StandardScaler()
#one_hot std
x_one_encode[cols]=scalar.fit_transform(x_one_encode[cols])
#label std
xlabel[all_col]=scalar.fit_transform(xlabel[all_col])

#LinearRegression Model
model=LinearRegression()

#For one_hot
X_train, X_test, y_train, y_test = train_test_split(x_one_encode, y, test_size=0.20, random_state=42)
model.fit(X_train,y_train)
#For labeled data
# X_train, X_test, y_train, y_test = train_test_split(xlabel, y, test_size=0.20, random_state=42)
# model.fit(X_train,y_train)

#predicting data
y_pre=model.predict(X_test)

#score of prediction
r2=r2_score(y_test,y_pre)
print(r2)
