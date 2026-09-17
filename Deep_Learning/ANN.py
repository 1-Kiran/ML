#ANN and Perceptron

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# TensorFlow and Keras (DL Utilites)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')

df=sns.load_dataset("iris")
# sns.pairplot(df,hue='species')
# plt.show()

x=df.drop(columns=['species'],axis=1)
y=df['species']
encoder=LabelEncoder()
y_int=encoder.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y_int,test_size=0.2,random_state=42,stratify=y_int)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)

per=Perceptron(max_iter=1000,random_state=42)
per.fit(x_train_scaled,y_train)
y_pred_per=per.predict(x_test_scaled)

model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train_scaled,y_train)
y_pre=model.predict(x_test_scaled)
acc=accuracy_score(y_test,y_pre)
accuracy=accuracy_score(y_test,y_pred_per)
print(acc)
# model=Sequential([])