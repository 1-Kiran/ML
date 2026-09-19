import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, Flatten, MaxPooling2D, Dropout)
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')
df=pd.read_csv('Deep_Learning\mnist_train.csv')
df_test=pd.read_csv('Deep_Learning\mnist_test.csv')

x_train=df.drop('label',axis=1).values
y_train=df['label'].values
x_test=df_test.drop('label',axis=1).values
y_test=df_test['label'].values

x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

x_train_img=x_train.reshape(-1,28,28)
x_test_img=x_test.reshape(-1,28,28)

y_train_cat=to_categorical(y_train,10)
y_test_cat=to_categorical(y_test,10)

perceptron=Sequential([
    Flatten(input_shape=(28,28)),
    Dense(10,activation='softmax')
])

perceptron.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

history=perceptron.fit(x_train_img,y_train_cat,
                       validation_data=(x_test_img,y_test_cat),
                       epochs=6,batch_size=30,verbose=0)

acc=perceptron.evaluate(x_test_img,y_test_cat,verbose=0)[1]

print(acc)

ann=Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
])
ann.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])

ann_history=ann.fit(x_train_img,y_train_cat,
                validation_data=(x_test_img,y_test_cat),
                epochs=5,batch_size=36,verbose=0)
ann_accuracy=ann.evaluate(x_test_img,y_test_cat,verbose=0)[1]
print(ann_accuracy)


x_train_cnn=x_train.reshape(-1,28,28,1) # if the color is RGB then it becomes : x_train.reshape(-1,28,28,3)
x_test_cnn=x_test.reshape(-1,28,28,1) # The last number 1 represent the grayscale image and the last position represent the color

cnn=Sequential([                                          # 28,28->height and weidth, 1 is number of channels
    Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)), # 32 -> This means the layer has 32 filters, kernel_size -> The filter is 3x3 feature maps
    MaxPooling2D(pool_size=(2,2)), #move pixel size 2
    Conv2D(64,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5), # It will deactivates the neurons half(50%) of 128 and it will reduces the overfitting
    Dense(10,activation='softmax')
])
cnn.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
cnn_history=cnn.fit(x_train_cnn,y_train_cat,verbose=0,validation_data=(x_test_cnn,y_test_cat),
                    epochs=5,batch_size=32)
cnn_accuracy=cnn.evaluate(x_test_cnn,y_test_cat,verbose=0)[1]
print(cnn_accuracy)

def plot_training(history,title):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f"{title} Accuracy")

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f"{title} Loss")
    plt.legend()
    plt.show()
plot_training(history,"Perceptron")