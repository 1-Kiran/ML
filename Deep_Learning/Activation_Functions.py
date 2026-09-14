import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import pandas as pd
from sklearn.preprocessing import Normalizer
warnings.filterwarnings('ignore')

df = pd.DataFrame({
    "soil_moisture": [0.10, 0.15, 0.20, 0.25, 0.40, 0.60, 0.35, 0.18,
                      0.45, 0.05, 0.80, 0.27, 0.55, 0.70, 0.12, 0.30],
    "temperature_c": [34, 30, 26, 22, 28, 30, 19, 22,
                      35, 24, 33, 33, 21, 25, 20, 29],
    "sunlight_hours": [9, 8, 7, 4, 8, 10, 3, 10,
                       12, 5, 9, 11, 2, 6, 1, 9],
    "needs_water": [1, 1, 1, 0, 0, 0, 0, 1,
                    0, 1, 0, 1, 0, 0, 1, 1]
})
x=df.drop(columns='needs_water',axis=1)
x_min=x.min()
x_max=x.max()
x_scaled=(x-x_min) / (x_max-x_min + 1e-8)
y=df['needs_water']

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.25,random_state=42)

model=keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(8,activation='relu'),
    layers.Dense(1,activation='sigmoid')
    ])
opt=optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy']) # or optimizer='sgd'
ok=model.fit(x_train.values,
        y_train.values,
        validation_data=(x_test.values,y_test.values),
        epochs=10,
        batch_size=4,
        verbose=1
        )
