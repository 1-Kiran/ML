import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns #type: ignore
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
import sheryanalysis as sh

warnings.filterwarnings("ignore")

data=pd.read_csv("heart.csv")
# df=data.drop_duplicates().copy()
# # print(df.info())
# df["Sex"]=df["Sex"].map({"M":1,"F":0})
# # print(df["Sex"].dtype)
# # print(df["ChestPainType"].value_counts())
# df=pd.get_dummies(df,columns=["ChestPainType","ST_Slope","RestingECG","ExerciseAngina"],drop_first=())
# col=['ChestPainType_ASY', 'ChestPainType_ATA',
#        'ChestPainType_NAP', 'ChestPainType_TA', 'ST_Slope_Down',
#        'ST_Slope_Flat', 'ST_Slope_Up', 'RestingECG_LVH', 'RestingECG_Normal',
#        'RestingECG_ST', 'ExerciseAngina_N', 'ExerciseAngina_Y']
# df[col]=df[col].astype(int)
# x=df.drop("HeartDisease",axis=1)
# y=df["HeartDisease"]
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# data["HeartDisease"].value_counts().plot(kind="bar")
# print(data.isnull().sum())


# col_m=data.loc[data["Cholesterol"] != 0,"Cholesterol"].mean()
# data["Cholesterol"]=data["Cholesterol"].replace(0,col_m)
# data["Cholesterol"]=data["Cholesterol"].round(2)

# def graph(var,num):
#     plt.subplot(2,2,num)
#     sns.histplot(var,kde=True)
# graph(data["Age"],1)
# graph(data["RestingBP"],2)
# graph(data["Cholesterol"],3)
# graph(data["MaxHR"],4)
# plt.tight_layout()
# plt.show()

# ok=sh.analyze(data)
# print(ok)

# sns.countplot(x=data["FastingBS"],hue=data["HeartDisease"])
# plt.show()

print(data["ChestPainType"].value_counts)