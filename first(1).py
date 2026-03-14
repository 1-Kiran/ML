import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns #type: ignore
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

df=pd.read_csv("insurance.csv")
data=df.drop_duplicates().copy()
data["sex"]=data["sex"].map({"male":1,"female":0})
data.rename(columns={
    "sex":"is_male",
    "smoker":"is_smoker"
},inplace=True)

#
data=pd.get_dummies(data,columns=["region"])
col=["region_northwest","region_northeast","region_southeast","region_southwest"]

#Bool->Int
data[col]=data[col].astype(int)

#Divide bmi_category based on bmi values
data["bmi_category"]=pd.cut(
    data["bmi"],
    bins=[0,18.5,24.9,29.9,float("inf")],
    labels=["Under","Normal","Over","Obesity"]
)

#It creates boolean values
data=pd.get_dummies(data,columns=["bmi_category"],drop_first=True)

cat_data=["bmi_category_Normal","bmi_category_Over","bmi_category_Obesity"]
data[cat_data]=data[cat_data].astype(int)

col=["age","bmi","children"]

scaler=StandardScaler()
data[col]=scaler.fit_transform(data[col])

selected_features=['age', 'is_male', 'bmi', 'children', 'is_smoker', 'charges',
       'region_northeast', 'region_northwest', 'region_southeast',
       'region_southwest', 'bmi_category_Normal', 'bmi_category_Over',
       'bmi_category_Obesity']

data["is_smoker"]=data["is_smoker"].map({"yes":1,"no":0})

correlation={
    feature: pearsonr(data[feature],data["charges"])[0]
    for feature in selected_features
}
correlation_df=pd.DataFrame(list(correlation.items()),columns=['Feature','Pearson Correlation'])
ok=correlation_df.sort_values(by='Pearson Correlation',ascending=False )