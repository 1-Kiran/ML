from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, BaggingClassifier, AdaBoostClassifier,GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score

df=sns.load_dataset('iris')
x=df.drop('species',axis=1)
encoder=LabelEncoder()
yl=encoder.fit_transform(df['species'])
y=yl
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# base_learners=[
#     ('dt',DecisionTreeClassifier(random_state=42)),
#     ('svc',SVC(probability=True,kernel='rbf',random_state=42)),
#     ('lr',LogisticRegression(max_iter=1000))
# ]
# meta_learner=LogisticRegression(max_iter=1000)
# stacking_clf=StackingClassifier(
#     estimators=base_learners,
#     final_estimator=meta_learner,
#     cv=5
# )

# boosting_clf= RandomForestClassifier(n_estimators=100,max_features='sqrt',max_depth=None, random_state=42)
# ada_model=AdaBoostClassifier(n_estimators=10,random_state=42)
# gb_model=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,random_state=42)

xg_model=XGBClassifier(n_estimators=100,max_depth=4)
xg_model.fit(x_train,y_train)
pre=xg_model.predict(x_test)
print(accuracy_score(y_test,pre))