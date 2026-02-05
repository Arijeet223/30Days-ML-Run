import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score

df=pd.read_csv("churn.csv")
#eda
print(df.head())
df = df.drop("customerID", axis=1)

df.replace(" ", np.nan, inplace=True)
df.dropna(inplace=True)

df['TotalCharges']=pd.to_numeric(df["TotalCharges"],errors='coerce')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("Class 0 (No Churn):", sum(df['Churn'] == 0))
print("Class 1 (Yes Churn):", sum(df['Churn'] == 1))

le =LabelEncoder()

for col in ['gender','Partner','Dependents','PhoneService','PaperlessBilling']:
    df[col]=le.fit_transform(df[col])

df=pd.get_dummies(df)

x=df.drop('Churn',axis=1)
y=df['Churn']
print(df.head())
print(df.columns)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train = x_train.values
x_test = x_test.values


sm = SMOTE(random_state=42)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
print("Before SMOTE:", Counter(y_train))
print("After SMOTE:", Counter(y_train_res))

ratio = 5163 / 1869 

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric='logloss'
)
model.fit(x_train_res, y_train_res)
y_pred=model.predict(x_test)

print("\n accuracy score",accuracy_score(y_test,y_pred))
print("\n confusion matrix",confusion_matrix(y_test,y_pred))
print("\n recall",recall_score(y_test,y_pred))

importances=pd.Series(model.feature_importances_,index=x.columns)
importances.sort_values().plot(kind='barh',figsize=(8,6))
plt.show()

