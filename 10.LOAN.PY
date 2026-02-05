import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

df=pd.read_csv("loan.csv")
df.columns = df.columns.str.strip()

print(df.head())
print(df.columns.tolist())

df=pd.get_dummies(df,drop_first=True)

df=df.drop("loan_id",axis=1)
print(df.head())
x=df.drop('loan_status_ Rejected',axis=1)
y=df['loan_status_ Rejected']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=RandomForestClassifier(n_estimators=200,random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("\n accuracy score: ",accuracy_score(y_test,y_pred))
print("\n confusion matrix: ",confusion_matrix(y_test,y_pred))

