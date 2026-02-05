import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC

df=pd.read_csv("brca.csv")
print(df.head())
le=LabelEncoder()
df['y']=le.fit_transform(df['y'])
print(df.head())

x=df.drop("y",axis=1)
y=df['y']

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=45
)
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  
svm_model.fit(x_train, y_train)

y_pred=svm_model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
