import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv("wine.csv")

print(df.head())

df['good']=[1 if q>=7 else 0 for q in df['quality']]

x=df.drop(['quality','good'],axis=1)
y=df['good']
print(df.head())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=200,random_state=45)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("\n accuracy: ",accuracy_score(y_test,y_pred))
print("\n confuison matrix: ",confusion_matrix(y_test,y_pred))

#importances
importances = pd.Series(model.feature_importances_, index=x.columns)
importances.sort_values().plot(kind='barh', figsize=(8,6))
plt.show()
