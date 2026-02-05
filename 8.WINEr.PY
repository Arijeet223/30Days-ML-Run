import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error

df=pd.read_csv("wine.csv")

print(df.head())

df['good']=[1 if q>=7 else 0 for q in df['quality']]

x=df.drop(['quality'],axis=1)
y=df['quality']
print(df.head())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestRegressor(n_estimators=200,random_state=45)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print("\n mse: ",mse)
print("\n rmse: ",np.sqrt(mse))

