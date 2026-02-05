import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("Housing.csv")
print(df.head())

b_col=["mainroad","guestroom","basement","hotwaterheating",'airconditioning','prefarea']
for i in b_col:
    df[i]=df[i].map({'yes':1,'no':0})
df=pd.get_dummies(df,columns=["furnishingstatus"],drop_first=True)


print(df.head())

x=df.drop("price",axis=1)
y=df['price']

x_train,x_test ,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("\n model score")
print("\n mean square error: ",mean_squared_error(y_test,y_pred))
print("\n mae: ",mean_absolute_error(y_test,y_pred))
print("\n root mean squared: ",np.sqrt(mean_squared_error(y_test,y_pred)))
print("\n r2 :",r2_score(y_test,y_pred))