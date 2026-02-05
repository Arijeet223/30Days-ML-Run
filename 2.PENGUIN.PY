import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import pickle

penguins=sns.load_dataset("penguins")
print("initial data")
print(penguins.head())

penguins=penguins.dropna()
print("\n after dropping na values: ",penguins.shape)

le=LabelEncoder()
penguins['species']=le.fit_transform(penguins['species'])

penguins=pd.get_dummies(penguins,columns=["sex","island"],drop_first=True)

print("\n data after encoding")
print(penguins.head())

x=penguins.drop('species',axis=1)
y=penguins["species"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print("\n train and test shape")
print(x_train.shape,x_test.shape)

model=RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("\n accuracy score:",accuracy_score(y_test,y_pred))
print("\n classification report: ",classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True,fmt="d",cmap="Blues")
plt.title("confusion matrix")
plt.show()


with open("penguin_model.pkl","wb") as f:
    pickle.dump(model,f)

print("\n model saved")

loaded_model=pickle.load(open("penguin_model.pkl","rb"))
sample_pred=loaded_model.predict([x.iloc[0]])

print("sample prediction:",sample_pred)
print("actual:",y.iloc[0])