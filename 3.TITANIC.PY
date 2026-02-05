import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv("tested.csv")
print("dataset loaded")

print(df.head())

print(df.isnull().sum())

#data cleaning

df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.drop(columns=['Cabin'],inplace=True)
df.drop(columns=['PassengerId','Name','Ticket'],inplace=True)

print(df.head())

le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])
df['Embarked']=le.fit_transform(df['Embarked'])

print(df.head())

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Model Trained Successfully")

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)