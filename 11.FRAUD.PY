import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,ConfusionMatrixDisplay

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

df=pd.read_csv("creditcard.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

print("\n missing values")
print(df.isnull().sum())

print("\nFraud Distribution")
print(df["Class"].value_counts())

#eda

plt.figure(figsize=(5,4))
df['Class'].value_counts().plot(kind="bar")
plt.title("fraud vs legit")
plt.xlabel("class")
plt.ylabel("count")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()


scaler=StandardScaler()
df["Amount"]=scaler.fit_transform(df[["Amount"]])

x=df.drop("Class",axis=1)
y=df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#normal=0.9993

#undersampling=0.9611 but better as more fraud were captured = fn high


rus=RandomUnderSampler(random_state=42)
X_train_under,Y_train_under=rus.fit_resample(X_train,y_train)

print("original train set")
print(y_train.value_counts())


print("\n after undersampling")
print(Y_train_under.value_counts())

#oversampling-0.9891

smote=SMOTE(random_state=42)

X_train_smote,Y_train_smote=smote.fit_resample(X_train,y_train)


print("Original Training Set")
print(y_train.value_counts())

print("\nAfter SMOTE")
print(Y_train_smote.value_counts())

model=LogisticRegression(max_iter=1000)
model.fit(X_train_smote,Y_train_smote)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm=confusion_matrix(y_test,y_pred)

print(cm)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()