import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score,roc_curve,classification_report
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

df=pd.read_csv("diabetes.csv")
print(df.head())

print(df.info())
print(df.describe())

plt.figure(figsize=(8,5))
sns.countplot(x=df['Outcome'])
plt.title("Diebetes Prediction")
plt.show()

plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title("Correlation heatmap")
plt.show()

cols =["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for i in cols:
    df[i]=df[i].replace(0,np.nan)
    df[i]=df[i].fillna(df[i].median())

x=df.drop("Outcome",axis=1)
y=df['Outcome']

x_train_full,x_test,y_train_full,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

x_train,x_val,y_train,y_val=train_test_split(x_train_full,y_train_full,test_size=0.2,random_state=42,stratify=y_train_full)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_val_scaled=scaler.fit_transform(x_val)
x_train_full_scaled=scaler.fit_transform(x_train_full)
x_test_scaled=scaler.fit_transform(x_test)

log_req=LogisticRegression(max_iter=200)
log_req.fit(x_train_scaled,y_train)
y_val_predict=log_req.predict(x_val_scaled)
print("classification report: ",classification_report(y_val,y_val_predict))

svm=SVC(kernel="rbf",probability=True)
svm.fit(x_train_scaled,y_train)
y_val_pred = svm.predict(x_val_scaled)
print("SVM")
print(classification_report(y_val, y_val_pred))

rfc=RandomForestClassifier(n_estimators=200,random_state=42)
rfc.fit(x_train_scaled,y_train)
y_val_predict=rfc.predict(x_val_scaled)
print("Classification Report",classification_report(y_val,y_val_predict))

scale_pos_weight = (len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(x_train, y_train)

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

xgb.fit(
    X_train_res, y_train_res,
    eval_set=[(x_val, y_val)],
    verbose=False
)
y_val_pred = xgb.predict(x_val_scaled)
print("XGBoost")
print(classification_report(y_val, y_val_pred))


models={
    "logistic regression":log_req,
    "random forest":rfc,
    "support vector machine":svm,
    "XGBoost":xgb
}

for name,model in models.items():
    pred = model.predict(x_val_scaled if name!="XGBoost" else x_val)
    auc = roc_auc_score(y_val, pred)
    print(name, "AUC =", auc)

best_model=log_req
y_test_pred=log_req.predict(x_test_scaled)
print("Final test performance")
print(classification_report(y_test,y_test_pred))