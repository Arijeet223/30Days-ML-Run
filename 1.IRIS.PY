import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['species']=iris.target
df.head()

#visualisation
sns.pairplot(df,hue='species')
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True,cmap='coolwarm')
plt.show()


#train test
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

#scaling
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


#train
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)


y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy: ",accuracy)


cm=confusion_matrix(y_test,y_pred)
print(cm)

print(classification_report(y_test,y_pred))



#sample
sample = np.array([[5.1, 3.4, 1.5, 0.2]])
sample = scaler.transform(sample)

prediction = model.predict(sample)
print("Predicted Species:", iris.target_names[prediction][0])