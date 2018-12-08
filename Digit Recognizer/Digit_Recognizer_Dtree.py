import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset= pd.read_csv('train.csv');
X = dataset.iloc[0:,1:].values
y = dataset.iloc[0:, 0].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

"""d = X_test[8]
d.shape=(28,28)
plt.imshow(d,cmap='gray')
plt.show()"""

dataset2=pd.read_csv('test.csv')
X_2 = dataset.iloc[0:,1:].values
y_2 = dataset.iloc[0:, 0].values

y_2_pred = clf.predict(X_2)
cm2=confusion_matrix(y_2,y_2_pred)
print(cm2)