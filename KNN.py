import numpy as np
import pandas as pd

df=pd.read_csv("Dataset.csv")
df.head()

to_drop=['Readings']
df.drop(to_drop, inplace=True,axis=1)
df.head()

df=pd.read_csv("Dataset.csv")

from sklearn.model_selection import train_test_split

train,test= train_test_split(df,test_size=0.3)

train_feat=train.iloc[:,:15]
train_targ=train["Pred"]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(train_feat,train_targ, test_size=0.3)

#Fine KNN
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=1)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Faultless', 'Faulty'])

import matplotlib.pyplot as plt
cm_display.plot()
plt.show()

Accuracy = metrics.accuracy_score(y_test, y_pred)
acc=Accuracy*100
print(acc)

#Medium KNN
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=10)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)


from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Faultless', 'Faulty'])

import matplotlib.pyplot as plt
cm_display.plot()
plt.show()

Accuracy = metrics.accuracy_score(y_test, y_pred)
acc=Accuracy*100
print(acc)

#Coarse KNN
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=100)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Faultless', 'Faulty'])

import matplotlib.pyplot as plt
cm_display.plot()
plt.show()

Accuracy = metrics.accuracy_score(y_test, y_pred)
acc=Accuracy*100
print(acc)


