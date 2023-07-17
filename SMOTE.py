import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
  
# load the data set
data = pd.read_csv('Dataset.csv')
  
# print info about columns in the dataframe
print(data.info())


# normalise the amount column
X = StandardScaler().fit_transform(np.array(data['VRR14']).reshape(-1, 1))


data = data.drop(['Readings'], axis = 1)


data['Pred'].value_counts()

from sklearn.model_selection import train_test_split
train,test= train_test_split(data,test_size=0.3)
train_feat=train.iloc[:,:2]
train_targ=train["Pred"]
  
# split into 70:30 ration
X_train, X_test, y_train, y_test = train_test_split(train_feat, train_targ, test_size = 0.3, random_state = 0)
  
train_feat=train.iloc[:,:15]
train_targ=train["Pred"]

# describes info about train and test set
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

# logistic regression object
lr = LogisticRegression()
  
# train the model on train set
lr.fit(X_train, y_train.ravel())
  
predictions = lr.predict(X_test)
  
# print classification report
print(classification_report(y_test, predictions))

from imblearn.over_sampling import RandomOverSampler

# Create the oversampler object
ros = RandomOverSampler(random_state=0)

# Apply the oversampler to the data
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

from imblearn.over_sampling import SMOTE

# Create the oversampler object
sm = SMOTE(random_state=0)

# Apply the oversampler to the data
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

