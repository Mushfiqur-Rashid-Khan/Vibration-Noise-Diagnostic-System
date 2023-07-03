#python libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import model_selection

#Defining Linear Support Vector Classifier
lr=LogisticRegression()
svc=LinearSVC(C=1.0)
rfc=RandomForestClassifier(n_estimators=100)

#Uploading the Dataset
df=pd.read_csv("Dataset.csv")

from sklearn.model_selection import train_test_split

#Defining Training Testing Ratio 70:30
train,test= train_test_split(df,test_size=0.3)

#Training and Testing
train_feat=train.iloc[:,:15]
train_targ=train["class"]


print("{0:0.2f}% in training set".format((len(train_feat)/len(df.index)) * 100))
print("{0:0.2f}% in testing set".format((1-len(train_feat)/len(df.index)) * 100))


#Defining seed, number of trees, number of estimatos, and random state
seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, train_feat, train_targ, cv=kfold)
acc=results.mean()
acc1=acc*100
print("The accuracy is: ",acc1,'%')
