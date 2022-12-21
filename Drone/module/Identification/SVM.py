import torch
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

names = np.array(torch.load("./userdata/names.pt"))
embeddings = np.array(torch.load("./userdata/database.pt"))

names_label=preprocessing.LabelEncoder().fit_transform(names)
label2name=dict(zip(names_label,names))

x_train,x_test,y_train,y_test=train_test_split(embeddings,names_label,test_size=0.3,random_state=0)

clf=SVC()
clf.fit(embeddings,names_label)
print(clf.score(x_test,y_test))

result=clf.predict(x_test)

joblib.dump(clf,'userdata/clf.pkl')

