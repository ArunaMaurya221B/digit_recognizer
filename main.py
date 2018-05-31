#importing libraries

import pandas as pd
import numpy as nu
import matplotlib.pyplot as pt
from sklearn import tree

data = pd.read_csv("/home/icts/digit_recognizer/train.csv").as_matrix()
#print data

#create training and testing data

feature_train = data[0:2100,1:]
feature_test = data[2100:,1:]
label_train = data[0:2100,0]
label_test = data[2100:,0]

#creating classifier and predicting

clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature_train, label_train)
x = feature_test[8]
print (clf.predict([feature_test[8]]))
x.shape=(28,28)
pt.imshow(255-x, cmap='gray')
pt.show()