# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:58:36 2017

@author: ANEESH
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split

from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('C:\Users\ANEESH\Desktop\quora_features.csv')

train_data_columns = data.columns.drop(['question1','question2', 'is_duplicate'])

X = normalize(data[train_data_columns])

x_train,x_test, y_train,y_test = train_test_split(X,
                                                  data['is_duplicate'],
                                                  random_state = 1)


result_cols = ["Classifier", "Accuracy"]
result_frame = pd.DataFrame(columns=result_cols)

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]


for clf in classifiers:
    name = clf.__class__.__name__
  
    clf.fit(x_train, y_train)
    
    predicted = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test,predicted)
    print (name+' accuracy = '+str(acc*100)+'%')
    acc_field = pd.DataFrame([[name, acc*100]], columns=result_cols)
    result_frame = result_frame.append(acc_field)
    
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=result_frame, color="r")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

