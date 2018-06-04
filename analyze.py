import pickle
import sys
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

## Defining method to be used in hw1.py.
## Method returns accuracy scores, 
## confusion matrix and top 20 features for each model
def metrics(model,x_test,y_test,feature_names):

## Defining KBest parser and fitting it to data.
	skb = SelectKBest(chi2, k=20)
	skb.fit(x_test,y_test)

## Creating two variables to support parsing of 20 best features.
	mask = skb.get_support()
	best_features = []

## Getting the predicted labels using the model given as a parameter in hw1.
## Reading accuracy of that model.
	y_predicted = model.predict(x_test)
	accuracy = accuracy_score(y_test,y_predicted)

## Defining confusion matrix using predicted and actual labels in test data.
	conf_matrix = confusion_matrix(y_predicted,y_test)

## Calling Pandas to get a clearer version of the confusion matrix.
	true = pd.Series(y_test)
	pred = pd.Series(y_predicted)

## Printing accuracy.
	print "The accuracy of this model is " + str(accuracy) + "."

## Printing confusion matrix as such and using Pandas.
	print "The confusion matrix for the model is, in two formats:"
	print conf_matrix
	print pd.crosstab(true, pred, rownames=['True'], colnames=['Predicted'], margins=True)

## Extracting feature names and appending to list of best features if 'TRUE'
## using support variables created above
	for bool, feature in zip(mask, feature_names):
	    if bool:
	       	best_features.append(feature)

## Printing the 20 best scoring features of each model.
	print "The 20 best scoring features of the model are:"
	i = 1
	for feature in best_features:
		print str(i) + ". " + str(feature)
		i = i + 1

	print true
	print pred
