# -*- coding: utf-8 -*-

import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
import analyze as an
from stop_words import get_stop_words

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import warnings

def main():

### Ignore depreiation warnings from NumPy
	warnings.filterwarnings("ignore", category=DeprecationWarning)

### Loading the data from the arguments given in the command line
	with open(sys.argv[1], 'r') as my_file:
	    data_train = pd.read_csv(my_file, sep="\t", header=None, names = ["corpus", "target"])

	with open(sys.argv[2], 'r') as my_file:
	    data_dev = pd.read_csv(my_file, sep="\t", header=None, names = ["independent", "target"])

	print('Data loaded.')

### Setting the main variables
	corpus_train = data_train['corpus']
	y_train = data_train['target'].tolist()
	x_dev = data_dev['independent']
	y_dev = data_dev['target'].tolist()

### Remove stop words and some other words with the pre_process function.
	corpus_train = pre_process(corpus_train)
	x_dev = pre_process(x_dev)

### Calling the unigram models using function defined below
	print "STARTING TESTS WITH UNIGRAMS AND THREE DIFFERENT MODELS"
	best_unigram_model, unigram_accuracy = ngram_three_models(1,corpus_train,y_train,x_dev,y_dev)
	
### Calling the bigram models using function defined below
	print "STARTING TESTS WITH BIGRAMS AND THREE DIFFERENT MODELS"
	best_bigram_model, bigram_accuracy = ngram_three_models(2,corpus_train,y_train,x_dev,y_dev)

### Calling the triigram models using function defined below	
	print "STARTING TESTS WITH TRIGRAMS AND THREE DIFFERENT MODELS"
	best_trigram_model, trigram_accuracy = ngram_three_models(3,corpus_train,y_train,x_dev,y_dev)

### Printing the best accuracy scores for each ngram
	print "The best unigram model was " + str(best_unigram_model) + " with an accuracy of " + str(unigram_accuracy)
	print "The best bigram model was " + str(best_bigram_model) + " with an accuracy of " + str(bigram_accuracy)
	print "The best trigram model was " + str(best_trigram_model) + " with an accuracy of " + str(trigram_accuracy)

### Calling Classify as an external script, not a function
	os.system('python classifier.py train_newline.txt dev_newline.txt')

### Defining a function that trains the data on three different 
### models and an ngram with the degree given in the function call
def ngram_three_models(ngram,x_train,y_train,x_dev,y_dev):

### Starting with vectorizing the data
	count_vectorizer = CountVectorizer(ngram_range=(ngram,ngram), analyzer = 'word')
	train_vectorized = count_vectorizer.fit_transform(x_train)
	dev_vectorized = count_vectorizer.transform(x_dev)
	feature_names = count_vectorizer.get_feature_names()

### Setting empty variables for storing the accuracy and names of the best models	
	accuracy = 0
	best_classifier = ""

### Running the multinomial Bayes Classifier 
### and testing it's accuracy and saving if higher than other models
	print "Testing the multinomial Naive Bayes classifier with n = %s " % (ngram)
	bayes_clf = MultinomialNB().fit(train_vectorized, y_train)
	y_predicted = bayes_clf.predict(dev_vectorized)
	bayes_accuracy = np.mean(y_predicted == y_dev) 
	print "The accuracy of the Naive Bayes classifier with %sgram was " % (ngram) + str(bayes_accuracy)  
	if bayes_accuracy > accuracy:
		accuracy = bayes_accuracy
		best_model = bayes_clf
		best_classifier = "Naive Bayes Classifier"
		best_y_predicted = y_predicted

### Running the SVM Classifier
### and testing it's accuracy and saving if higher than other models
	print "Testing the SVM classifier with n = %s " % (ngram)
	svm_clf = SGDClassifier(loss='hinge', penalty='l2',
			alpha=1e-3, random_state=42,
			max_iter=5, tol=None).fit(train_vectorized, y_train)
	y_predicted = svm_clf.predict(dev_vectorized)
	svm_accuracy = np.mean(y_predicted == y_dev) 
	print "The accuracy of the SVM classifier with %sgram was " % (ngram) + str(svm_accuracy) 
	if svm_accuracy > accuracy:
		accuracy = svm_accuracy
		best_model = svm_clf
		best_classifier = "SVM Classifier"
		best_y_predicted = y_predicted

### Running the Logistic Regression Classifier
### and testing it's accuracy and saving if higher than other models
	print "Testing the Logistic regression classifier with n = %s " % (ngram)
	log_clf = LogisticRegression().fit(train_vectorized, y_train)
	y_predicted = log_clf.predict(dev_vectorized)
	log_accuracy = np.mean(y_predicted == y_dev) 
	print "The accuracy of the Logistic regression classifier with %sgram was " % (ngram) + str(log_accuracy) 
	if log_accuracy > accuracy:
		accuracy = log_accuracy
		best_model = log_clf
		best_classifier = "Logistic Regression Classifier"
		best_y_predicted = y_predicted

### Printing the name and accuracy of the best classifier
	print "The best %sgram classifier is %s with an accuracy of %s " % (ngram,best_classifier,accuracy) 

### Calling analyze as a function on the best scoring model
	an.metrics(best_model,dev_vectorized,y_dev,feature_names)
	return best_classifier, accuracy

### Defining pre_process function
def pre_process(corpus_train):
	process_corpus = []
	clean_corpus = []

### Common stop words from the stop-words Python library.
	stop = get_stop_words('en')
### Some words that occurred in the top 20 features of the early bi- and
### trigram tests.
	delete_list = ["unfollowers","unfollower","followed", "http", "bit", 
		"ly" ,"www", "youtube", "stats", "via", "follower", "followers", 
		"com", "ow","//t","https","rt","the","to","and","you","lt","that",
		"one","person"]

### Iterating the corpus twice, once for stop words and the second time for 
### the custom delete_list.
	for row in corpus_train:
		row_words = row.split()
		resultwords  = [word for word in row_words if word.lower() not in delete_list]
		result = ' '.join(resultwords)
		process_corpus.append(result)

	for row in process_corpus:
		row_words = row.split()
		resultwords  = [word for word in row_words if word.lower() not in delete_list]
		result = ' '.join(resultwords)
		clean_corpus.append(result)

### Return preprocessed corpus.
	return clean_corpus

if __name__ == '__main__':
    main()