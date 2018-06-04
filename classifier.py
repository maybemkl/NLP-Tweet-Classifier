# -*- coding: utf-8 -*-

import os
import sys
import pickle
import emoji
import numpy as np
import pandas as pd
import scipy.sparse
from stop_words import get_stop_words

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def main():
## Loading the training and test data from the command line.
	with open(sys.argv[1], 'r') as my_file:
	    data_train = pd.read_csv(my_file, sep="\t", header=None, names = ["corpus", "target"])

	with open(sys.argv[2], 'r') as my_file:
	    data_dev = pd.read_csv(my_file, sep="\t", header=None, names = ["independent", "target"])

## Setting the main variables, same style as in hw1.py.
	corpus_train = data_train['corpus']
	y_train = data_train['target']
	x_dev = data_dev['independent']
	y_dev = data_dev['target']

### Remove stop words and some other words with the pre_process function.
	corpus_train = pre_process(corpus_train)
	x_dev = pre_process(x_dev)

## Defining vectorized and fitting both training and dev data to it.
	count_vectorizer = CountVectorizer(ngram_range=(1,1), analyzer = 'word')
	train_vectorized = count_vectorizer.fit_transform(corpus_train)
	dev_vectorized = count_vectorizer.transform(x_dev)

## Training the best scoring model and getting it's accuracy score.
	print "Training the best scoring model, which was the multinomial Naive Bayes classifier with unigrams."
	model = MultinomialNB().fit(train_vectorized, y_train)
	y_predicted = model.predict(dev_vectorized)
	accuracy = np.mean(y_predicted == y_dev) 

## Print accuracy of best scoring model.
	print "The accuracy of the best scoring model was " + str(accuracy) + "."

## Saving model using pickle.
	model_filename = 'model.pkl'
	pickle.dump(model, open(model_filename,'wb'))

### Defining pre_process function, same as in hw1.py.
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

### Never got to this part, include it here as sign of effort. Intension
### was to use the emoji library or a custom list like this one.
	common_emojis = [": )","; )",": - )",": - (",": (","; - )"]
	def get_emojis(string):
		return ''.join(char for char in string if char in emoji.UNICODE_EMOJI)
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

if __name__ == "__main__":
    main()
