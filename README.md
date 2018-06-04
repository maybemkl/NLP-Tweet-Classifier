# Tweet classifier: Democratic or republican?

### 1. Training the classifier 
This repository contains a classifier identifying tweets as republican or democratic. The classifier was written as part of a class in Natural Language Processing at Columbia University and utilizes the Python library sci-kit learn.  

To train the classifier, you have to run the file hw1.py in the command line, with two added arguments, one for the training data and an other for the development / test data. In this repository the files containing this data are train_newline.txt and dev_newline.txt. These files can also be run in conjunction with the file classifier.py, which contains the most accurate model that I was able to train (Naive Bayes with unigrams).

For preprocessing, common stop words have been removed using the Python library [stop-words](https://pypi.python.org/pypi/stop-words) as well as a list of words that includes url components, some stop words that seemed to stay despite the use of the aforementioned library and other words that came up when looking at the top features of the bi- and trigrams but were unlikely to actually improve accuracy (for instance, "via" and "unfollowers").