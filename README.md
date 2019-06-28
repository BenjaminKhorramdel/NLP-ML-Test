##########    NLP-ML-Test  ############

Candidate: Benjamin Khorramdel

In this test, 31 articles with limited sentences are collected from Wikipedia and stored in Excel
spreadsheet. The content of articles is about a brief bio for Hollywood actors. The defined theme
for articles is being “action” or “non-action” actors.

This data collection can be done using web scraping with BeautifulSoup library that is a python
library for pulling data out of HTML and XML files. However, since web scraping is an action that
should be carefully done to avoid any destructive effect on the sites being scraped, in this test, a simple
way is used for collecting articles.

Different techniques for cleaning the raw text are used to remove any punctuation, extra words
(stopwords), numbers, and symbols in the text. Also, in the developed python code, different
libraries in Python such as nltk, Scikit-learn, pandas, numpy, and matplotlib are imported to handle
data and build basically diverse types of machine learning techniques (i.e., Naïve Bayes (NB),
Support Vector Machine (SVM), and Random Forest (RF)).

The data is split into train and test sets, and the results of training and testing are shown in the
form of confusion matrix with related accuracy mentioned at the bottom of the matrix (please se
Fig. 1 and Fig. 2). This process is the same for SVM and RF. It should be mentioned that the accuracy
of the train set for NB classifier is 100% and it is somehow unreasonable, but this is because we
have limited number of datasets. If we deal with real world datasets with thousands of articles,
the accuracy of train will also decrease to a reasonable value. The simulation results show that the
performance of NB and SVM is better that RF classifier for our small dataset. However, to promote
the performance of RF, the tuning of RF should be done using a grid search algorithm to optimally
find the number of depths and the number of estimators.

In this test, conventional machine learning technique was used; however, using conventional
machine learning techniques for large datasets is not recommended because the accuracy of the
model cannot be improved from a specific value by adding more datasets. Instead., the process
of training an intelligent model with a huge number of texts can be done with deep learning
models. In this type of training, word2vec should be used to transform each word of the text to a
vector for training a deep neural network. In this case, TensorFlow is one of the best libraries which
can do such process, efficiently.
