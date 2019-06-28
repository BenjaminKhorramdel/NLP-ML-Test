### This Python code receives several articles about Hollywood actors who play action and non-action movies (Selected theme is "action").
### The goal is to build a model that can detect the contents of new unseen articles and predict which articles are for action actors.

import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
######################## read the articles of Hollywood actors########################################
Actors = pd.read_excel("Articles.xlsx", dtype='unicode')
##--------------------------- Labeling ------------------------------------------------------
def targetLable(dataframe):
    dataframe['target'] = 0
    for index, item in dataframe.iterrows():
        # convert text to lowercase
        dataframe.at[index, 'Description'] = item['Description'].strip().lower()
        if item['Description'].rfind('action') != -1:
            dataframe.at[index, 'target'] = 1
    return dataframe
##------------------------ Removing numbers from text ---------------------------------------------------
def removeDigit(list):
    pattern = '[0-9]'
    list = re.sub(pattern, '', list)
    return list
##------- Removing punctuation and stopwords from text and making clean tokens to have a clean text ------
def transformation(input):
    array = []
    for text in input:
        # replace punctuation characters with spaces
        filters = '!"\'#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n–'
        translate_dict = dict((c, "") for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)
        text = removeDigit(text)
        token = [t for t in text.split()]
        token = [w for w in token if len(w) > 2]
        array.append(token)

    Extra = set(stopwords.words('english'))
    # clean_item = []
    cleanFinalToken = []
    for item in array:
        clean_item = item[:]
        for t in item:
            if t in Extra:
                clean_item.remove(t)
        cleanFinalToken.append(' '.join(clean_item))
    return cleanFinalToken
##-------------------------------------------------------------------------------------------------
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.0f}%".format(cm[i, j] * 100), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}'.format(100 * accuracy))
    plt.show()

targetLable(Actors)
Actors['Cleaned_Description'] = transformation(Actors['Description'])
##--------------------  Split data into train and test sets -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(Actors['Cleaned_Description'], Actors['target'], random_state=35)
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)

##----------------------------    Naive Bayes -----------------------------------------------------------------
NBclf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
preds_NB = NBclf.predict(vect.transform(X_test))
tarin_preds_NB = NBclf.predict(X_train_vectorized)
plot_confusion_matrix(cm=confusion_matrix(y_train, tarin_preds_NB), normalize=True, target_names=['NotAction', 'Action'],
                      title="NB Confusion Matrix for Train-set")
plot_confusion_matrix(cm=confusion_matrix(y_test, preds_NB), normalize=True, target_names=['NotAction', 'Action'],
                      title="NB Confusion Matrix for Test-set")

##--------------------------------- Random Forest  ----------------------------------------------------------

RFclf = RandomForestClassifier(max_depth=4, n_estimators=2).fit(X_train_vectorized, y_train)
tarin_preds_RF = RFclf.predict(X_train_vectorized)
preds_RF = RFclf.predict(vect.transform(X_test))
plot_confusion_matrix(cm=confusion_matrix(y_train, tarin_preds_RF), normalize=True, target_names=['NotAction', 'Action'],
                      title="RFC Confusion Matrix for Train-set")
plot_confusion_matrix(cm=confusion_matrix(y_test, preds_RF), normalize=True, target_names=['NotAction', 'Action'],
                      title="RFC Confusion Matrix for Test-set")

##------------------------------------ SVM  ----------------------------------------------------------------

SVMclf = svm.SVC(kernel='linear').fit(X_train_vectorized, y_train)  # Linear Kernel
tarin_preds_SVM = SVMclf.predict(X_train_vectorized)
preds_SVM = SVMclf.predict(vect.transform(X_test))
plot_confusion_matrix(cm=confusion_matrix(y_train, tarin_preds_SVM), normalize=True, target_names=['NotAction', 'Action'],
                      title="SVM Confusion Matrix for Train-set")
plot_confusion_matrix(cm=confusion_matrix(y_test, preds_SVM), normalize=True, target_names=['NotAction', 'Action'],
                      title="SVM Confusion Matrix for Test-set")

def Output(predictedValue, realTest):
    df = pd.DataFrame(columns=realTest.columns)
    for i in range(len(predictedValue)):
        if predictedValue[i] == 1:
            df = df.append(realTest.loc[i, :], ignore_index=True)
    return df
###################################################################################################
########################### Test with New Data  ###################################################

Newdata = pd.read_excel("RealTestArticles.xlsx", dtype='unicode')
print('New unseen data')
print(Newdata)
targetLable(Newdata)
Newdata['Cleaned_Description'] = transformation(Newdata['Description'])

##################################  Outputs ########################################################

##### Prediction using Naive Bayes Classifier  ######################
preds1 = NBclf.predict(vect.transform(Newdata['Cleaned_Description']))

print("Classified New Articles with Naive Bayes Classifier:  (0: Non-Action , 1:Action) ", preds1)
print("Articles Related to 'Action':", Output(preds1, Newdata))
##### Prediction using SVM Classifier  #############################
preds2 = SVMclf.predict(vect.transform(Newdata['Cleaned_Description']))

print("Classified New Articles with SVM:  (0: Non-Action , 1:Action) ", preds2)
print("Articles Related to 'Action':", Output(preds2, Newdata))
######## Prediction using Random Forest Classifier  ################
preds3 = RFclf.predict(vect.transform(Newdata['Cleaned_Description']))

print("Classified New Articles with RF:  (0: Non-Action , 1:Action) ", preds3)
print("Articles Related to 'Action':", Output(preds3, Newdata))
print("Based on the contents of new articles, the first two articles are "
      "for non-action actors (Adam Sandler and Julia Roberts) and the third "
      "one is for an action actor (Ben Affleck). You may tend to test the model with your articles,"
      "so just place your file name instead of RealTestArticles.xlsx in outputs section of the code!")