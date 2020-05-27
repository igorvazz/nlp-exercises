import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from math import log, sqrt
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn import preprocessing

#funcao para plotar wordcloud
def plot_wordcloud(dataframe):
    spam_words = ' '.join(list(
        dataframe[dataframe['LABEL'] == 'blocked']['SMS']))
    spam_wc = WordCloud(width=512,
                        height=512,
                        background_color='lightgrey',
                        colormap='viridis').generate(spam_words)
    plt.figure(figsize=(12, 8), facecolor='g')
    plt.imshow(spam_wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    

#funcao para plotar histogramas sobrepostos das mensagens de sms e spam
def plot_histogram(ok_messages, spam_messages, range_to_plot, xlabel):
    pal = sns.color_palette()
    plt.figure(figsize=(8, 5))
    plt.hist(ok_messages,
             bins=100,
             range=range_to_plot,
             color=pal[0],
             density=True,
             label='ok')
    plt.hist(spam_messages,
             bins=100,
             range=range_to_plot,
             color=pal[3],
             density=True,
             alpha=0.5,
             label='spam')
    plt.title('Histograma normalizado',
              fontsize=15)
    plt.legend()
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Probabilidade', fontsize=15)

    print('# Estatisticas para {}'.format(xlabel))
    print(
        'mean-ok  {:.2f}   mean-spam {:.2f} \nstd-ok   {:.2f}   std-spam   {:.2f} \nmin-ok    {:.2f}   min-spam    {:.2f} \nmax-ok  {:.2f}   max-spam  {:.2f}'
        .format(ok_messages.mean(), spam_messages.mean(), ok_messages.std(), spam_messages.std(),
                ok_messages.min(), spam_messages.min(), ok_messages.max(), spam_messages.max()))
    
#contar quantos caracteres sao em caixa alta num string usando regex
def upper_case_count(text):
    return len(re.findall(r'[A-Z]',text))

#contar numero de caracteres que sao digitos
def number_digit_characters(text):
    return sum(character.isdigit() for character in text)

#funcao para treinar os classificadores
def train_classifier(clf, X_train, y_train):    
    clf.fit(X_train, y_train)
    
#funcao para predicao dos classificadores
def predict_labels(clf, features):
    return (clf.predict(features))

#funcao para plotar uma confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    #fig, ax = plt.subplots(figsize=(4,4))
    ax = sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  
                cmap="Blues", square=True, cbar=False)
    ax.set_ylim(2, 0.1)
    plt.ylabel('true label')
    plt.xlabel('predicted label')    
    
    
#funcao para plotar uma grid de confusion matrixes
def plot_all_confusion_matrices(y_true, dict_all_pred, str_title):
    
    list_classifiers = list(dict_all_pred.keys())
    plt.figure(figsize=(10,10))
    plt.suptitle(str_title, fontsize=20, fontweight='bold')
    n=331

    for clf in list_classifiers : 
        plt.subplot(n)
        plot_confusion_matrix(y_true, dict_all_pred[clf][0])
        plt.title(clf, fontweight='bold')
        n+=1

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
#pipeline de processamento de uma nova mensagem para ser passado no classificador treinado
def processing_pipeline(validation_data, scaler, vectorizer, clf):
    validation_data['TAMANHO'] = validation_data['SMS'].apply(len)
    validation_data['UPPER_CASE'] = validation_data['SMS'].apply(
        upper_case_count)
    validation_data['NUM_DIGIT_CHARACTERS'] = validation_data['SMS'].apply(
        number_digit_characters)
    validation_data[['TAMANHO', 'UPPER_CASE', 'NUM_DIGIT_CHARACTERS'
                     ]] = scaler.transform(validation_data[[
                         'TAMANHO', 'UPPER_CASE', 'NUM_DIGIT_CHARACTERS'
                     ]])
    validation_data_SMS_TFIDF = vectorizer.transform(validation_data['SMS'])

    features = validation_data.loc[:, [
        'TAMANHO', 'UPPER_CASE', 'NUM_DIGIT_CHARACTERS'
    ]].values
    X_validation = np.hstack((validation_data_SMS_TFIDF.todense(), features))
    
    prediction = clf.predict(X_validation)
    validation_data['LABEL'] = prediction
    validation_data = validation_data.drop(
    ['TAMANHO', 'UPPER_CASE', 'NUM_DIGIT_CHARACTERS'], axis=1)
    validation_data['LABEL'] = validation_data['LABEL'].map({1: 'blocked', 0: 'ok'})
    
    return validation_data
