
import pandas as pd
import nltk
import pandas as pd
import numpy as np
import re
import codecs
import seaborn as sns
import matplotlib.pyplot as plt

clean_data =  pd.read_csv("clean_data.csv")


from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

clean_data["tokens"] = clean_data["text"].apply(tokenizer.tokenize)
print(clean_data.head())

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical

all_words = [word for tokens in clean_data["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_data["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))

print(clean_data.head())


# fig = plt.figure(figsize=(10, 10)) 
# plt.xlabel('Sentence length')
# plt.ylabel('Number of sentences')
# plt.hist(sentence_lengths)
# plt.show()


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

X_train,X_test,y_train,y_test = train_test_split(clean_data["text"],clean_data["num_label"], test_size = 0.2, random_state = 10)

vect = TfidfVectorizer()
vect.fit(X_train)

X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)

prediction = dict()



# Performaing LSA on the data

# from sklearn.decomposition import TruncatedSVD
# from sklearn.neighbors import KNeighborsClassifier

# lsa = TruncatedSVD(100)
# lsa.fit(X_train_df)

# X_train_lsa = lsa.transform(X_train_df)
# X_test_lsa = lsa.transform(X_test_df)

# model = LogisticRegression()
# model.fit(X_train_lsa,y_train)



# prediction["LSA"] = model.predict(X_test_lsa)
# from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report

# print(accuracy_score(y_test,prediction["LSA"]))
# print(precision_score(y_test,prediction["LSA"],pos_label=None,average='weighted'))
# conf_mat = confusion_matrix(y_test, prediction['LSA'])
# sns.heatmap(conf_mat,annot=True,fmt="d")
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()


#Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train_df,y_train)

prediction["LogisticRegression"] = model.predict(X_test_df)
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report

print(accuracy_score(y_test,prediction["LogisticRegression"]))
print(precision_score(y_test,prediction["LogisticRegression"],pos_label=None,average='weighted'))
conf_mat = confusion_matrix(y_test, prediction['LogisticRegression'])
sns.heatmap(conf_mat,annot=True,fmt="d")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#Multinomial NB

# from sklearn.naive_bayes import MultinomialNB
# model = MultinomialNB()

# model.fit(X_train_df,y_train)
# print(X_train_df)

# prediction["Multinomial"] = model.predict(X_test_df)
# from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report

# print(accuracy_score(y_test,prediction["Multinomial"]))
# print(precision_score(y_test,prediction["Multinomial"],pos_label=None,average='weighted'))
# conf_mat = confusion_matrix(y_test, prediction['Multinomial'])
# sns.heatmap(conf_mat,annot=True,fmt="d")
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()




def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes

importance = get_most_important_features(vect, model , 10)
def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    
    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    
    fig = plt.figure(figsize=(10, 10))  

    plt.subplot(121)
    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
    plt.title('Irrelevant', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Disaster', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplots_adjust(wspace=0.8)
    plt.show()

top_scores = [a[0] for a in importance[1]['tops']]
top_words = [a[1] for a in importance[1]['tops']]
bottom_scores = [a[0] for a in importance[1]['bottom']]
bottom_words = [a[1] for a in importance[1]['bottom']]

plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")