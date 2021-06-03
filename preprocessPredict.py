import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from naiveBayes import nb_predict
from SVM import svm_predict

np.random.seed(100)

data = pd.read_csv('emaildata.csv')

print(data.head(5))

# Drop all empty rows
data['text'].dropna(inplace=True)

# Converting to lowercase
data['text'] = [sentence.lower() for sentence in data['text']]

# Tokenization
data['text'] = [word_tokenize(sentence)
                for sentence in data['text']]

# Word Lemmatization
# Adding POS tags for word lemmatizer to be used
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index, sentence in enumerate(data['text']):

    # Declaring empty list to store words that follow the rules
    final_words = []

    word_lemmatized = WordNetLemmatizer()  # Initializing word lemmatizer

    for word, tag in pos_tag(sentence):

        # Check for stopwords (and, the etc)
        if word not in stopwords.words('english') and word.isalpha():
            word_final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
            final_words.append(word_final)

    # Storing final set of preprocessed words for each iteration
    data.loc[index, 'text_final'] = str(final_words)

# Training and Testing
# 70-30 train test split
train_x, test_x, train_y, test_y = model_selection.train_test_split(
    data['text_final'], data['spam'], test_size=0.3)

# Word Vectorization
# Using TF-IDF (Term Frequency Inverse Document Frequency)
tfidf_vect = TfidfVectorizer(max_features=5000)
tfidf_vect.fit(data['text_final'])

train_x_tfidf = tfidf_vect.transform(train_x)
test_x_tfidf = tfidf_vect.transform(test_x)

# View vocabulary
print(tfidf_vect.vocabulary_)

# Training on various ML Algos

# Naive Bayes

predictions_nb = nb_predict(train_x_tfidf, train_y, test_x_tfidf, test_y)
print("Naive Bayes accuracy : ", accuracy_score(predictions_nb, test_y)*100)

# Support Vector Mahchine

predictions_svm = svm_predict(train_x_tfidf, train_y, test_x_tfidf)
print("SVM Accuracy Score : ", accuracy_score(predictions_svm, test_y)*100)
