from venv import create
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

### PREPROCESSING ###
# read file
df = pd.read_csv("/Users/Carla/Documents/CODE_local/NLP_txtmsg_project/clean_nus_sms.csv")

# create list of sentences
sentences = [row["Message"] for index, row in df.iterrows()]

# noise removal
sentences_clean = []
for sentence in sentences:
    sentences_clean.append(re.sub(r"[\.\?\!\,\:\;\#\"]", "", str(sentence)))

# tokenization
sentences_tokenized = [word_tokenize(i) for i in sentences_clean]

# lemmatization
lemmatizer = WordNetLemmatizer()
sentences_lemmatized = []

for sentence in sentences_tokenized:
    sentences_lemmatized.append([lemmatizer.lemmatize(token) for token in sentence])

# stopwords removal
stop_words = set(stopwords.words("english"))

clean_texts = []

for sentence in sentences_lemmatized:
    clean_texts.append([token.lower() for token in sentence if token not in stop_words])

### ANALYSIS ###

# split into test & training data (clean sentences)

training_sentences = sentences_clean[:32500].lower()
test_sentences = sentences_clean[32501:].lower()

# split into test & training data (tokenized)

training_text = clean_texts[:32500]
test_text = clean_texts[32501:]

# merge sublists into single lists

all_document = []
for sentence in clean_texts:
    for token in sentence:
        all_document.append(token)

training_document = []
for sentence in clean_texts:
    for token in sentence:
        training_document.append(token)

test_document = []
for sentence in clean_texts:
    for token in sentence:
        test_document.append(token)

# Bag of Words
def text_to_bow(input):
    bow_dictionary = {}
    for token in input:
        if token in bow_dictionary:
            bow_dictionary[token] += 1
        else:
            bow_dictionary[token] = 1
    return bow_dictionary

def create_features_dictionary(input):
    features_dictionary = {}
    index = 0
    for token in input:
        if token not in features_dictionary:
            features_dictionary[token] = index
            index +=1
    return features_dictionary

def text_to_bow_vector(input, features_dictionary):
    bow_vector = len(features_dictionary) * [0]
    for token in input:
        feature_index = features_dictionary[token]
        bow_vector[feature_index] += 1
    return bow_vector

features_dict = create_features_dictionary(training_document)
#print("selfmade:", text_to_bow_vector(test_document[157], features_dict))

total_bow = Counter(all_document)
#print(total_bow.most_common(100))

# tfidf

vectorizer = CountVectorizer()
term_frequencies = vectorizer.fit_transform(training_sentences)
feature_names = vectorizer.get_feature_names_out()

# create pandas DataFrame with term frequencies
df_term_frequencies = pd.DataFrame(term_frequencies.T.todense(), index=feature_names, columns=['Term Frequency'])
print(df_term_frequencies)