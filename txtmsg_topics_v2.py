import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from nltk.corpus import stopwords
import re
from collections import Counter


### PREPROCESSING
# load text messages
df = pd.read_csv("/Users/Carla/Documents/CODE_local/NLP_txtmsg_project/clean_nus_sms.csv")
messages = [row["Message"] for index, row in df.iterrows()]

def doc_cleaner(doc, stop_words):
    doc_cleaned = []
    for text in doc:
        text = re.sub(r"[\.\?\!\,\:\;\#\<\>\"\']", "", str(text))
        words = text.split()
        no_stops = []
        for word in words:
            if word.lower() not in stop_words:
                no_stops.append(word.lower())
        no_stops = " ".join(no_stops)
        doc_cleaned.append(no_stops)
        doc_cleaned = doc_cleaned
    return doc_cleaned

annoying_words = ["lol", "haha", "ok", "ur", "hey", "hahaha", "yeah", "yea", "got", "hi"]
stop_words = list(stopwords.words("english")) + annoying_words
messages_cleaned = doc_cleaner(messages, stop_words)
# total_bow = Counter(messages_cleaned)
# print(total_bow.most_common(100))

# Tfidf
vectorizer = TfidfVectorizer(lowercase=True, max_features=100, max_df=0.8, min_df=5, ngram_range=(1,3))

vectors = vectorizer.fit_transform(messages_cleaned)
feature_names = vectorizer.get_feature_names_out()

dense  = vectors.todense()
denselist = dense.tolist()

all_keywords = []

for message in denselist:
    x = 0
    keywords = []
    for word in message:
        if word > 0:
            keywords.append(feature_names[x])
        x += 1
    all_keywords.append(keywords)

# kmeans
true_k = 20

model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)

model.fit(vectors)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

with open ("txtmsg_topics_results.txt", "w", encoding="utf-8") as f:
    for i in range(true_k):
        f.write(f"Cluster {i}")
        f.write("\n")
        for ind in order_centroids[i, :10]:
            f.write(' %s' % terms[ind],)
            f.write("\n")        
    f.write("\n")
    f.write("\n")