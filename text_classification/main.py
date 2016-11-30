from nltk.corpus import movie_reviews
import nltk.probability
import random

nltk.download("movie_reviews")

# Build data structure for the movie reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:1000]


def document_feature(document):
    document_word = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_word)
    return features


