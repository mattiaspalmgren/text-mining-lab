from nltk.corpus import movie_reviews
import nltk.probability, nltk.metrics
import random, math

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

N = len(documents)
feature_sets = [(document_feature(d), c) for (d, c) in documents]
train_set, test_set = feature_sets[math.floor(N*0.8):], feature_sets[:math.floor(N*0.8)]

classifier = nltk.NaiveBayesClassifier.train(train_set)

test_set_without_lables = [d for (d, l) in test_set]
test_labels = [l for (d, l) in test_set]
predictions = classifier.classify_many(test_set_without_lables)


