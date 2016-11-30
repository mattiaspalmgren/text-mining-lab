from math import log
import numpy


def build_vocabulary(document_tokens):
    vocabulary = set()
    for document in document_tokens:
            vocabulary.update([token for token in document])
    return vocabulary


def build_tf_matrix(vocabulary, document_tokens):
    document_term_matrix = []
    for document in document_tokens:
            freq_vector = [document.count(word) for word in vocabulary]
            tf_vector = [freq / max(freq_vector) for freq in freq_vector]
            document_term_matrix.append(tf_vector)
    return document_term_matrix


def n_document_containing(word, document_list):
    document_count = 0
    for document in document_list:
        if document.count(word) > 0:
            document_count += 1
    return document_count


def idf(word, document_list):
    n_documents = len(document_list)
    df = n_document_containing(word, document_list)
    return log(n_documents / df)


def build_document_matrix(document_tokens, document_term_matrix, vocabulary):
    idf_vector = [idf(word, document_tokens) for word in vocabulary]

    document_matrix = []
    for i in range(0, len(document_term_matrix)):
            document_matrix.append(numpy.multiply(document_term_matrix[i], idf_vector))

    return document_matrix

