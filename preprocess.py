from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import re, nltk


def preprocess(description):
    st = LancasterStemmer()
    stop = stopwords.words('english')
    stop.append('')

    # Tokenize
    temp_tokens = (nltk.word_tokenize(description))

    # Remove non alpha-numeric characters and lowercase words
    temp_tokens = [re.sub(r'\W+', '', str(token)).lower() for token in temp_tokens]

    # Remove stopwords and stem tokens
    temp_tokens = [st.stem(str(token)) for token in temp_tokens if token not in stop]

    return temp_tokens
