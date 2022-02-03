import re
import time

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from offline.post_preprocesser import porter_stemmer

eng_stopwords = set()
for sw in stopwords.words('english'):
    eng_stopwords.add(sw)


def preprocess(s):
    """
    Preprocess a string, e.g., a query.
    """
    stems, token_stem_dict = [], {}
    s = re.sub('[^a-zA-Z0-9-_+.]', ' ', s.replace('\n', ' '))
    s = re.sub(' +', ' ', s.lower())

    for sen in sent_tokenize(s):

        sen = re.sub('[.?!,;]', ' ', sen)
        for token in word_tokenize(sen):
            if token in eng_stopwords or len(token) > 100:
                continue  # remove stopwords and the tokens with > 100 characters
            if token in token_stem_dict:
                stems.append(token_stem_dict[token])
            else:
                try:
                    stem = porter_stemmer.stem(token)  # stemming
                    if stem is not None and stem != '':
                        token_stem_dict[token] = stem
                        stems.append(stem)
                except Exception as e:
                    print('***** Error in StemToken *****', e, ':', token)

    return ' '.join(stems)


if __name__ == '__main__':

    _query = "Create a single pdf from multiple text, images or pdf files"
    _query = 'How can I prevent SQL injection in PHP?'
    _query = 'display lyrics'

    start = time.time()
    print('Preprocessed query:', preprocess(_query))  # creat singl pdf multipl text imag pdf file
    print(time.time() - start, 's')  # ~0.008s
