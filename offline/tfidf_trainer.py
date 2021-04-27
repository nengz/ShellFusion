import codecs
import time
import warnings

from conf import conf
from file_utils import dumpObj

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')

from gensim import corpora, models, similarities
from six import iteritems
from nltk.corpus import stopwords
import numpy as np

eng_stopwords = set()
for sw in stopwords.words('english'):
    eng_stopwords.add(sw)


class MyCorpus(object):

    def __init__(self, docs_file, dictionary):
        self.docs_file = docs_file
        self.dictionary = dictionary

    def __iter__(self):
        for line in codecs.open(self.docs_file, 'r', encoding='utf-8'):
            # one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(line.lower().split())


def train(tfidf_input, saved_dict_file,
          serialized_corpus_file, saved_model_file, saved_index_file):
    """
    Train TFIDF model for a set of docs (one doc per line, tokens separated by whitespace).
    """
    print('Build dict ================================')
    dictionary = corpora.Dictionary(line.lower().split() for
                                    line in codecs.open(tfidf_input, 'r', encoding='utf-8'))
    stop_ids = [dictionary.token2id[w] for w in eng_stopwords if w in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)
    dictionary.compactify() # remove gaps in id sequence
    dictionary.save(saved_dict_file.replace('/', '\\'))

    print('\n\nBuild corpus ==========================')
    # corpus = [dictionary.doc2bow(line.lower().split()) for line in codecs.open(docs_file, 'r', encoding='utf-8')]
    corpus = MyCorpus(tfidf_input, dictionary)  # doesn't load the corpus into memory!
    corpora.MmCorpus.serialize(serialized_corpus_file.replace('/', '\\'), corpus)

    print('\n\nFit TFIDF model =======================')
    model = models.TfidfModel(corpus)  # fit TFIDF model from corpus
    model.save(saved_model_file.replace('/', '\\'))
    corpus_tfidf = model[corpus]  # transform corpus to TFIDF vectors

    print('\n\nPrepare similarity index===============')
    # NOTE: using MyCorpus, SparseMatrixSimilarity should be used instead of MatrixSimilarity
    index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))  # preparation for similaritiy queries
    index.save(saved_index_file.replace('/', '\\'))  # store the index for future similarity query

    return dictionary, corpus, model, index


def loadDictionary(saved_dict_file):
    """
    Load dictionary from a saved file
    :param saved_dict_file: the file that contains a dictionary.
    :return: a dictionary object.
    """
    return corpora.Dictionary.load(saved_dict_file.replace('/', '\\'))


def deserializeCorpus(serialized_corpus_file):
    """
    Deserialized corpus from a serialzied file.
    :param serialized_corpus_file: the file that containes a serialzied corpus.
    :return: a corpus object.
    """
    return corpora.MmCorpus(serialized_corpus_file.replace('/', '\\'))


def loadModel(saved_model_file):
    """
    Load TFIDF model from a saved file.
    :param saved_model_file: the file that contains an TFIDF model.
    :return: a TFIDF model object.
    """
    return models.TfidfModel.load(saved_model_file.replace('/', '\\'))


def loadIndex(saved_index_file):
    """
    Load index from a saved file.
    :param saved_index_file: the file that contains an index.
    :return: an index object.
    """
    return similarities.SparseMatrixSimilarity.load(saved_index_file.replace('/', '\\'))


def buildIDF4docs(docs_txt, token_idf_dump, token_df_dump, token_idf_txt, token_df_txt):
    """
    Build IDF for a set of docs in a file (one doc per line).
    """
    dictionary = corpora.Dictionary(line.lower().split() for
                                    line in codecs.open(docs_txt, 'r', encoding='utf-8'))
    stop_ids = [dictionary.token2id[w] for w in eng_stopwords if w in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)
    dictionary.compactify()
    buildIDFFromDictionary(dictionary, token_idf_dump, token_df_dump, token_idf_txt, token_df_txt)


def buildIDFFromDictionary(dictionary, token_idf_dump, token_df_dump, token_idf_txt, token_df_txt):
    """
    Build IDF from a dictionary object.
    """
    token2idf_dict, token2df_dict = {}, {}
    for token, _id in dictionary.token2id.items():
        df = dictionary.dfs[_id]
        token2idf_dict[token] = np.log2(dictionary.num_docs / df)
        token2df_dict[token] = df

    dumpObj(token_idf_dump, token2idf_dict)
    dumpObj(token_df_dump, token2df_dict)

    if token_idf_txt != '':
        sl = sorted(token2idf_dict.items(), key=lambda x:x[1], reverse=True)
        f = codecs.open(token_idf_txt, 'w', encoding='utf-8')
        for item in sl:
            f.write(item[0] + '\t' + str(item[1]) + '\n')
        f.close()

    if token_df_txt != '':
        sl = sorted(token2df_dict.items(), key=lambda x:x[1], reverse=True)
        f = codecs.open(token_df_txt, 'w', encoding='utf-8')
        for item in sl:
            f.write(item[0] + '\t' + str(item[1]) + '\n')
        f.close()


if __name__ == '__main__':

    start = time.time()
    buildIDF4docs(conf.exp_models_dir + '/tfidf.input',
                  conf.exp_models_dir + '/token_idf.dump',
                  conf.exp_models_dir + '/token_df.dump', '', '')
    print(time.time() - start, 's')  # 62s
