import time
import warnings

from conf import conf

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')

from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec, LineSentence


def train(w2v_input, saved_model_file, saved_kv_file):
    """
    Train word2vec model for a corpus of sentences (each sentence one line).
    """
    sentences = LineSentence(w2v_input.replace('/', '\\'))
    model = Word2Vec(sentences, size=200, window=5, min_count=0, workers=4, iter=100)  # default settings
    if saved_model_file != '':
        model.save(saved_model_file.replace('/', '\\'))
    model.wv.save(saved_kv_file.replace('/', '\\'))
    return model


def loadModel(saved_model_file):
    """
    Load word2vec model from a saved file.
    """
    return Word2Vec.load(saved_model_file.replace('/', '\\'))


def loadKV(saved_kv_file):
    """
    Load KeyedVectors from a saved file.
    """
    return KeyedVectors.load(saved_kv_file.replace('/', '\\'))


if __name__ == '__main__':

    start = time.time()
    train(conf.exp_models_dir + '/w2v.input', '', conf.exp_models_dir + '/w2v.kv')
    print(time.time() - start, 's')  # 5560s
