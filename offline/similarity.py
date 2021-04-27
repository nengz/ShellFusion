import numpy as np


def transformDoc(doc, kv, idf):
    """
    Transform a preprocessed doc to a matrix and an IDF vector based on
    kv (the word2vec model) and idf (the word IDF vocabulary).
    """
    doc_words = doc.split()
    if len(doc_words) > 0:
        matrix = initMatrix4DocWords(doc_words, kv)
        if matrix is not None:
            idfv = initIDFVector4DocWords(doc_words, idf)
            if idfv.sum() > 0:
                return matrix, idfv
    return None, None


def initMatrix4DocWords(doc_words, kv):
    """
    Initialize a matrix that contains the word embedding vectors of words in a doc.
    """
    matrix = np.zeros((len(doc_words), 200)) # default word embedding size is 200
    for i, word in enumerate(doc_words):
        if word in kv.vocab:
            matrix[i] = np.array(kv[word])

    try: # l2 normalize
        norm = np.linalg.norm(matrix, axis=1).reshape(len(doc_words), 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
    except Exception as e:
        print('***** Error in initDocMatrix() *****', e, ':', doc_words)
        return None
    return matrix


def initIDFVector4DocWords(doc_words, idf):
    """
    Initialize an IDF vector of words in a doc.
    """
    idfv = np.zeros((1, len(doc_words)))
    for i, word in enumerate(doc_words):
        if word in idf:
            idfv[0][i] = idf[word]
    return idfv


def docSySim(matrix1, matrix2, idf1, idf2):
    """
    Calculate the Symmetric similarity between two docs
    based on their matrices and IDF vectors.
    """
    if matrix1 is not None and matrix2 is not None \
            and idf1.sum() > 0 and idf2.sum() > 0:
        sim21 = (idf1 * (matrix1.dot(matrix2.T).max(axis=1))).sum() / idf1.sum()  # asy(2->1)
        sim12 = (idf2 * (matrix2.dot(matrix1.T).max(axis=1))).sum() / idf2.sum()  # asy(1->2)
        if sim12 + sim21 > 0:
            return 2 * sim12 * sim21 / (sim12 + sim21)
    return 0.0
