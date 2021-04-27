import os
import time

import w2v_trainer
from conf import conf
from file_utils import readTxt, load
from similarity import transformDoc, docSySim


def retrieve(pquery, lucene_topN_txt, lucene_docs_txt, kv, idf, n, res_txt):
    """
    Retrieve the top-n similar questions from the N questions reduced by Lucene for a query.
    This is the 2nd-phase of our two-phase retrieval method, which leverages a language model-based method.
    """
    qid_doc_dict = readTransformLuceneDocs(lucene_docs_txt, kv, idf)

    candi_qids = set()  # candidate questions' ids for the 2nd-phase retrieval
    if os.path.exists(lucene_topN_txt):
        for line in readTxt(lucene_topN_txt):
            sa = line.split('\t')
            if len(sa) == 2:
                candi_qids.add(sa[0])
    else:
        candi_qids = qid_doc_dict.keys()

    start, qid_sim_dict = time.time(), {}
    matrix, idfv = transformDoc(pquery, kv, idf)
    if matrix is not None and idfv is not None:
        for qid in candi_qids:
            qid_sim_dict[qid] = docSySim(matrix, qid_doc_dict[qid]['matrix'], idfv, qid_doc_dict[qid]['idf'])
    sl = sorted(qid_sim_dict.items(), key=lambda x:x[1], reverse=True)
    print(time.time() - start)  # 32.391s for 434 queries  ~ 0.075s per query

    with open(res_txt, 'w', encoding='utf-8') as f:
        for item in sl[:n]:
            f.write(' ===> '.join([item[0], qid_doc_dict[item[0]]['doc'], str(item[1])]) + '\n')
            f.flush()


def readTransformLuceneDocs(lucene_docs_txt, kv, idf):
    """
    Read the questions' docs used for Lucene indexing, i.e., the whole questions in the respository.
    Transform each question's preprocessed doc into a matrix representation based on the word embedding model
    and a vector representation based on the word IDF vocabulary.
    """
    id_doc_dict = {}
    for line in readTxt(lucene_docs_txt):
        sa = line.split(' ===> ')
        if len(sa) == 2:
            _id, doc = sa[0], sa[1]
            matrix, idfv = transformDoc(doc, kv, idf)
            if matrix is not None and idfv is not None:
                id_doc_dict[_id] = { 'doc': doc, 'matrix': matrix, 'idf': idfv }
    return id_doc_dict


if __name__ == '__main__':

    if not os.path.exists(conf.exp_evaluation_dir):
        os.makedirs(conf.exp_evaluation_dir)

    _pquery = "creat singl pdf multipl text imag pdf file"
    _kv = w2v_trainer.loadKV(conf.exp_models_dir + '/w2v.kv')
    _idf = load(conf.exp_models_dir + '/token_idf.dump')
    _lucene_docs_txt = conf.exp_models_dir + '/lucene_docs.txt'
    _lucene_topN_txt = conf.experiment_dir + '/_test/lucene_topN.txt'
    _embed_topn_txt = conf.experiment_dir + '/_test/embed_topn.txt'

    retrieve(_pquery, _lucene_topN_txt, _lucene_docs_txt, _kv, _idf, 50, _embed_topn_txt)
