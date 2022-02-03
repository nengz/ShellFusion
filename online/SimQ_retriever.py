import os
import time

from offline import w2v_trainer
from conf import conf
from offline.file_utils import readTxt, load
from offline.similarity import transformDoc, docSySim


def retrieve(queries_txt, lucene_topN_dir, lucene_docs_txt, kv, idf, n, res_dir):
    """
    Rerank the candidate docs retrieved by lucene for queries using a word embedding based approach.
    """
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    queries_dict = readQueries(queries_txt)
    qid_doc_dict = readTransformLuceneDocs(lucene_docs_txt, kv, idf)

    total_time, topn_qids = 0, set()
    for query_id in queries_dict:
        candi_qids = set()
        topN_txt = lucene_topN_dir + '/' + query_id + '.txt'
        if os.path.exists(topN_txt):
            for line in readTxt(topN_txt):
                sa = line.split('\t')
                if len(sa) == 2:
                    candi_qids.add(sa[0])
        else:
            candi_qids = qid_doc_dict.keys()

        start, qid_sim_dict = time.time(), {}
        matrix, idfv = transformDoc(queries_dict[query_id]['P-Query'], kv, idf)
        if matrix is not None and idfv is not None:
            for qid in candi_qids:
                qid_sim_dict[qid] = docSySim(matrix, qid_doc_dict[qid]['matrix'], idfv, qid_doc_dict[qid]['idf'])
        sl = sorted(qid_sim_dict.items(), key=lambda x:x[1], reverse=True)
        print(query_id, '->', time.time() - start)
        total_time += time.time() - start

        with open(res_dir + '/' + query_id + '.txt', 'w', encoding='utf-8') as f:
            for item in sl[:n]:
                topn_qids.add(item[0])
                f.write(' ===> '.join([item[0], qid_doc_dict[item[0]]['doc'], str(item[1])]) + '\n')
                f.flush()

    return total_time


def readQueries(queries_txt):
    """
    Read queries.
    """
    queries_dict = {}
    lines = readTxt(queries_txt)
    for line in lines:
        sa = line.split(' ===> ')
        if len(sa) == 3:
            queries_dict[sa[0]] = { 'Query': sa[1], 'P-Query': sa[2] }
    return queries_dict


def readTransformLuceneDocs(lucene_docs_txt, kv, idf):
    """
    Read the docs file for lucene and transform them into matrix representation based on language models.
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

    _kv = w2v_trainer.loadKV(conf.exp_models_dir + '/w2v.kv')
    _idf = load(conf.exp_models_dir + '/token_idf.dump')
    _lucene_docs_txt = conf.exp_models_dir + '/lucene_docs.txt'
    _queries_txt = conf.exp_evaluation_dir + '/queries.txt'
    _lucene_topN_dir = conf.exp_evaluation_dir + '/lucene_topN'
    _embed_topn_dir = conf.exp_evaluation_dir + '/embed_topn'

    _time = retrieve(_queries_txt, _lucene_topN_dir, _lucene_docs_txt, _kv, _idf, 50, _embed_topn_dir)
    print(_time, 's')  # 32.391s for 434 queries  ~ 0.075s per query
