import math
import os
import re
import time

from offline import w2v_trainer
from online.ShellFusion import detectOpsInTLDRScript, readCmdInfo
from conf import conf
from offline.file_utils import readTxt, readJson, writeJson, load
from offline.mp_analyzer import rankDocsBySimilarityToTarget
from online.question_retriever import readQueries


def generate(queries_txt, embed_topn_dir, QAPairs_det_json, cmd_info_json, kv, idf, k, res_dir):
    """
    Generate answers for queries.
    """
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    queries_dict = readQueries(queries_txt)
    qid_info_dict = readJson(QAPairs_det_json)
    cmd_info_dict, cmd_mid_desc_dict = readCmdInfo(cmd_info_json)
    total_time = 0

    for name in os.listdir(embed_topn_dir):

        query_id = name[:name.rfind('.')]
        p_query = queries_dict[query_id]['P-Query']
        qid_sim_dict, cmd_qids_dict, mid_desc_dict = {}, {}, {}
        start = time.time()

        for line in readTxt(embed_topn_dir + '/' + name):
            sa = line.split(' ===> ')
            if len(sa) == 3:
                qid = sa[0]
                if qid in qid_info_dict:
                    qid_sim_dict[qid] = float(sa[2].strip())
                    accans = qid_info_dict[qid]['AcceptedAnswer']
                    candi_cmds = accans['ShellFusion Command-Options']
                    for cmd in candi_cmds:
                        if cmd in cmd_mid_desc_dict:
                            for mid in cmd_mid_desc_dict[cmd]:
                                mid_desc_dict[mid] = cmd_mid_desc_dict[cmd][mid]
                            if cmd not in cmd_qids_dict:
                                cmd_qids_dict[cmd] = set()
                            cmd_qids_dict[cmd].add(qid)

        """ Measure the similarities between detected commands and the query """
        mid_sim_dict, _dict = {}, {}
        for mid in mid_desc_dict:
            tldr_desc = mid_desc_dict[mid]['TLDR']
            if tldr_desc != '':
                _dict[mid] = tldr_desc
        mid_sim_dict = rankDocsBySimilarityToTarget(_dict, p_query, kv, idf, False)

        """ Filter irrelevant commands and rank the retained top-k commands """
        cmd_mid_dict, cmd_likelihood_dict = {}, {}
        for item in sorted(mid_sim_dict.items(), key=lambda x:x[1], reverse=True):
            mid, simdoc = item[0], item[1]
            cmd = mid[mid.find('_')+1:mid.rfind('_')]
            if cmd not in cmd_likelihood_dict:
                cmd_mid_dict[cmd], n = mid, len(cmd_qids_dict[cmd])
                simso = sum([ qid_sim_dict[qid] for qid in cmd_qids_dict[cmd] ]) / n * math.log2(1+n)
                simso = min(simso, 1.0)
                cmd_likelihood_dict[cmd] = 2 * simso * simdoc / (simso + simdoc) \
                    if simso + simdoc > 0.0 else 0.0
                if len(cmd_likelihood_dict) == k:
                    break

        # generate answer for each candidate cmd
        generated_answers = []  # generated answer for each candidate cmd
        sorted_qids = [ item[0] for item in sorted(qid_sim_dict.items(), key=lambda x:x[1], reverse=True) ]
        for item in sorted(cmd_likelihood_dict.items(), key=lambda x:x[1], reverse=True)[:5]:

            cmd = item[0]
            mid = cmd_mid_dict[cmd]
            mid_dict = cmd_info_dict[cmd][mid]
            """ 1. TLDR Summary """
            tldrsumm = mid_dict['TLDR Summary'] \
                if 'TLDR Summary' in mid_dict else ''

            """ 2. Most Similar TLDR Task-Script Pair """
            mostsim_task, mostsim_script, tldr_ops = '', '', set()
            if 'TLDR Task-Script' in mid_dict:
                task_script_dict, id_task_dict = mid_dict['TLDR Task-Script'], {}
                for i, task in enumerate(task_script_dict.keys()):
                    id_task_dict[i] = task
                id_sim_dict = rankDocsBySimilarityToTarget(id_task_dict, p_query, kv, idf, True)
                mostsim_id = sorted(id_sim_dict.items(), key=lambda x:x[1], reverse=True)[0][0]
                mostsim_task = id_task_dict[mostsim_id]
                mostsim_script = task_script_dict[mostsim_task]
                tldr_ops = detectOpsInTLDRScript(mostsim_script)

            """ 3. Top-3 Similar Questions with Accepted Scripts """
            top3_qtitles, top3_scripts, top3_ops, top3_abodies = [], [], set(), []
            for qid in sorted_qids:
                if qid in cmd_qids_dict[cmd]:
                    if len(top3_qtitles) < 3:
                        top3_qtitles.append(qid + ': ' + qid_info_dict[qid]['Title'])
                    if len(top3_scripts) < 3:
                        accans, scripts, ops = qid_info_dict[qid]['AcceptedAnswer'], [], set()
                        ind_scriptcmdsops_dict = accans['Command-Options in Scripts']
                        for _item in sorted(ind_scriptcmdsops_dict.items(), key=lambda x:int(x[0])):
                            script, cmd_ops_dict = _item[1]['Script'][2:], _item[1]['ShellFusion Command-Options']
                            if cmd != script and cmd in cmd_ops_dict and len(script.split('\n')) <= 10:
                                scripts.append(script)
                                ops = set(cmd_ops_dict[cmd].split())
                        scripts = '\n\n'.join(scripts).replace('&amp;', '&').replace('&gt;', '>').\
                            replace('&lt;', '<').replace('&quot;', '"').replace('&nbsp;', ' ').strip('\n ')
                        if scripts != '':
                            top3_scripts.append(qid + ': ' + scripts)
                            top3_abodies.append(accans['C-Body'])
                            top3_ops |= ops

            """ 4. Explanations about Options """
            top3_ops |= tldr_ops
            op_cdesc_dict = {}
            for abody in top3_abodies:
                for sen in re.split('\. +[a-zA-Z]', abody):
                    if not abody.startswith(sen):
                        sen = abody[abody.find(sen)-1] + sen
                    sen = sen.rstrip('. ')
                    if sen != '' and not sen.endswith(':'):
                        matched_ops = set(sen.split()).intersection(top3_ops)
                        if len(matched_ops) > 0:
                            op = sorted(matched_ops)[0]
                            if op not in op_cdesc_dict:
                                op_cdesc_dict[op] = set()
                            op_cdesc_dict[op].add(sen + '.')

            top3_op_desc_dict = {}
            for op in op_cdesc_dict:
                top3_op_desc_dict[op + '(C)'] = ' '.join(op_cdesc_dict[op])

            generated_answers.append({
                'Command': cmd, 'TLDR Summary': tldrsumm,
                'Most Similar TLDR Task': mostsim_task, 'Most Similar TLDR Script': mostsim_script,
                'Top-3 Similar Questions': top3_qtitles, 'Top-3 Scripts': top3_scripts,
                'Explanations about Options': top3_op_desc_dict
            })

        total_time += time.time() - start
        writeJson({ 'Query': queries_dict[query_id]['Query'], 'Answers': generated_answers },
                  res_dir + '/' + query_id + '.json')

    return total_time


if __name__ == '__main__':

    _icse2022_dir = conf.exp_evaluation_dir + '/icse_2022'
    _kv = w2v_trainer.loadKV(conf.exp_models_dir + '/w2v.kv')
    _idf = load(conf.exp_models_dir + '/token_idf.dump')
    _mpcmd_info_json = conf.exp_manual_dir + '/mpcmd_info.json'
    _embed_topn_dir = conf.exp_evaluation_dir + '/embed_topn'
    _QAPairs_det_json = conf.exp_posts_dir + '/QAPairs_det.json'
    _queries_txt = conf.exp_evaluation_dir + '/queries.txt'
    _genans_dir = _icse2022_dir + '/ShellFusion-QA+TLDR'

    _time = generate(_queries_txt, _embed_topn_dir, _QAPairs_det_json, _mpcmd_info_json, _kv, _idf, 5, _genans_dir)
    print(_time, 's')  # 11.930s for 434 queries
