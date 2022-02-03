import math
import os
import re
import time

from offline import w2v_trainer
from conf import conf
from offline.file_utils import readTxt, readJson, writeJson, load
from offline.mp_analyzer import rankDocsBySimilarityToTarget
from online.question_retriever import readQueries


def generate(queries_txt, embed_topn_dir, QAPairs_det_json, cmd_info_json, kv, idf, use_bikercmds, k, res_dir):
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
                    candi_cmds = accans['ShellFusion Command-Options'] if not use_bikercmds \
                        else accans['BIKER Commands'].split()
                    for cmdname in candi_cmds:
                        if cmdname in cmd_mid_desc_dict:
                            for mid in cmd_mid_desc_dict[cmdname]:
                                mid_desc_dict[mid] = cmd_mid_desc_dict[cmdname][mid]
                            if cmdname not in cmd_qids_dict:
                                cmd_qids_dict[cmdname] = set()
                            cmd_qids_dict[cmdname].add(qid)

        """ Measure the similarities between detected commands and the query """
        mid_sim_dict, _dict1, _dict2 = {}, {}, {}
        for mid in mid_desc_dict:
            _dict1[mid] = mid_desc_dict[mid]['MP']
            _dict2[mid] = mid_desc_dict[mid]['TLDR']
        mid_sim_dict1 = rankDocsBySimilarityToTarget(_dict1, p_query, kv, idf, False)
        mid_sim_dict2 = rankDocsBySimilarityToTarget(_dict2, p_query, kv, idf, False)
        for mid in mid_sim_dict1:
            mid_sim_dict[mid] = 0.5 * mid_sim_dict1[mid]
            if mid in mid_sim_dict2:
                mid_sim_dict[mid] += 0.5 * mid_sim_dict2[mid]

        """ Filter irrelevant commands and rank the retained top-k commands """
        cmd_mid_dict, cmd_likelihood_dict = {}, {}
        for item in sorted(mid_sim_dict.items(), key=lambda x:x[1], reverse=True):
            mid, simdoc = item[0], item[1]
            cmdname = mid[mid.find('_')+1:mid.rfind('_')]
            if cmdname not in cmd_likelihood_dict:
                cmd_mid_dict[cmdname], n = mid, len(cmd_qids_dict[cmdname])
                simso = sum([ qid_sim_dict[qid] for qid in cmd_qids_dict[cmdname] ]) / n * math.log2(1+n)
                simso = min(simso, 1.0)
                cmd_likelihood_dict[cmdname] = 2 * simso * simdoc / (simso + simdoc) \
                    if simso + simdoc > 0.0 else 0.0
                if len(cmd_likelihood_dict) == k:
                    break

        # generate answer for each candidate cmd
        generated_answers = []  # generated answer for each candidate cmd
        sorted_qids = [ item[0] for item in sorted(qid_sim_dict.items(), key=lambda x:x[1], reverse=True) ]
        for item in sorted(cmd_likelihood_dict.items(), key=lambda x:x[1], reverse=True)[:5]:

            cmdname = item[0]
            mid = cmd_mid_dict[cmdname]
            mid_dict = cmd_info_dict[cmdname][mid]
            """ 1. MP Summary """
            mpsumm = mid_dict['Summary']

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
                if qid in cmd_qids_dict[cmdname]:
                    if len(top3_qtitles) < 3:
                        top3_qtitles.append(qid + ': ' + qid_info_dict[qid]['Title'])
                    if len(top3_scripts) < 3:
                        accans, scripts, ops = qid_info_dict[qid]['AcceptedAnswer'], [], set()
                        if not use_bikercmds:
                            ind_scriptcmdsops_dict = accans['Command-Options in Scripts']
                            for _item in sorted(ind_scriptcmdsops_dict.items(), key=lambda x:int(x[0])):
                                script, cmd_ops_dict = _item[1]['Script'][2:], _item[1]['ShellFusion Command-Options']
                                if cmdname != script and cmdname in cmd_ops_dict and len(script.split('\n')) <= 10:
                                    scripts.append(script)
                                    ops = set(cmd_ops_dict[cmdname].split())
                        else:
                            ind_script_dict = accans['Scripts']
                            for _item in sorted(ind_script_dict.items(), key=lambda x:int(x[0])):
                                script = _item[1][2:]
                                if cmdname in script and cmdname != script and len(script.split('\n')) <= 10:
                                    scripts.append(script)
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

            op_desc_dict, top3_op_desc_dict = mid_dict['Option-Description'], {}
            for op in top3_ops:
                if op in op_desc_dict:
                    top3_op_desc_dict[op + '(M)'] = op_desc_dict[op]
                if op in op_cdesc_dict:
                    top3_op_desc_dict[op + '(C)'] = ' '.join(op_cdesc_dict[op])

            generated_answers.append({
                'Command': cmdname, 'MP Summary': mpsumm,
                'Most Similar TLDR Task': mostsim_task, 'Most Similar TLDR Script': mostsim_script,
                'Top-3 Similar Questions': top3_qtitles, 'Top-3 Scripts': top3_scripts,
                'Explanations about Options': top3_op_desc_dict
            })

        total_time += time.time() - start
        writeJson({ 'Query': queries_dict[query_id]['Query'], 'Answers': generated_answers },
                  res_dir + '/' + query_id + '.json')

    return total_time


def detectOpsInTLDRScript(script):
    """
    Detect the options of a cmd in a TLDR script.
    """
    ops = set()
    for token in script.split()[1:]:
        if token.startswith('-'):
            ops.add(token)
        if re.match('-[a-zA-Z]{2,}$', token):
            for j in range(1, len(token)):
                ops.add('-' + token[j])
    return ops


def readCmdInfo(cmd_info_json):
    """
    Read MP cmds' information.
    """
    cmd_info_dict, cmd_mid_desc_dict = readJson(cmd_info_json), {}

    for cmd in cmd_info_dict:
        mid_desc_dict, mapped_mid = {}, ''
        for mid in cmd_info_dict[cmd]:
            if mid.startswith('man'):
                mid_dict = cmd_info_dict[cmd][mid]
                mp_desc = ' '.join([mid_dict['P-Summary'], mid_dict['P-Option-Description']])
                mid_desc_dict[mid] = { 'MP': mp_desc, 'TLDR': '' }
                if 'TLDR Summary' in mid_dict:
                    mid_desc_dict[mid]['TLDR'] = ' '.join([mid_dict['TLDR P-Summary'], mid_dict['TLDR P-Tasks']])
                    mapped_mid = mid
        cmd_mid_desc_dict[cmd] = { mapped_mid: mid_desc_dict[mapped_mid] } \
            if mapped_mid != '' else mid_desc_dict

    return cmd_info_dict, cmd_mid_desc_dict


if __name__ == '__main__':

    _icse2022_dir = conf.exp_evaluation_dir + '/icse_2022'
    _kv = w2v_trainer.loadKV(conf.exp_models_dir + '/w2v.kv')
    _idf = load(conf.exp_models_dir + '/token_idf.dump')
    _mpcmd_info_json = conf.exp_manual_dir + '/mpcmd_info.json'
    _embed_topn_dir = conf.exp_evaluation_dir + '/embed_topn'
    _QAPairs_det_json = conf.exp_posts_dir + '/QAPairs_det.json'
    _queries_txt = conf.exp_evaluation_dir + '/queries.txt'
    _genans_dir = _icse2022_dir + '/ShellFusion'

    for _k in [ 1, 3, 5, 10, 15, 20 ]:
        _time = generate(_queries_txt, _embed_topn_dir, _QAPairs_det_json, _mpcmd_info_json,
                         _kv, _idf, False, _k, _genans_dir + '_k' + str(_k))
        print(_time, 's')  # k = 5, 85.577s for 434 queries
