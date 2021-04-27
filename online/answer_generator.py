import math
import re
import time

import w2v_trainer
from conf import conf
from file_utils import readTxt, readJson, writeJson, load
from mp_analyzer import rankDocsBySimilarityToTarget


def generate(pquery, embed_topn_txt, QAPairs_det_json, cmd_info_json, kv, idf, sf_type, k, res_json):
    """
    Generate answers for queries.
    """
    qid_info_dict = readJson(QAPairs_det_json)
    cmd_info_dict, cmd_mid_desc_dict = readCmdInfo(cmd_info_json)
    qid_sim_dict, cmd_qids_dict, mid_desc_dict = {}, {}, {}

    start = time.time()

    for line in readTxt(embed_topn_txt):
        sa = line.split(' ===> ')
        if len(sa) == 3:
            qid = sa[0]
            if qid in qid_info_dict:
                qid_sim_dict[qid] = float(sa[2].strip())
                accans = qid_info_dict[qid]['AcceptedAnswer']
                candi_cmds = accans['ShellFusion Command-Options']
                for cmdname in candi_cmds:
                    if cmdname in cmd_mid_desc_dict:
                        # A command may have multiple MPs in Ubuntu, e.g., ls and echo,
                        # We label the command extract from each MP by an unique mid: man_cmdname_no, e.g.,
                        # man1_ls_18038, man1_echo_8389, etc.
                        for mid in cmd_mid_desc_dict[cmdname]:
                            mid_desc_dict[mid] = cmd_mid_desc_dict[cmdname][mid]
                        if cmdname not in cmd_qids_dict:
                            cmd_qids_dict[cmdname] = set()
                        cmd_qids_dict[cmdname].add(qid)

    """ Measure the similarities between detected commands and the query """
    mid_sim_dict = {}
    if sf_type == 'mps':  # ShellFusion-MPs
        mp_dict = {}
        for mid in mid_desc_dict:
            mp_dict[mid] = mid_desc_dict[mid]['MP']
        mid_sim_dict = rankDocsBySimilarityToTarget(mp_dict, pquery, kv, idf, False)
    elif sf_type == 'tldr':  # ShellFusion-TLDR
        tldr_dict = {}
        for mid in mid_desc_dict:
            tldr_desc = mid_desc_dict[mid]['TLDR']
            if tldr_desc != '':
                tldr_dict[mid] = tldr_desc
        mid_sim_dict = rankDocsBySimilarityToTarget(tldr_dict, pquery, kv, idf, False)
    else:  # ShellFusion
        mp_dict, tldr_dict = {}, {}
        for mid in mid_desc_dict:
            mp_dict[mid] = mid_desc_dict[mid]['MP']
            tldr_dict[mid] = mid_desc_dict[mid]['TLDR']
        mid_mpsim_dict = rankDocsBySimilarityToTarget(mp_dict, pquery, kv, idf, False)
        mid_tldrsim_dict = rankDocsBySimilarityToTarget(tldr_dict, pquery, kv, idf, False)
        for mid in mid_mpsim_dict:
            mid_sim_dict[mid] = 0.5 * mid_mpsim_dict[mid]
            if mid in mid_tldrsim_dict:
                mid_sim_dict[mid] += 0.5 * mid_tldrsim_dict[mid]

    """ Filter irrelevant commands and rank the retained top-k commands """
    cmd_mid_dict, cmd_recscore_dict = {}, {}
    for item in sorted(mid_sim_dict.items(), key=lambda x:x[1], reverse=True):
        mid, sim = item[0], item[1]
        cmdname = mid[mid.find('_')+1:mid.rfind('_')]
        if cmdname not in cmd_recscore_dict:
            cmd_mid_dict[cmdname], n = mid, len(cmd_qids_dict[cmdname])
            likelihood = sum([ qid_sim_dict[qid] for qid in cmd_qids_dict[cmdname] ]) / n * math.log2(1+n)
            likelihood = min(likelihood, 1.0)
            cmd_recscore_dict[cmdname] = 2 * likelihood * sim / (likelihood + sim) \
                if likelihood + sim > 0.0 else 0.0
            if len(cmd_recscore_dict) == k:
                break

    generated_answers = []  # generated answers
    sorted_qids = [ item[0] for item in sorted(qid_sim_dict.items(), key=lambda x:x[1], reverse=True) ]
    for item in sorted(cmd_recscore_dict.items(), key=lambda x:x[1], reverse=True)[:5]:

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
            id_sim_dict = rankDocsBySimilarityToTarget(id_task_dict, pquery, kv, idf, True)
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
                    ind_scriptcmdsops_dict = accans['Command-Options in Scripts']
                    for _item in sorted(ind_scriptcmdsops_dict.items(), key=lambda x:int(x[0])):
                        script, cmd_ops_dict = _item[1]['Script'][2:], _item[1]['ShellFusion Command-Options']
                        if cmdname != script and cmdname in cmd_ops_dict and len(script.split('\n')) <= 10:
                            scripts.append(script)
                            ops = set(cmd_ops_dict[cmdname].split())
                    scripts = '\n\n'.join(scripts).replace('&amp;', '&').replace('&gt;', '>').\
                        replace('&lt;', '<').replace('&quot;', '"').replace('&nbsp;', ' ').strip('\n ')
                    if scripts != '':
                        top3_scripts.append(qid + ': ' + scripts)
                        top3_abodies.append(accans['C-Body'])
                        top3_ops |= ops

        """ 4. Crowdsourcing Explanations about Options """
        top3_ops |= tldr_ops
        op_cdesc_dict = {}
        for abody in top3_abodies:
            for sen in re.split('\. +[a-zA-Z]', abody):
                sen = sen.strip('. ')
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
                top3_op_desc_dict[op] = { 'MP': op_desc_dict[op] }
            if op in op_cdesc_dict:
                s = ' '.join(op_cdesc_dict[op])
                if op not in top3_op_desc_dict:
                    top3_op_desc_dict[op] = { 'Crowdsourcing': s }
                else:
                    top3_op_desc_dict[op]['Crowdsourcing'] = s

        generated_answers.append({
            'Command': cmdname, 'MP Summary': mpsumm,
            'Most Similar TLDR Task-Script Pair': { mostsim_task: mostsim_script },
            'Top-3 Similar Questions': top3_qtitles, 'Top-3 Scripts': top3_scripts,
            'Explanations about Options': top3_op_desc_dict
        })

    print(time.time() - start, 's')
    writeJson(generated_answers, res_json)


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

    for cmdname in cmd_info_dict:
        mid_desc_dict, mapped_mid = {}, ''
        for mid in cmd_info_dict[cmdname]:
            if mid.startswith('man'):
                mid_dict = cmd_info_dict[cmdname][mid]
                mp_desc = ' '.join([mid_dict['P-Summary'], mid_dict['P-Option-Description']])
                mid_desc_dict[mid] = { 'MP': mp_desc, 'TLDR': '' }
                if 'TLDR Summary' in mid_dict:
                    mid_desc_dict[mid]['TLDR'] = ' '.join([mid_dict['TLDR P-Summary'], mid_dict['TLDR P-Tasks']])
                    mapped_mid = mid
        cmd_mid_desc_dict[cmdname] = { mapped_mid: mid_desc_dict[mapped_mid] } \
            if mapped_mid != '' else mid_desc_dict

    return cmd_info_dict, cmd_mid_desc_dict


if __name__ == '__main__':

    _pquery = "creat singl pdf multipl text imag pdf file"
    _kv = w2v_trainer.loadKV(conf.exp_models_dir + '/w2v.kv')
    _idf = load(conf.exp_models_dir + '/token_idf.dump')
    _mpcmds_info_json = conf.exp_manual_dir + '/mpcmds_info.json'
    _embed_topn_txt = conf.experiment_dir + '/_test/embed_topn.txt'
    _QAPairs_det_json = conf.exp_posts_dir + '/QAPairs_det.json'
    _genans_dir = conf.exp_evaluation_dir + '/ShellFusion'

    generate(_pquery, _embed_topn_txt, _QAPairs_det_json, _mpcmds_info_json, _kv, _idf, '', 5, _genans_dir)
    generate(_pquery, _embed_topn_txt, _QAPairs_det_json, _mpcmds_info_json, _kv, _idf, 'mps', 5, _genans_dir + '-MPs')
    generate(_pquery, _embed_topn_txt, _QAPairs_det_json, _mpcmds_info_json, _kv, _idf, 'tldr', 5, _genans_dir + '-TLDR')
