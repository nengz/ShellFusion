import math
import os
import time

from conf import conf
from offline.file_utils import readTxt, readJson, writeJson
from online.question_retriever import readQueries


site_rooturl_dict = {
    'so': 'https://stackoverflow.com/questions/',
    'au': 'https://askubuntu.com/questions/',
    'su': 'https://superuser.com/questions/',
    'ul': 'https://unix.stackexchange.com/questions/'
}


def generate(queries_txt, embed_topn_dir, QAPairs_det_json, k, res_dir):
    """
    Generate answers for queries.
    """
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    queries_dict = readQueries(queries_txt)
    qid_info_dict = readJson(QAPairs_det_json)
    total_time = 0

    for name in os.listdir(embed_topn_dir):

        query_id = name[:name.rfind('.')]
        qid_sim_dict, cmd_qids_dict = {}, {}
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
                        if cmd not in cmd_qids_dict:
                            cmd_qids_dict[cmd] = set()
                        cmd_qids_dict[cmd].add(qid)

        """ rank the candidate commands """
        cmd_likelihood_dict = {}
        for cmd in cmd_qids_dict:
            n = len(cmd_qids_dict[cmd])
            simso = sum([ qid_sim_dict[qid] for qid in cmd_qids_dict[cmd] ]) / n * math.log2(1+n)
            cmd_likelihood_dict[cmd] = simso

        # generate answer for each candidate cmd
        generated_answers = []  # generated answer for each candidate cmd
        sorted_qids = [ item[0] for item in sorted(qid_sim_dict.items(), key=lambda x:x[1], reverse=True) ]
        for item in sorted(cmd_likelihood_dict.items(), key=lambda x:x[1], reverse=True)[:k]:

            cmd = item[0]

            """ Top-3 Similar Questions with Accepted Scripts """
            top3_qtitles, top3_scripts = [], []
            for qid in sorted_qids:
                if qid in cmd_qids_dict[cmd]:
                    if len(top3_qtitles) < 3:
                        top3_qtitles.append(qid + ': ' + qid_info_dict[qid]['Title'])
                    if len(top3_scripts) < 3:
                        accans, scripts = qid_info_dict[qid]['AcceptedAnswer'], []
                        ind_scriptcmdsops_dict = accans['Command-Options in Scripts']
                        for _item in sorted(ind_scriptcmdsops_dict.items(), key=lambda x:int(x[0])):
                            script, cmd_ops_dict = _item[1]['Script'][2:], _item[1]['ShellFusion Command-Options']
                            if cmd != script and cmd in cmd_ops_dict and len(script.split('\n')) <= 10:
                                scripts.append(script)
                        scripts = '\n\n'.join(scripts).replace('&amp;', '&').replace('&gt;', '>').\
                            replace('&lt;', '<').replace('&quot;', '"').replace('&nbsp;', ' ').strip('\n ')
                        if scripts != '':
                            top3_scripts.append(qid + ': ' + scripts)

            generated_answers.append({
                'Command': cmd, 'Top-3 Similar Questions': top3_qtitles,
                'Top-3 Scripts': top3_scripts
            })

            # """ Top-3 Similar Questions with Accepted Scripts """
            # top3_questions = []
            # for qid in sorted_qids:
            #     if qid in cmd_qids_dict[cmd]:
            #         if len(top3_questions) < 3:
            #             accans, scripts = qid_info_dict[qid]['AcceptedAnswer'], []
            #             ind_scriptcmdsops_dict = accans['Command-Options in Scripts']
            #             for _item in sorted(ind_scriptcmdsops_dict.items(), key=lambda x:int(x[0])):
            #                 script, cmd_ops_dict = _item[1]['Script'][2:], _item[1]['ShellFusion Command-Options']
            #                 if cmd != script and cmd in cmd_ops_dict and len(script.split('\n')) <= 10:
            #                     scripts.append(script)
            #             scripts = '\n\n'.join(scripts).replace('&amp;', '&').replace('&gt;', '>').\
            #                 replace('&lt;', '<').replace('&quot;', '"').replace('&nbsp;', ' ').strip('\n ')
            #             if scripts != '':
            #                 site, _qid = qid[:2], qid[3:]
            #                 top3_questions.append({
            #                     'Question Id': qid, 'Question Link': site_rooturl_dict[site] + _qid,
            #                     'Title': qid_info_dict[qid]['Title'], 'Scripts': scripts
            #                 })
            #
            # generated_answers.append({
            #     'Command': cmd, 'Top-3 Similar Questions': top3_questions
            # })

        total_time += time.time() - start
        writeJson({ 'Query': queries_dict[query_id]['Query'], 'Answers': generated_answers },
                  res_dir + '/' + query_id + '.json')

    return total_time


if __name__ == '__main__':

    _icse2022_dir = conf.exp_evaluation_dir + '/icse_2022'
    _embed_topn_dir = conf.exp_evaluation_dir + '/embed_topn'
    _QAPairs_det_json = conf.exp_posts_dir + '/QAPairs_det.json'
    _queries_txt = conf.exp_evaluation_dir + '/queries.txt'
    _genans_dir = _icse2022_dir + '/ShellFusion-QA'

    _time = generate(_queries_txt, _embed_topn_dir, _QAPairs_det_json, 5, _genans_dir)
    print(_time, 's')  # 0.350s for 434 queries
