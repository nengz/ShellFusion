import codecs
import os
import re
import time

from lxml import etree

from conf import conf
from file_utils import readTxt, writeXlsx, readXlsx, writeJson, readJson
from post_preprocesser import preprocessStr, long_code_patterns, short_code_pattern, cleanHtmlTags


def collectShellTags(so_tags_xml, su_tags_xml, res_txt):
    """
    Collect shell-related tags in SO and SU.
    """
    shell_tags = set()
    for tags_xml in [so_tags_xml, su_tags_xml]:
        context = etree.iterparse(tags_xml, events=('end',), tag='row')
        for event, row in context:
            if event == 'end' or row.tag == 'row':
                tagname = row.get('TagName')
                if 'bash' in tagname or 'shell' in tagname or 'linux' in tagname \
                        or 'ubuntu' in tagname or 'unix' in tagname or 'sh' == tagname:
                    shell_tags.add(tagname)
            row.clear()
            while row.getprevious() is not None:
                del row.getparent()[0]
        del context

    print('# shell-related tags:', len(shell_tags))  # 252
    with open(res_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(shell_tags)))


def collectShellQuestions(site_postsxml_dict, shell_tags_txt, res_dir):
    """
    Collect shell-related questions from several Q&A sites.
    """
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    shell_tags = set()
    for line in readTxt(shell_tags_txt):
        if line != '':
            shell_tags.add(line.strip())
    print('# shell-related tags:', len(shell_tags))

    statis_lines = []
    for site in site_postsxml_dict:
        qids, accans_qids, lines = set(), set(), []
        context = etree.iterparse(site_postsxml_dict[site], events=('end',), tag='row')
        for event, row in context:
            if event == 'end' or row.tag == 'row':
                if row.get('PostTypeId') == '1':
                    tags = set(str(row.get('Tags'))[1:-1].split('><'))
                    if site in {'au', 'ul'} or len(shell_tags.intersection(tags)) > 0:
                        qid = row.get('Id')
                        qids.add(qid)
                        accansid = '' if row.get('AcceptedAnswerId') is None else row.get('AcceptedAnswerId')
                        if accansid != '':
                            accans_qids.add(qid)
                        lines.append([
                            qid, accansid, row.get('Title'), row.get('Body'), ', '.join(tags),
                            row.get('Score'), row.get('ViewCount')
                        ])
            row.clear()
            while row.getprevious() is not None:
                del row.getparent()[0]
        del context

        statis_lines.append([site, len(qids), len(accans_qids)])
        writeXlsx(['Id', 'AcceptedAnswerId', 'Title', 'Body', 'Tags', 'Score', 'ViewCount'],
                  lines, res_dir + '/shell_questions(' + site + ').xlsx')
        del lines

    writeXlsx(['Q&A Community', '# Shell-Related Questions', '# Shell-Related Questions with Accepted Answers'],
              statis_lines, res_dir + '/shell_questions_statis.xlsx')


def collectQAPairs(site_postsxml_dict, posts_dir):
    """
    Collect the shell-related question-accepted answer pairs.
    """
    if not os.path.exists(posts_dir):
        os.makedirs(posts_dir)

    for site in site_postsxml_dict:

        QAPairs = {}
        for line in readXlsx(posts_dir + '/shell_questions(' + site + ').xlsx')[1:]:
            qid, accansid = line[0].strip(), line[1].strip()
            if accansid != '':
                QAPairs[qid] = {
                    'Title': line[2], 'Body': line[3], 'Tags': line[4],
                    'AcceptedAnswer': { 'Id': accansid, 'Body': '' }  # some accepted answers are not obtained below
                }
        print('# accepted answers:', len(QAPairs))

        context = etree.iterparse(site_postsxml_dict[site], events=('end',), tag='row')
        for event, row in context:
            if event == 'end' or row.tag == 'row':
                if row.get('PostTypeId') == '2':
                    aid, qid = row.get('Id'), row.get('ParentId')
                    if qid in QAPairs and aid == QAPairs[qid]['AcceptedAnswer']['Id']:
                        QAPairs[qid]['AcceptedAnswer']['Body'] = row.get('Body')
            row.clear()
            while row.getprevious() is not None:
                del row.getparent()[0]
        del context

        writeJson(QAPairs, posts_dir + '/QAPairs(' + site + ').json')
        del QAPairs


def processQAPairs(posts_dir):
    """
    Process the collected shell-related question-accepted answer pairs.
    """
    for name in os.listdir(posts_dir):
        if name.startswith('QAPairs'):
            print(name)
            QAPairs = readJson(posts_dir + '/' + name)
            for qid in QAPairs:
                QAPairs[qid]['P-Title'] = preprocessStr(QAPairs[qid]['Title'], '2')
                QAPairs[qid]['P-Body'] = preprocessStr(QAPairs[qid]['Body'], '1')
                QAPairs[qid]['P-Tags'] = preprocessStr(QAPairs[qid]['Tags'], '2')
                QAPairs[qid]['AcceptedAnswer']['Scripts'] = extractScripts(QAPairs[qid]['AcceptedAnswer']['Body'])
                del QAPairs[qid]['Body']
            writeJson(QAPairs, posts_dir + '/QAPairs(' + name[name.find('(') + 1:name.rfind(')')] + ')_pre.json')
            del QAPairs


def extractScripts(abody):
    """
    Extract short and long code snippets (i.e., shell scripts) from an accepted answer.
    """
    if abody == '':
        return {}

    ind_script_dict = {}
    for lcp in long_code_patterns:
        items = re.finditer(lcp, abody)
        for item in items:
            s = re.sub('</?pre[^>]*>', '', item.group())
            script = re.sub('</?code[^>]*>', '', s).strip('\n ')
            ind_script_dict[abody.find(script)] = 'L:' + script
    items = re.finditer(short_code_pattern, abody)
    for item in items:
        script = re.sub('</?code[^>]*>', '', item.group()).strip('\n ')
        ind = abody.find(script)
        if ind not in ind_script_dict:
            ind_script_dict[ind] = 'S:' + script

    return ind_script_dict


def buildInputs(posts_dir, tfidf_input, w2v_input, lucene_docs_txt):
    """
    Build inputs for two language models and lucene.
    """
    tfidf_f = codecs.open(tfidf_input, 'w', encoding='utf-8')
    w2v_f = codecs.open(w2v_input, 'w', encoding='utf-8')
    lucene_f = codecs.open(lucene_docs_txt, 'w', encoding='utf-8')

    for name in os.listdir(posts_dir):
        if name.endswith('_pre.json'):
            site = name[name.find('(')+1:name.find(')')]
            QAPairs = readJson(posts_dir + '/' + name)
            for qid in QAPairs:
                title, body, tags = QAPairs[qid]['P-Title'], QAPairs[qid]['P-Body'], QAPairs[qid]['P-Tags']
                tb = title + ' ' + tags
                ttb = tb + ' ' + body
                tfidf_f.write(' '.join(ttb.split()) + '\n')
                tfidf_f.flush()
                w2v_f.write('\n'.join([title, body, '']))
                w2v_f.flush()
                lucene_f.write(site + '_' + qid + ' ===> ' + ' '.join(tb.split()) + '\n')
                # lucene_f.write(site + '_' + qid + ' ===> ' + ' '.join(ttb.split()) + '\n')
                lucene_f.flush()
            del QAPairs

    tfidf_f.close()
    w2v_f.close()
    lucene_f.close()


"""
The following program should be performed after completing the steps in '_a2_mp_parser.py'.
Before '_a2_mp_parser.py', we need to build the language models, i.e., word IDF vocabulary and word2vec model, 
and also complete '_a1_tldr_parser.py'.
"""

def detectCmdsOpsInQAPairs(posts_dir, mpcmd_info_json, topn_qids_txt):
    """
    Detect shell commands and options from the code snippets in the accepted answers
    of similar questions retrieved for queries.
    NOTE: we can detect the commands and options for all questions, which is too time-consuming.
    Therefore, we only consider the top-n similar questions retrieved for queries.
    :param posts_dir: the dir that contains the results of shell-related questions and Q&A pairs.
    :param mpcmd_info_json: the json file that contains MP & TLDR information of shell commands.
    :param topn_qids_txt: the txt file that contains the entire set of the top-n questions' ids.
    """
    topn_qids, mpcmd_info_dict, QAPairs_det = \
        set(readTxt(topn_qids_txt)), readJson(mpcmd_info_json), {}

    for name in os.listdir(posts_dir):
        if '_pre' in name:
            site = name[name.find('(')+1:name.rfind(')')]
            QAPairs = readJson(posts_dir + '/' + name)
            for qid in QAPairs:
                s = site + '_' + qid
                if s in topn_qids:
                    print(s)
                    accans = QAPairs[qid]['AcceptedAnswer']
                    ind_scriptcmdsops_dict, biker_cmds, ourcmd_ops_dict = {}, set(), {}

                    for ind, script in accans['Scripts'].items():
                        biker_cmd, cmd_ops_dict = detectCmdsOpsInScript(script, mpcmd_info_dict)
                        if biker_cmd != '' or len(cmd_ops_dict) > 0:
                            ind_scriptcmdsops_dict[ind] = { 'Script': script }
                            if biker_cmd != '':
                                biker_cmds.add(biker_cmd)
                                ind_scriptcmdsops_dict[ind]['BIKER Command'] = biker_cmd
                            if len(cmd_ops_dict) > 0:
                                for cmdname in cmd_ops_dict:
                                    ops = cmd_ops_dict[cmdname]
                                    if cmdname not in ourcmd_ops_dict:
                                        ourcmd_ops_dict[cmdname] = set()
                                    ourcmd_ops_dict[cmdname] |= ops
                                    cmd_ops_dict[cmdname] = ' '.join(sorted(ops))
                                ind_scriptcmdsops_dict[ind]['ShellFusion Command-Options'] = cmd_ops_dict

                    for cmdname in ourcmd_ops_dict:
                        ourcmd_ops_dict[cmdname] = ' '.join(sorted(ourcmd_ops_dict[cmdname]))
                    accans['Command-Options in Scripts'] = ind_scriptcmdsops_dict
                    accans['BIKER Commands'] = ' '.join(sorted(biker_cmds))
                    accans['ShellFusion Command-Options'] = ourcmd_ops_dict
                    accans['C-Body'] = cleanAnswerBody(accans['Body'])
                    QAPairs_det[s] = QAPairs[qid]

    s = 'QAPairs_det.json' if 'BIKER' not in topn_qids_txt else 'QAPairs_det-BIKER.json'
    writeJson(QAPairs_det, posts_dir + '/' + s)


def detectCmdsOpsInScript(script, mpcmd_info_dict):
    """
    Detect cmds and options in a script.
    """
    s, script = script[0], script[2:]
    mpcmds, cmd_ops_dict, biker_cmd = set(mpcmd_info_dict.keys()), {}, ''

    if s == 'S' and script in mpcmds:
        biker_cmd = script

    script = re.sub('[$(|={/]', ' ', script)
    for line in script.split('\n'):
        line = line.strip('\n ')
        if line != '' and not line.startswith('#'):
            sa = line.split()
            candi_cmds = set(sa).intersection(mpcmds)
            if len(candi_cmds) > 0:
                i, candi_cmd, ops = 0, '', set()
                while i < len(sa):
                    token = sa[i]
                    if token in mpcmds:
                        candi_cmd = token
                        ops = set(mpcmd_info_dict[candi_cmd]['Options'])
                        if candi_cmd not in cmd_ops_dict:
                            cmd_ops_dict[candi_cmd] = set()
                    elif candi_cmd != '' and token.startswith('-'):
                        if token in ops:
                            cmd_ops_dict[candi_cmd].add(token)
                        elif re.match('-[a-zA-Z]$', token):
                            cmd_ops_dict[candi_cmd].add(token)
                        elif re.match('-[a-zA-Z]{2,}$', token):
                            for j in range(1, len(token)):
                                cmd_ops_dict[candi_cmd].add('-' + token[j])
                    i += 1

    return biker_cmd, cmd_ops_dict


def cleanAnswerBody(abody):
    """
    Clean the body of an answer. The cleaned bodies will be used for extracting
    explanations about cmd's options, see '_b2_answer_generator.py'.
    """
    for lcp in long_code_patterns:
        items = re.finditer(lcp, abody)
        for item in items:
            abody = abody.replace(item.group(), '')
    items = re.finditer(short_code_pattern, abody)
    for item in items:
        s = item.group()
        script = re.sub('</?code[^>]*>', '', s).strip('\n ')
        abody = abody.replace(s, '\"' + script + '\"')
    abody = re.sub(' +', ' ', abody.replace('\n', ' '))
    abody = abody.replace('</p>', ' . </p>')
    cleaned_abody = cleanHtmlTags(abody)
    if cleaned_abody is None:
        return ''
    return cleaned_abody.replace('. (', '. ')


if __name__ == '__main__':

    if not os.path.exists(conf.exp_posts_dir):
        os.makedirs(conf.exp_posts_dir)
    if not os.path.exists(conf.exp_models_dir):
        os.makedirs(conf.exp_models_dir)

    _so_tags_xml = conf.so_dir + '/Tags.xml'
    _su_tags_xml = conf.superuser_dir + '/Tags.xml'
    _shell_tags_txt = conf.exp_posts_dir + '/shell_tags.txt'
    _tfidf_input = conf.exp_models_dir + '/tfidf.input'
    _w2v_input = conf.exp_models_dir + '/w2v.input'
    _lucene_docs_txt = conf.exp_models_dir + '/lucene_docs.txt'
    _mpcmd_info_json = conf.exp_manual_dir + '/mpcmd_info.json'
    _topn_qids_txt = conf.exp_evaluation_dir + '/topn_qids.txt'
    _topn_qids_BIKER_txt = conf.exp_evaluation_dir + '/topn_qids-BIKER.txt'

    _site_postsxml_dict = {
        'so': conf.so_dir + '/Posts.xml',
        'su': conf.superuser_dir + '/Posts.xml',
        'au': conf.askubuntu_dir + '/Posts.xml',
        'ul': conf.unixlinux_dir + '/Posts.xml'
    }

    start = time.time()
    collectShellTags(_so_tags_xml, _su_tags_xml, _shell_tags_txt)
    collectShellQuestions(_site_postsxml_dict, _shell_tags_txt, conf.exp_posts_dir)
    collectQAPairs(_site_postsxml_dict, conf.exp_posts_dir)
    processQAPairs(conf.exp_posts_dir)  # 1659s
    buildInputs(conf.exp_posts_dir, _tfidf_input, _w2v_input, _lucene_docs_txt)  # 54s

    # TODO: run the following steps after performing 'tldr_analyzer.py' and 'data_preparer.py'
    detectCmdsOpsInQAPairs(conf.exp_posts_dir, _mpcmd_info_json, _topn_qids_txt)  # 514.630s for 18645 posts
    detectCmdsOpsInQAPairs(conf.exp_posts_dir, _mpcmd_info_json, _topn_qids_BIKER_txt)  # 508.687s for 18562 posts
    print(time.time() - start, 's')
