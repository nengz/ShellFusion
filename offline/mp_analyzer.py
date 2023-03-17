import math
import re

from conf import conf
from offline.file_utils import readJson, writeJson
from offline.post_preprocesser import preprocessStr
from offline.similarity import transformDoc, docSySim


def analyzeMPCmds(focal_cmds_json, res_json):
    """
    Analyze the cmds extracted from the MPs of Focal.
    """
    cmds_dict, tr_cmds_dict, man1_cmdnum, man8_cmdnum = \
        readJson(focal_cmds_json), {}, 0, 0

    for cmdname in cmds_dict:
        for relman in cmds_dict[cmdname]:
            _dict = cmds_dict[cmdname][relman]
            summ = _dict['cmd summary']
            if summ != '':
                sec = relman[relman.find('/')+1:]
                paras_dict = _dict['described paras']
                op_desc_dict, op_descs = {}, []
                for p in paras_dict:
                    desc = paras_dict[p]['description']
                    if desc.startswith('-') or desc.startswith(', -'):
                        sa = desc.split('\n')
                        desc = '\n'.join(sa[1:]) if len(sa) > 1 else ''
                        if sa[0].startswith(', -') and len(sa[0]) < 50:
                            p += sa[0]
                    if desc != '' and len(desc) <= 500 and p.startswith('-') and len(p) <=50:
                        for op in identifyParas(p):
                            op_desc_dict[op] =  desc
                        if desc not in op_descs:
                            op_descs.append(desc)

                tr_cmds_dict[sec + '_' + cmdname] = {
                    'Section': sec,
                    'Command': _dict['cmd'],
                    'Summary': summ,
                    'Synopsis': [ item['template'] for item in _dict['cmd templates'] ],
                    'Option-Description': op_desc_dict,
                    'P-Summary': preprocessStr(summ, '2'),
                    'P-Option-Description': preprocessStr(' '.join(op_descs), '2')
                }
                if relman.endswith('man1'):
                    man1_cmdnum += 1
                if relman.endswith('man8'):
                    man8_cmdnum += 1

    print('# cmds in section 1:', man1_cmdnum)  # 44423
    print('# cmds in section 8:', man8_cmdnum)  # 6418
    print('# all cmds in focal:', len(tr_cmds_dict))  # 50841
    writeJson(tr_cmds_dict, res_json)


def identifyParas(paras_str):
    """
    Identify paras from a paras string.
    """
    paras = set()
    paras_str = re.sub('[|/]-', ',-', paras_str)
    for s in paras_str.split(','):
        para = re.split('[ :=\[]', s.strip())[0].strip()
        if para.startswith('-'):
            if '<' in para and '>' not in para:
                para = para[:para.find('<')]
            paras.add(para)
    return paras


def mapMPCmds2TLDRCmds(mpcmds_json, tldrcmds_json, kv, idf, res_json):
    """
    Map the MP cmds to the TLDR cmds based on their descriptions, DUE TO THE
    FACT: there are duplicate cmds in MP, e.g., echo.
    """
    mpcmds_dict = readJson(mpcmds_json)
    tldrcmds_dict = readJson(tldrcmds_json)

    cmdname_mids_dict = {}
    for mid in mpcmds_dict:
        cmdname = mpcmds_dict[mid]['Command']
        if cmdname not in cmdname_mids_dict:
            cmdname_mids_dict[cmdname] = set()
        cmdname_mids_dict[cmdname].add(mid)

    mpcmd_tldrcmd_dict = {}
    for tid in tldrcmds_dict:
        cmdname, candi_dict = tldrcmds_dict[tid]['Command'], {}
        tldr_desc = tldrcmds_dict[tid]['P-Summary'] + ' ' + tldrcmds_dict[tid]['P-Tasks']
        tldr_ops, task_script_dict = set(), tldrcmds_dict[tid]['Task-Script']
        for task in task_script_dict:
            tldr_ops |= extractOpsInTLDRScript(task_script_dict[task])

        if cmdname in cmdname_mids_dict:
            candi_dict = {}
            for mid in cmdname_mids_dict[cmdname]:
                candi_dict[mid] = mpcmds_dict[mid]['P-Summary'] + ' ' + mpcmds_dict[mid]['P-Option-Description']
            if len(candi_dict) == 1:
                mid = sorted(candi_dict.keys())[0]
            else:
                candi_sim_dict = rankDocsBySimilarityToTarget(candi_dict, tldr_desc, kv, idf, False)
                for mid in candi_sim_dict:  # consider the options used in TLDR scripts when mapping
                    matched_opnum = len(set(mpcmds_dict[mid]['Option-Description'].keys()).intersection(tldr_ops))
                    candi_sim_dict[mid] *= 1 + math.log2(1 + matched_opnum)
                mid = sorted(candi_sim_dict.items(), key=lambda x:x[1], reverse=True)[0][0]

            mpcmd_tldrcmd_dict[mid] = tid

    print('# mapped cmds between MP and TLDR:', len(mpcmd_tldrcmd_dict))  # 1079
    writeJson(mpcmd_tldrcmd_dict, res_json)


def extractOpsInTLDRScript(script):
    """
    Extract the cmd options used in a TLDR script.
    """
    ops = set()
    for token in script.split():
        if token.startswith('-'):
            ops.add(token)
            if re.match('[a-zA-Z]{2,}$', token[1:]):
                for t in token[1:]:
                    ops.add('-' + t)
    return ops


def rankDocsBySimilarityToTarget(id_doc_dict, target, kv, idf, b):
    """
    Rank a dict of docs (id_doc) by their semantic similarities to a target (e.g., a query).
    """
    id_sim_dict = {}
    matrix, idfv = transformDoc(target, kv, idf)
    if matrix is not None and idfv is not None:
        for _id in id_doc_dict:
            doc = id_doc_dict[_id]
            _doc = doc if not b else preprocessStr(doc, '2').replace('\n', ' ')
            d_matrix, d_idfv = transformDoc(_doc, kv, idf)
            if d_matrix is not None and d_idfv is not None:
                id_sim_dict[_id] = docSySim(matrix, d_matrix, idfv, d_idfv)
    return id_sim_dict


def prepareCmdsOps4Detection(mpcmds_json, tldrcmds_json, mpcmd_tldrcmd_json, res_json):
    """
    Prepare cmds and options for cmd & op detecton.
    """
    cmd_info_dict = {}
    mpcmds_dict, tldrcmds_dict, mpcmd_tldrcmd_dict = \
        readJson(mpcmds_json), readJson(tldrcmds_json), readJson(mpcmd_tldrcmd_json)

    for mid in mpcmds_dict:
        cmdname = mpcmds_dict[mid]['Command']
        op_desc_dict = mpcmds_dict[mid]['Option-Description']
        if cmdname not in cmd_info_dict:
            cmd_info_dict[cmdname] = { 'Options': set() }
        cmd_info_dict[cmdname]['Options'] |= set(op_desc_dict.keys())
        cmd_info_dict[cmdname][mid] = {}
        for s in [ 'Summary', 'P-Summary', 'Option-Description', 'P-Option-Description' ]:
            cmd_info_dict[cmdname][mid][s] = mpcmds_dict[mid][s]
        if mid in mpcmd_tldrcmd_dict:
            tid = mpcmd_tldrcmd_dict[mid]
            for s in [ 'Summary', 'P-Summary', 'Task-Script', 'P-Tasks' ]:
                cmd_info_dict[cmdname][mid]['TLDR ' + s] = tldrcmds_dict[tid][s]

    for cmdname in cmd_info_dict:
        cmd_info_dict[cmdname]['Options'] = sorted(cmd_info_dict[cmdname]['Options'])

    print('# unique cmd names in focal:', len(cmd_info_dict))  # 35086
    writeJson(cmd_info_dict, res_json)


if __name__ == '__main__':

    _focal_cmds_json = conf.exp_manual_dir + '/_parsed/focal_cmds.json'
    _mpcmds_json = conf.exp_manual_dir + '/mpcmds.json'
    _linux_tldrcmds_json = conf.exp_tldr_dir + '/linux_tldrcmds.json'
    _mpcmd_tldrcmd_json = conf.exp_manual_dir + '/mpcmd_tldrcmd.json'
    _mpcmd_info_json = conf.exp_manual_dir + '/mpcmd_info.json'

    # analyzeMPCmds(_focal_cmds_json, _mpcmds_json)

    # _kv = w2v_trainer.loadKV(conf.exp_models_dir + '/w2v.kv')
    # _idf = load(conf.exp_models_dir + '/token_idf.dump')
    # mapMPCmds2TLDRCmds(_mpcmds_json, _tldrcmds_json, _kv, _idf, _mpcmd_tldrcmd_json)
    prepareCmdsOps4Detection(_mpcmds_json, _linux_tldrcmds_json, _mpcmd_tldrcmd_json, _mpcmd_info_json)
