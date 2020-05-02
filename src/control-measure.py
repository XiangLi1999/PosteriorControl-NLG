import sys
import os
import ast
import torch
from utils_data import get_wikibio_poswrds, get_e2e_poswrds
import json
from collections import defaultdict, Counter
import numpy as np
import string


def process_full_beam(model_path, data_path, beam_path, out_path, corpus_type='valid'):
    saved_stuff = torch.load(model_path)
    temp_name = corpus_type
    data_src = os.path.join(data_path, "src_{}.txt".format(temp_name))
    data_tgt = os.path.join(data_path, "{}.txt".format(temp_name))
    lst_tgt, lst_src = get_sents(data_tgt, data_src)
    saved_dict = saved_stuff["dict"]

    temp_full = []
    for temp in lst_tgt:
        for idx, word in enumerate(temp):
            if word not in saved_dict.word2idx:
                temp[idx] = '<unk>'
        temp_full.append(' '.join(temp))

    vit_lst, ref_lst = read_beam_files(beam_path, ref_only=False)
    combine_beam(ref_lst, temp_full, out_path) # int ordering, ref answer with unk.
    with open(out_path + '_sys', 'w') as f:
        for elem in vit_lst:
            print(elem, file=f)
    return


def process_full_beam3(model_path, data_path, beam_path, out_path, corpus_type='valid'):
    saved_stuff = torch.load(model_path)
    temp_name = corpus_type
    data_src = os.path.join(data_path, "src_{}.txt".format(temp_name))
    data_tgt = os.path.join(data_path, "{}.txt".format(temp_name))
    saved_dict = saved_stuff["dict"]

    vit_lst, ref_lst = read_beam_files3(beam_path)

    temp_full = []
    for temp in ref_lst:
        for idx, word in enumerate(temp):
            if word not in saved_dict.word2idx:
                temp[idx] = '<unk>'
        temp_full.append(' '.join(temp))

    with open(out_path + '_corr', 'w') as f:
        for elem in temp_full:
            print(elem, file=f)
    return

def process_full_beam2(data_path, beam_path, out_path, corpus_type='valid'):
    temp_name = corpus_type
    data_src = os.path.join(data_path, "src_{}.txt".format(temp_name))
    data_tgt = os.path.join(data_path, "{}.txt".format(temp_name))

    lst_tgt, lst_src = get_sents(data_tgt, data_src)
    beam_lst, ref_lst = read_beam_files(beam_path, ref_only=False)

    # the gold reference.
    temp_full = []
    for temp in lst_tgt:
        temp_full.append(' '.join(temp))
    combine_beam(ref_lst, temp_full, out_path)  # int ordering, ref answer with unk.

    # the beam result.
    result_2 = replace_unk(beam_lst ,lst_src, ref_lst)
    with open(out_path+'_sys', 'w') as f:
        for elem in result_2:
            if elem[:6] == '<bos> ':
                elem = elem[6:]
            print(elem, file=f)
    return

def get_wiki_mapj():
    mapj = defaultdict(list)
    mapj_rev = {}
    base_path = '../data/wb_aligned/'
    with open(base_path + 'wb_field', 'rb') as ff:
        field_lst, field_dict = json.load(ff)
    field_dict['PUNCT'] = len(field_dict)
    field_dict['EOS'] = len(field_dict)
    field_dict['UNK'] = len(field_dict)
    field_lst.append('PUNCT')
    field_lst.append('EOS')
    field_lst.append('UNK')

    with open(base_path + 'field60+8.pkl', 'r') as ff:
    # with open(base_path + 'field145+3.pkl', 'r') as ff:
        final_result = json.load(ff)

    print(len(final_result))
    print(len(field_dict))
    print(len(field_lst))


    # print(final_result)
    for elem, val in field_dict.items():
        # print(elem, val, )
        if str(val) not in final_result:
            continue
            # print(elem, val, 'not in')
        else:
            final_z = final_result[str(val)]
            # print(elem, val, final_z)
            mapj_rev[elem] = final_z
            mapj[final_z].append(elem)

    # mapj is a map from the state name to the original field name.

    print(len(mapj))
    print(len(mapj_rev))
    return mapj, mapj_rev

def process_wibi_control(data_path, beam_path, out_path, corpus_type='valid'):
    temp_name = corpus_type
    data_src = os.path.join(data_path, "src_{}.txt".format(temp_name))
    data_tgt = os.path.join(data_path, "{}.txt".format(temp_name))

    lst_tgt, lst_src = get_sents(data_tgt, data_src)
    beam_lst, ref_lst = read_beam_files(beam_path, ref_only=False)

    # the gold reference.
    temp_full = []
    for temp in lst_tgt:
        temp_full.append(' '.join(temp))
    combine_beam(ref_lst, temp_full, out_path)  # int ordering, ref answer with unk.

    # the beam result.
    map_j, map_j_rev = get_wiki_mapj()
    result_2 = replace_control(beam_lst ,lst_src, ref_lst, map_j)
    with open(out_path+'_sys', 'w') as f:
        for elem in result_2:
            if elem[:6] == '<bos> ':
                elem = elem[6:]
            print(elem, file=f)
    return







def replace_unk2(beam_lst, lst_src, int_order):
    result = []
    for idx, num in enumerate(int_order):
        fields = get_wikibio_poswrds(lst_src[num])
        fields = [wrd for (k, idx), wrd in fields.items()]
        result.append(fields)

    result_2 = []
    for ii, (x, y) in enumerate(zip(result, beam_lst)):
        try:
            y, _, _, rank, copy = y.split('|||')
        except:
            continue

        if int(rank) == 0:
            copy = ast.literal_eval(copy)
            y = y.split()
            # print(y)
            for idx, elem in enumerate(y):
                if elem == '<unk>':
                    # print(y[idx], copy[idx])
                    if copy[idx] >= 0 and copy[idx] < len(result[ii]):
                        # print(result[ii], len(result[ii]), copy[idx])
                        y[idx] = result[ii][copy[idx]]
            if '<eos>' in y:
                temp_id = y.index('<eos>')
                y = y[:temp_id+1]
            result_2.append(' '.join(y))
    return result_2


def replace_unk(beam_lst, lst_src, int_order):
    result = []
    for idx, num in enumerate(int_order):
        fields = get_wikibio_poswrds(lst_src[num])
        fields = [wrd for (k, idx), wrd in fields.items()]
        result.append(fields)

    result_2 = []
    x_idx = 0
    for ii in range(len(beam_lst)):
        try:
            x = result[x_idx]
            y = beam_lst[ii]
        except:
            print('x_idx is out of range for x:', x_idx, ii)
        try:
            y, _, states, rank, copy = y.split('|||')
        except:
            continue

        if int(rank) == 0:
            copy = ast.literal_eval(copy)
            y = y.split()
            # print(y)
            for idx, elem in enumerate(y):
                if elem == '<unk>':
                    # print(y[idx], copy[idx])
                    if copy[idx] >= 0 and copy[idx] < len(x):
                        # print(result[ii], len(result[ii]), copy[idx])
                        y[idx] = x[copy[idx]]
            if '<eos>' in y:
                temp_id = y.index('<eos>')
                y = y[:temp_id+1]
            result_2.append(' '.join(y))
            x_idx += 1
    return result_2


def replace_control(beam_lst, lst_src, int_order, map_j):
    # have loaded some pre-defined map from state name to field names.
    total_captured = 0
    result = []
    for idx, num in enumerate(int_order):
        fields = get_wikibio_poswrds(lst_src[num])
        temp_dict = defaultdict(list)
        for (k, idx), wrd in fields.items():
            # print(k, idx, wrd)
            temp_dict[k].append((idx, wrd))
        fields = [wrd for (k, idx), wrd in fields.items()]
        result.append((fields, temp_dict))

    result_2 = []
    x_idx = 0
    score_lst = []
    for ii in range(len(beam_lst)):
        try:
            x, x_dict = result[x_idx]
            y = beam_lst[ii]
        except:
            print('x_idx is out of range for x:', x_idx, ii)
        try:
            y, _, states, rank, copy = y.split('|||')
        except:
            continue

        if int(rank) == 0:
            states = ast.literal_eval(states)
            copy =  ast.literal_eval(copy)
            y = y.split()
            # print(y)
            for idx, elem in enumerate(y):
                if elem == '<unk>':
                    # print(y[idx], copy[idx])
                    if copy[idx] >= 0 and copy[idx] < len(x):
                        # print(result[ii], len(result[ii]), copy[idx])
                        y[idx] = x[copy[idx]]
            if '<eos>' in y:
                temp_id = y.index('<eos>')
                y = y[:temp_id+1]
            filled_y = ' '.join(y)
            result_2.append(filled_y)

            # index by y to get the content form .

            assert len(states) == len(y)
            # write the state in span form.
            states_span = (states)

            for idx_j, (a,b,j) in enumerate(states_span):
                if j not in map_j:
                    continue
                field_names = map_j[j]
                # print(field_names)
                if field_names[0] == 'PUNCT' or field_names[0] == 'EOS':
                    continue
                # print('survive')
                ref = []
                for elem in field_names:
                    ref.append(x_dict[elem])
                pred = y[a:b]
                # print(len(ref))
                # similarity measure between ref and pred.
                # print('pred is ', pred)
                # print('ref is ', ref)
                start_ = []
                for ee in ref:
                    score_ = 0
                    ref_word = [n for (m, n) in ee]
                    # punct_len = 0
                    for word in pred:
                        if word in ref_word:
                            score_ += 1

                    final_score_ = score_ / (len(pred))
                    # print(final_score_)
                    start_.append(final_score_)
                score = np.array(start_).max()
                # print('score is ', score)
                score_lst.append(score)

                total_captured += 1
            x_idx += 1
    print('all the captured things is ', total_captured)
    full_score = np.array(score_lst).mean()
    print('score is ', full_score)
    return result_2



def get_span(lst):
    result = []
    prev = 0
    curr  = 0
    while curr < len(lst):
        if lst[curr] != lst[prev]:
            # put the past result.
            result.append((prev, curr, lst[prev]))
            prev = curr
            curr += 1
        else:
            curr += 1

    result.append((prev, curr, lst[prev]))
    return result




def get_sents(path, src_path):
    lst_tgt, lst_src = [], []
    print(path)
    assert os.path.exists(path)

    with open(src_path, 'r') as f:
        for line in f:
            tokes = line.strip().split()
            lst_src.append(tokes)

    with open(path, 'r') as f:
        for line in f:
            words, spanlabels = line.strip().split('|||')
            words = words.split()
            lst_tgt.append(words)

    return lst_tgt, lst_src

def read_vit_files(file_path):
    vit_lst = []
    with open(file_path, 'r') as f:
        for line in f:
            vit_lst.append(ast.literal_eval(line.strip()))
    print(vit_lst[0])
    return vit_lst

def read_beam_files(file_base_path, ref_only=True):
    vit_path = file_base_path + 'beam'
    corr_path = file_base_path + 'corr'
    src_path = file_base_path + 'src'

    vit_lst = []
    if not ref_only:
        with open(vit_path, 'r') as f:
            for line in f:
                vit_lst.append(line.strip())

    corr_lst = []
    with open(corr_path, 'r') as f:
        for line in f:
            corr_lst.append(int(line.strip()))

    return vit_lst, corr_lst


def read_beam_files3(file_base_path):
    vit_path = file_base_path + 'beam'
    corr_path = file_base_path + 'corr'
    src_path = file_base_path + 'src'

    vit_lst = []
    if True:
        with open(vit_path, 'r') as f:
            for line in f:
                vit_lst.append(line.strip())

    corr_lst = []
    with open(corr_path, 'r') as f:
        for line in f:
            corr_lst.append(line.strip().split())

    return vit_lst, corr_lst

def visual_viterb(viterb, sentence):
    lst = []
    for (start, end, state) in viterb:
        lst.append('[{}'.format(state))
        lst += sentence[start:end]
        lst.append('{}]'.format(state))
    return lst

def combine_vit(vit_result, sent_lst, out_path):
    print(len(vit_result), len(sent_lst))

    seg_result = []
    for idx, (cont, it) in enumerate(vit_result):
        entgt = sent_lst[it]
        seg_result.append(' '.join(visual_viterb(cont, entgt)))

    with open(out_path, 'w') as f:
        for elem in seg_result:
            print(elem, file=f)

    return

def combine_beam(int_order, true_ref, out_path):
    result = []
    for idx, num in enumerate(int_order):
        result.append(true_ref[num])
    with open(out_path + '_ref', 'w') as f:
        for elem in result:
            print(elem, file=f)
    return






if __name__ == '__main__':
    # task = sys.argv[1]
    # assert task in ['beam', 'viterbi']
    # print('task is {}'.format(task))
    # # read from the dataset.
    # dev_path = sys.argv[2]
    # src_path = sys.argv[3]
    # lst_tgt, lst_src = get_sents(dev_path, src_path)
    # print(len(lst_tgt), len(lst_src))
    # viterbi_result = read_vit_files(sys.argv[4])
    # combine_vit(viterbi_result, lst_tgt, sys.argv[5])

    ''' mode 1 -> replacing the reference tokens with UNK '''
    # print('changing the correct reference to contain unk. ')
    # model_path = sys.argv[1] # the model path
    # data_path = sys.argv[2] # the path to data directory.
    # beam_path = sys.argv[3] # the path to beam output base.
    # out_path = sys.argv[4] # the file we want to output to.
    # print('{}\n{}\n{}\n{}'.format(model_path, data_path, beam_path, out_path))
    # process_full_beam3(model_path, data_path, beam_path, out_path, corpus_type='valid')


    ''' mode 2 -> use copy attention idea. '''
    print('chaning our version to largely remove unk')
    data_path = sys.argv[2] # the path to data directory.
    beam_path = sys.argv[3] # the path to beam output base.
    out_path = sys.argv[4] # the file we want to output to.
    print('{}\n{}\n{}'.format(data_path, beam_path, out_path))
    process_wibi_control(data_path, beam_path, out_path, corpus_type='test')
    # process_full_beam2(data_path, beam_path, out_path, corpus_type='test')






    # read from our simplified output files
    # combine them.







