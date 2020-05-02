import sys
import os
import ast
import torch
from utils_data import get_wikibio_poswrds, get_e2e_poswrds
import json
from collections import defaultdict, Counter
import numpy as np

def replace_control(beam_lst, lst_src, int_order, map_j):
    map_j_rev = {v[0]:k for k, v in map_j.items()}
    # have loaded some pre-defined map from state name to field names.
    total_captured = 0
    result = []
    for num in range(len(lst_src)):
        # print(lst_src[num])
        fields = get_e2e_poswrds(lst_src[num].split())
        # print(fields)
        temp_dict = defaultdict(list)
        for (k, idx), wrd in fields.items():
            temp_dict[k].append((idx, wrd))
        fields = [wrd for (k, idx), wrd in fields.items()]
        result.append((fields, temp_dict))

    # print(result)
    # print(len(result))

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
            y, _, states = y.split('|||')
        except:
            continue

        if True:
            # print('here')
            # print(len(states), len(y), y)
            states = ast.literal_eval(states)
            # copy =  ast.literal_eval(copy)
            y = y.split()
            # print(y)
            # if '<eos>' in y:
            #     temp_id = y.index('<eos>')
            #     y = y[:temp_id+1]
            filled_y = ' '.join(y)
            result_2.append(filled_y)

            # index by y to get the content form .

            assert len(states) == len(y)
            # write the state in span form.
            states_span = get_span(states)

            ##################################################################################
            # checking for recall score.
            # first gather span information. # currently this is count per phrasal, maybe later could do
            # count per tokens.
            sent_score = []
            labelseq = [c for a, b, c in states_span]
            for key1, val1 in x_dict.items():
                local_total = 0
                local_good = 0
                # print(val1, len(val1))
                temp_val = map_j_rev[key1]
                if temp_val in labelseq:
                    find_idx = labelseq.index(temp_val)
                    temp1 = states_span[find_idx]
                    aa, bb, cc = temp1
                    ##################################
                    # compute phrasal recall.
                    # local_good += 1
                    # local_total += 1
                    ################################

                    ################################
                    # compute token recall.
                    # score_ = 0
                    ref_word = [n for (m, n) in val1]
                    # print(ref_word)
                    pred = y[aa:bb]
                    # print(pred)
                    for word in ref_word:
                        if word in pred:
                            local_good += 1

                    # for word in pred:
                    #     if word in ref_word:
                    #         local_good += 1

                    # local_good += 0
                    local_total += len(val1)
                    ################################
                    # print(temp1)
                elif temp_val == 7:
                    # print('caught')
                    continue
                else:
                    ##################################
                    # compute phrasal recall.
                    # local_total += 1
                    ################################
                    # compute token recall.
                    local_good += 0
                    local_total += len(val1)
                    ################################
                span_score = local_good / local_total
                sent_score.append(span_score)

            sent_score = np.array(sent_score).mean()
                # print(span_score)
            score_lst.append(sent_score)

            #
            # ##################################################################################
            # # checking for recall score.
            # # first gather span information. # currently this is count per phrasal, maybe later could do
            # # count per tokens.
            # local_total = 0
            # local_good = 0
            # labelseq = [c for a,b,c in states_span]
            # for key1, val1 in x_dict.items():
            #     # print(val1, len(val1))
            #     temp_val = map_j_rev[key1]
            #     if temp_val in labelseq:
            #         find_idx = labelseq.index(temp_val)
            #         temp1 = states_span[find_idx]
            #         aa, bb, cc = temp1
            #         ##################################
            #         # compute phrasal recall.
            #         # local_good += 1
            #         # local_total += 1
            #         ################################
            #
            #         ################################
            #         # compute token recall.
            #         # score_ = 0
            #         ref_word = [n for (m, n) in val1]
            #         # print(ref_word)
            #         pred = y[aa:bb]
            #         # print(pred)
            #         for word in ref_word:
            #             if word in pred:
            #                 local_good += 1
            #
            #         # for word in pred:
            #         #     if word in ref_word:
            #         #         local_good += 1
            #
            #         # local_good += 0
            #         local_total += len(val1)
            #         ################################
            #         # print(temp1)
            #     elif temp_val == 7:
            #         # print('caught')
            #         pass
            #     else:
            #         ##################################
            #         # compute phrasal recall.
            #         # local_total += 1
            #         ################################
            #         # compute token recall.
            #         local_good += 0
            #         local_total += len(val1)
            #         ################################
            # score_lst.append(local_good/local_total)
            #

            ##################################################################################




            # # checking for precision score.
            # for idx_j, (a,b,j) in enumerate(states_span):
            #     # print(j, map_j)
            #     # print(x_dict)
            #     if j not in map_j:
            #         continue
            #     field_names = map_j[j]
            #     # print(field_names)
            #     if field_names[0] == 'PUNCT' or field_names[0] == 'EOS':
            #         continue
            #     # print('survive')
            #     ref = []
            #     for elem in field_names:
            #         ref.append(x_dict[elem])
            #     pred = y[a:b]
            #     start_ = []
            #
            #     for ee in ref:
            #         score_ = 0
            #         ref_word = [n for (m, n) in ee]
            #         # punct_len = 0
            #         for word in pred:
            #             if word in ref_word:
            #                 score_ += 1
            #
            #         final_score_ = score_ / (len(pred))
            #         # print(final_score_)
            #         start_.append(final_score_)
            #     score = np.array(start_).max()
            #     # print('score is ', score)
            #     score_lst.append(score)
            #
            #     total_captured += 1
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
            corr_lst.append(line.strip())

    src_lst = []
    with open(src_path, 'r') as f:
        for line in f:
            src_lst.append(line.strip())

    return vit_lst, corr_lst, src_lst

def process_e2e_control(beam_path, out_path, corpus_type='valid'):

    beam_lst, ref_lst, src_lst = read_beam_files(beam_path, ref_only=False)

    # the beam result.
    # e2e_keys = ["name", "eatType", "food", "priceRange", "customerrating", "area", "near"]
    map_j = {0: ['_name'], 1:['_eatType'], 2:['_priceRange'], 3:['_customerrating'], 4:['_near'],
             5:['_food'], 6:['_area'], 7:['_familyFriendly']} # from state name to field name
    result_2 = replace_control(beam_lst ,src_lst, ref_lst, map_j)
    # with open(out_path+'_sys', 'w') as f:
    #     for elem in result_2:
    #         if elem[:6] == '<bos> ':
    #             elem = elem[6:]
    #         print(elem, file=f)
    return


if __name__ == '__main__':
    print('chaning our version to largely remove unk')
    # data_path = sys.argv[2] # the path to data directory.
    beam_path = sys.argv[1] # the path to beam output base.
    out_path = sys.argv[2] # the file we want to output to.
    # print('{}\n{}\n{}'.format(data_path, beam_path, out_path))
    process_e2e_control(beam_path, out_path, corpus_type='test')