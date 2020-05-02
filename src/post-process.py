import sys
import os
import ast
import torch
from utils_data import get_wikibio_poswrds, get_e2e_poswrds

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

def process_full_beam_e2e(data_path, beam_path, out_path1, corpus_type='valid'):
    temp_name = corpus_type
    data_src = os.path.join(data_path, "src_{}.txt".format(temp_name))
    data_tgt = os.path.join(data_path, "{}.txt".format(temp_name))

    suggested_path = os.path.split(beam_path)[-1]
    print('suggested')
    print(suggested_path)
    out_path = os.path.join(out_path1, suggested_path)
    lst_tgt, lst_src = get_sents(data_tgt, data_src)
    beam_lst, ref_lst = read_beam_files(beam_path, ref_only=False)

    # the gold reference.
    temp_full = []
    for temp in lst_tgt:
        temp_full.append(' '.join(temp))

    temp_src = []
    for temp in lst_src:
        temp_src.append(' '.join(temp))
    combine_beam_e2e(ref_lst, temp_full, temp_src, out_path)  # int ordering, ref answer with unk.

    # the beam result.
    # result_2 = replace_unk(beam_lst ,lst_src, ref_lst)
    result_2 = replace_unk_e2e(beam_lst ,lst_src, ref_lst)
    with open(out_path+'beam', 'w') as f:
        for elem in result_2:
            if elem[:6] == '<bos> ':
                elem = elem[6:]
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
    result_2 = replace_unk_full(beam_lst ,lst_src, ref_lst)
    # result_2 = replace_unk_e2e(beam_lst ,lst_src, ref_lst)
    with open(out_path+'_sys', 'w') as f:
        for elem in result_2:
            if elem[:6] == '<bos> ':
                elem = elem[6:]
            print(elem, file=f)
    return



def replace_unk_e2e_(beam_lst, lst_src, int_order):
    result = []
    for idx, num in enumerate(int_order):
        fields = get_e2e_poswrds(lst_src[num])
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
                    if copy[idx] >= 0 and copy[idx] < len(x):
                        # print(result[ii], len(result[ii]), copy[idx])
                        y[idx] = x[copy[idx]]
            if '<eos>' in y:
                temp_id = y.index('<eos>')
                y = y[:temp_id]
            result_2.append(' '.join(y))
            x_idx += 1
    return result_2

def replace_unk22(beam_lst, lst_src, int_order):
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
                y = y[:temp_id]
            result_2.append(' '.join(y))
    return result_2

def replace_unk_e2e(beam_lst, lst_src, int_order):
    result = []
    for idx, num in enumerate(int_order):
        fields = get_e2e_poswrds(lst_src[num])
        fields = [wrd for (k, idx), wrd in fields.items()]
        result.append(fields)

    result_2 = []
    x_idx = 0


    temp_store = []
    for ii in range(len(beam_lst)):
        try:
            x = result[x_idx]
            y = beam_lst[ii]
        except:
            print('x_idx is out of range for x:', x_idx, ii)
        try:
            y1, score_1, state_1, rank1, copy1 = y.split('|||')
        except:
            continue



        if int(rank1) == 0:
            # handle the rescore of the previous list:
            if len(temp_store) > 0:
                # sort  from  temp_score
                for score_, elem in sorted(temp_store, key=lambda a: a[0])[:1]:
                    y, score_, state_, rank, copy = elem
                    copy = ast.literal_eval(copy)
                    y = y.split()
                    for idx, elem in enumerate(y):
                        if elem == '<unk>':
                            # print(y[idx], copy[idx])
                            if copy[idx] >= 0 and copy[idx] < len(x):
                                # print(result[ii], len(result[ii]), copy[idx])
                                y[idx] = x[copy[idx]]
                    if '<eos>' in y:
                        temp_id = y.index('<eos>')
                        y = y[:temp_id]
                    result_2.append(' '.join(y))
                x_idx += 1
            temp_store = []
            rescore = ((5 + len(y1.split())) / 6)**0.5
            score_ = float(score_1)
            temp_store.append((score_ * rescore, (y1, score_1, state_1, rank1, copy1)))

        else:
            rescore = ((5 + len(y1.split())) / 6)**0.5
            score_ = float(score_1)
            temp_store.append((score_ * rescore, (y1, score_1, state_1, rank1, copy1)))

    return result_2


def replace_unk(beam_lst, lst_src, int_order):
    result = []
    for idx, num in enumerate(int_order):
        fields = get_wikibio_poswrds(lst_src[num])
        fields = [wrd for (k, idx), wrd in fields.items()]
        result.append(fields)

    result_2 = []
    x_idx = 0

    temp_store = []
    for ii in range(len(beam_lst)):
        try:
            x = result[x_idx]
            y = beam_lst[ii]
        except:
            print('x_idx is out of range for x:', x_idx, ii)
        try:
            y1, score_1, state_1, rank1, copy1 = y.split('|||')
        except:
            continue


        if int(rank1) == 0:
            # handle the rescore of the previous list:
            if len(temp_store) > 0:
                # sort  from  temp_score
                for score_, elem in sorted(temp_store, key= lambda a: a[0])[:1]:
                    y, score_, state_, rank, copy = elem
                    copy = ast.literal_eval(copy)
                    y = y.split()
                    for idx, elem in enumerate(y):
                        if elem == '<unk>':
                            # print(y[idx], copy[idx])
                            if copy[idx] >= 0 and copy[idx] < len(x):
                                # print(result[ii], len(result[ii]), copy[idx])
                                y[idx] = x[copy[idx]]
                    if '<eos>' in y:
                        temp_id = y.index('<eos>')
                        y = y[:temp_id]
                    result_2.append(' '.join(y))
                x_idx += 1
            temp_store = []
            rescore = (5 + len(y1.split())) / 6
            score_ = float(score_1)
            temp_store.append((score_ / rescore, (y1, score_1, state_1, rank1, copy1)))

        else:
            rescore = (5 + len(y1.split())) / 6
            score_ = float(score_1)
            temp_store.append((score_ / rescore, (y1, score_1, state_1, rank1, copy1)))


    return result_2

def replace_unk_full(beam_lst, lst_src, int_order):
    result = []
    for idx, num in enumerate(int_order):
        fields = get_wikibio_poswrds(lst_src[num])
        fields = [wrd for (k, idx), wrd in fields.items()]
        result.append(fields)

    result_2 = []
    x_idx = 0

    temp_store = []
    for ii in range(len(beam_lst)):
        try:
            x = result[x_idx]
            y = beam_lst[ii]
        except:
            print('x_idx is out of range for x:', x_idx, ii)
        try:
            y1, score_1, state_1, rank1, copy1 = y.split('|||')
        except:
            continue


        if int(rank1) == 0:
            # handle the rescore of the previous list:
            if len(temp_store) > 0:
                # sort  from  temp_score
                for score_, elem in sorted(temp_store, key= lambda a: a[0])[:1]:
                    y, score_, state_, rank, copy = elem
                    copy = ast.literal_eval(copy)
                    y = y.split()
                    for idx, elem in enumerate(y):
                        if elem == '<unk>':
                            # print(y[idx], copy[idx])
                            if copy[idx] >= 0 and copy[idx] < len(x):
                                # print(result[ii], len(result[ii]), copy[idx])
                                y[idx] = x[copy[idx]]
                    # if '<eos>' in y:
                    #     temp_id = y.index('<eos>')
                    #     y = y[:temp_id]
                    result_2.append('{}|||{}|||{}|||{}|||{}'.format(' '.join(y), score_, state_, rank, copy))
                x_idx += 1
            temp_store = []
            rescore = 1 #(5 + len(y1.split())) / 6
            score_ = float(score_1)
            temp_store.append((score_ / rescore, (y1, score_1, state_1, rank1, copy1)))

        else:
            rescore = 1 #(5 + len(y1.split())) / 6
            score_ = float(score_1)
            temp_store.append((score_ / rescore, (y1, score_1, state_1, rank1, copy1)))


    return result_2

def replace_unk_(beam_lst, lst_src, int_order):
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
                    if copy[idx] >= 0 and copy[idx] < len(x):
                        # print(result[ii], len(result[ii]), copy[idx])
                        y[idx] = x[copy[idx]]
            if '<eos>' in y:
                temp_id = y.index('<eos>')
                y = y[:temp_id]
            result_2.append(' '.join(y))
            x_idx += 1
    return result_2





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


def combine_beam_e2e(int_order, true_ref, true_src, out_path):
    result = []
    for idx, num in enumerate(int_order):
        result.append(true_ref[num])
    with open(out_path + 'corr', 'w') as f:
        for elem in result:
            print(elem, file=f)

    result_src = []
    for idx, num in enumerate(int_order):
        result_src.append(true_src[num])

    with open(out_path + 'src', 'w') as f:
        for elem in result_src:
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
    # process_full_beam_e2e(data_path, beam_path, out_path, corpus_type='test_full1')
    # process_full_beam_e2e(data_path, beam_path, out_path, corpus_type=sys.argv[1])
    process_full_beam2(data_path, beam_path, out_path, corpus_type=sys.argv[1])
    #





    # read from our simplified output files
    # combine them.







