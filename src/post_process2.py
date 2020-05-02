import sys
import argparse
import os
import ast

def print_segment(segment_lst, sent):
    result = ''
    for (start, end, state) in segment_lst:
        result += '{' + str(state) + ' ' + ' '.join(sent[start:end]) + '} '
    return result

def see_result(lst_tgt, result_path, out_path):
    temp_full = []
    for temp in lst_tgt:
        temp_full.append(temp)

    result = []
    with open(result_path, 'r') as f:
        for line in f:
            line = ast.literal_eval(line)
            line_num = int(line[1])
            segment = line[0]
            gold = temp_full[line_num]
            result.append(print_segment(segment, gold))

    print('final result is written to {}'.format(out_path + '_ref'))
    with open(out_path + '_ref', 'w') as f:
        for elem in result:
            print(elem, file=f)

    return


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

def get_sents(path, src_path):
    lst_tgt, lst_src = [], []
    print(path)
    assert os.path.exists(path)

    if src_path is not None:
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
                y = y[:temp_id+1]
            result_2.append(' '.join(y))
            x_idx += 1
    return result_2


def combine_beam(int_order, true_ref, out_path):
    result = []
    for idx, num in enumerate(int_order):
        result.append(true_ref[num])
    with open(out_path + '_ref', 'w') as f:
        for elem in result:
            print(elem, file=f)
    return


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--result_path', type=str, default='', help='path to save the final model')
    parser.add_argument('--out_path', type=str, default='save_out', help='path to save the output')
    parser.add_argument('--type', default=1, type=int, help='1: look at state, 2: eval for NLG result')
    parser.add_argument('--data_path', type=str, default='', help='path to data')
    parser.add_argument('--notes', type=str, default='', help='wiki or e2e')

    args = parser.parse_args()

    print('start post-process the result for {}'.format(args.notes))
    print('the path prefix is {}'.format(args.result_path))
    if args.type == 1:
        # src_path = 'src_' + args.data_path
        path = args.data_path
        lst_tgt, lst_src = get_sents(path, None)
        see_result(lst_tgt, args.result_path, args.out_path)

    if args.type == 2:
        result_src_path, result_ref_path, result_sys_path = args.result_path.format('src'), args.result_path.format('corr'), args.result_path.format('beam')
        print('src:{}\nref:{}\nsys:{}'.format(result_src_path, result_ref_path, result_sys_path))

        data_path = sys.argv[2]  # the path to data directory.
        beam_path = sys.argv[3]  # the path to beam output base.
        out_path = sys.argv[4]  # the file we want to output to.
        print('{}\n{}\n{}'.format(data_path, beam_path, out_path))
        process_full_beam2(data_path, beam_path, out_path, corpus_type='test')





