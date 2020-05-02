import os
import sys
import torch
import json

from utils import get_wikibio_fields

train_dir = "wikipedia-biography-dataset/train"
# val_dir = "wikipedia-biography-dataset/valid"
val_dir = "wikipedia-biography-dataset/test"

punctuation = set(['.', '!', ',', ';', ':', '?', '--', '-rrb-', '-lrb-'])

# from wikipedia
prepositions = set(['aboard', 'about', 'above', 'absent', 'across', 'after', 'against', 'along', 'alongside', 'amid', 'among',
                    'apropos', 'apud', 'around', 'as', 'astride', 'at', 'atop', 'bar', 'before', 'behind', 'below', 'beneath',
                    'beside', 'besides', 'between', 'beyond', 'but', 'by', 'chez', 'circa', 'come', 'despite', 'down', 'during',
                    'except', 'for', 'from', 'in', 'inside', 'into', 'less', 'like', 'minus', 'near', 'notwithstanding', 'of',
                    'off', 'on', 'onto', 'opposite', 'out', 'outside', 'over', 'pace', 'past', 'per', 'plus', 'post', 'pre',
                    'pro', 'qua', 're', 'sans', 'save', 'short', 'since', 'than', 'through', 'throughout', 'till', 'to', 'toward',
                    'under', 'underneath', 'unlike', 'until', 'unto', 'up', 'upon', 'upside', 'versus', 'via', 'vice', 'aboard',
                    'about', 'above', 'absent', 'across', 'after', 'against', 'along', 'alongside', 'amid', 'among', 'apropos',
                    'apud', 'around', 'as', 'astride', 'at', 'atop', 'bar', 'before', 'behind', 'below', 'beneath', 'beside', 'besides',
                    'between', 'beyond', 'but', 'by', 'chez', 'circa', 'come', 'despite', 'down', 'during', 'except', 'for', 'from', 'in',
                    'inside', 'into', 'less', 'like', 'minus', 'near', 'notwithstanding', 'of', 'off', 'on', 'onto', 'opposite', 'out',
                    'outside', 'over', 'pace', 'past', 'per', 'plus', 'post', 'pre', 'pro', 'qua', 're', 'sans', 'save', 'short', 'since',
                    'than', 'through', 'throughout', 'till', 'to', 'toward', 'under', 'underneath', 'unlike', 'until', 'unto', 'up', 'upon',
                    'upside', 'versus', 'via', 'vice', 'with', 'within', 'without', 'worth'])


splitters = set(['and', ',', 'or', 'of', 'for', '--', 'also'])

goodsplitters = set([',', 'of', 'for', '--', 'also']) # leaves out and and or

def splitphrs(tokes, l, r, max_phrs_len, labelist, lab_field):
    if r-l <= max_phrs_len:
        labelist.append((l, r, lab_field))
    else:
        i = r-1
        found_a_split = False
        while i > l:
            if tokes[i] in goodsplitters or tokes[i] in prepositions:
                splitphrs(tokes, l, i, max_phrs_len, labelist, lab_field)
                if i < r-1:
                    splitphrs(tokes, i+1, r, max_phrs_len, labelist, lab_field)
                found_a_split = True
                break
            i -= 1
        if not found_a_split: # add back in and and or
            i = r-1
            while i > l:
                if tokes[i] in splitters or tokes[i] in prepositions:
                    splitphrs(tokes, l, i, max_phrs_len, labelist, lab_field)
                    if i < r-1:
                        splitphrs(tokes, i+1, r, max_phrs_len, labelist, lab_field)
                    found_a_split = True
                    break
                i -= 1
        if not found_a_split: # just do something
            i = r-1
            while i >= l:
                max_len = min(max_phrs_len, i-l+1)
                labelist.append((i-max_len+1, i+1, lab_field))
                i = i-max_len


def stupid_search(tokes, fields, field_dict):
    """
    greedily assigns longest labels to spans from right to left
    """
    PFL = 4
    labels = []
    i = len(tokes)
    corr_fields = [k for k, v in fields.iteritems()]
    wordsets = [set(toke for toke in v if toke not in punctuation) for k, v in fields.iteritems()]
    pfxsets = [set(toke[:PFL] for toke in v if toke not in punctuation) for k, v in fields.iteritems()]
    while i > 0:
        matched = False
        if tokes[i-1] in punctuation:
            labels.append((i-1, i, len(field_dict))) # all punctuation
            i -= 1
            continue
        if tokes[i-1] in punctuation or tokes[i-1] in prepositions or tokes[i-1] in splitters:
            i -= 1
            continue
        for j in xrange(i):
            if tokes[j] in punctuation or tokes[j] in prepositions or tokes[j] in splitters:
                continue
            # then check if it matches stuff in the table
            tokeset = set(toke for toke in tokes[j:i] if toke not in punctuation)
            for ii, vset in enumerate(wordsets):
                if tokeset == vset or (tokeset.issubset(vset) and len(tokeset) > 1):
                    if i - j > max_phrs_len:
                        nugz = []
                        if corr_fields[ii] not in field_dict:
                            field_lab = len(field_dict) + 2
                        else:
                            field_lab = field_dict[corr_fields[ii]]
                        splitphrs(tokes, j, i, max_phrs_len, nugz, field_lab)
                        labels.extend(nugz)
                    else:
                        if corr_fields[ii] not in field_dict:
                            field_lab = len(field_dict) + 2
                        else:
                            field_lab = field_dict[corr_fields[ii]]
                        labels.append((j, i, field_lab))
                    i = j
                    matched = True
                    break
            if matched:
                break
            pset = set(toke[:PFL] for toke in tokes[j:i] if toke not in punctuation)
            for ii, pfxset in enumerate(pfxsets):
                if pset == pfxset or (pset.issubset(pfxset)and len(pset) > 1):
                    if i - j > max_phrs_len:
                        nugz = []
                        if corr_fields[ii] not in field_dict:
                            field_lab = len(field_dict) + 2
                        else:
                            field_lab = field_dict[corr_fields[ii]]
                            
                        splitphrs(tokes, j, i, max_phrs_len, nugz, field_lab)
                        labels.extend(nugz)
                    else:
                        if corr_fields[ii] not in field_dict:
                            field_lab = len(field_dict) + 2
                        else:
                            field_lab = field_dict[corr_fields[ii]]
                        
                        labels.append((j, i, field_lab))
                    i = j
                    matched = True
                    break
            if matched:
                break
        if not matched:
            i -= 1
    labels.sort(key=lambda x: x[0])
    return labels

def print_data(direc, dev=0):
    fis = os.listdir(direc)
    srcfi = [fi for fi in fis if fi.endswith('.box')][0]
    tgtfi = [fi for fi in fis if fi.endswith('.sent')][0]
    nbfi = [fi for fi in fis if fi.endswith('.nb')][0]

    with open(os.path.join(direc, srcfi)) as f:
        srclines = f.readlines()
    with open(os.path.join(direc, nbfi)) as f:
        nbs = [0]
        [nbs.append(int(line.strip())) for line in f.readlines()]
        nbs = set(torch.Tensor(nbs).cumsum(0))

    tgtlines = []
    with open(os.path.join(direc, tgtfi)) as f:
        for i, tgtline in enumerate(f):
            if i in nbs:
                tgtlines.append(tgtline)

    if dev == 0:
        field_sets = set()
        for i in xrange(len(srclines)):
            fields = get_wikibio_fields(srclines[i].strip().split())
            field_sets.update([key for key in fields])

        field_lst = list(field_sets)
        field_dict = {x:idx for idx, x in enumerate(field_lst)}
        ''' write to the system -> a lookup table for the field_lst'''
        with open('wb_field', 'wb') as ff:
            json.dump([field_lst, field_dict], ff)
    else:
        with open('wb_field', 'rb') as ff:
            field_lst, field_dict = json.load(ff)


    assert len(srclines) == len(tgtlines)
    for i in xrange(len(srclines)):
        fields = get_wikibio_fields(srclines[i].strip().split())
        tgttokes = tgtlines[i].strip().split()
        labels = stupid_search(tgttokes, fields, field_dict)
        labels = [(str(tup[0]), str(tup[1]), str(tup[2])) for tup in labels]
        # add eos stuff
        tgttokes.append("<eos>")
        labels.append((str(len(tgttokes)-1), str(len(tgttokes)), str(len(field_lst) + 1) )) # label doesn't matter

        labelstr = " ".join([','.join(label) for label in labels])
        sentstr = " ".join(tgttokes)

        outline = "%s|||%s" % (sentstr, labelstr)
        print outline

if __name__ == "__main__":
    max_phrs_len = int(sys.argv[2])
    if sys.argv[1] == "train":
        print_data(train_dir, 0)
    elif sys.argv[1] == "valid":
        print_data(val_dir, 1)
    else:
        assert False
