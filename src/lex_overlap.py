
import argparse
import labeled_data2 as labeled_data

parser = argparse.ArgumentParser(description='')
parser.add_argument('--tagset_size', type=int, default=45, help='path to save the final model')
parser.add_argument('--max_seqlen', type=int, default=70, help='')
parser.add_argument('--bsz', default=10, type=int, help='do not decay learning rate for at least this many epochs')
parser.add_argument('--data_ptb_path', type=str, default='../data/ptb', help='path to saved model')
parser.add_argument('--data', type=str, default='../data/e2e_aligned', help='path to saved model')
parser.add_argument('--induced_path', type=str, default='', help='path to saved model')
parser.add_argument('--test', action='store_true', help='use test data')
args = parser.parse_args()

e2e_keys = ["name", "eatType", "food", "priceRange", "customerrating", "area", "near", 'familyFriendly']
e2e_key2idx = dict((key, i) for i, key in enumerate(e2e_keys))

corpus = labeled_data.SentenceCorpus(args.data, args.bsz, thresh=1, add_bos=False,
                                                 add_eos=False, test=False, L=8, K=11, maxseqlen=70)

composed_map = {}
print(corpus.field_names2idx)
for xx_k, xx_v in corpus.field_names2idx.items():
    if xx_k[1:] in e2e_key2idx:
        composed_map[xx_v] = e2e_key2idx[xx_k[1:]]

print(composed_map)

corpora = corpus.train
for i in range(len(corpora)):
    x, lab, src, locs, inps, en_src, en_targ, src_wrd2fields = corpora[i]
    cidxs = None
    seqlen, bsz = x.size()
    nfields = src.size(1)
    print(corpus.field_names2idx['<fc_field>'])

    print(en_targ)
    print(lab)
    full_lst = []
    for b in range(bsz):
        cpp = inps[:,b,0,1].tolist()
        known_lst = [-1] * seqlen
        for (s1, ee, s2) in lab[b]:

            known_lst[s1:ee] = [s2] * (ee - s1)
        for widx, word in enumerate(cpp):
            if known_lst[widx] == -1:
                # use the copy check
                if word in composed_map:
                    known_lst[widx] = composed_map[word]


