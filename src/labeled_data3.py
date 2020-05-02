"""
this file modified from the word_language_model example
"""
import os
import torch
import sys

from collections import Counter, defaultdict

from utils_data import get_wikibio_poswrds, get_e2e_poswrds
from utils import gen_detailed_tgt_mask_pre, make_bwd_idxs_pre2
import pickle
import random
random.seed(1111)

#punctuation = set(['.', '!', ',', ';', ':', '?', '--', '-rrb-', '-lrb-'])
punctuation = set() # i don't know why i was so worried about punctuation



e2e_keys = ["name", "eatType", "food", "priceRange", "customerrating", "area", "near", 'familyFriendly']
e2e_key2idx = dict((key, i) for i, key in enumerate(e2e_keys))


# print(composed_map)


class Dictionary(object):
    def __init__(self, unk_word="<unk>"):
        self.unk_word = unk_word
        self.idx2word = [unk_word, "<pad>", "<bos>", "<eos>"] # OpenNMT constants
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def add_word(self, word, train=False):
        """
        returns idx of word
        """
        if train and word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word] if word in self.word2idx else self.word2idx[self.unk_word]

    def bulk_add(self, words):
        """
        assumes train=True
        """
        self.idx2word.extend(words)
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.idx2word)


class SentenceCorpus(object):
    def __init__(self, path, bsz, L, K, maxseqlen, option='train', thresh=5, add_bos=False, add_eos=False,
                 test=False, task='bert'):
        self.dictionary = Dictionary()

        self.L = L
        self.K = K
        self.maxseqlen = maxseqlen

        self.bsz = bsz
        self.wiki = "wb" in path

        self.field_names2idx = {}
        # self.field_names2idx = {'<unk_field>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2idx = {'<unk_idx>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
        # self.idx2idx = {}

        temp_name = 'train'
        temp_val_name = 'valid'

        print(task)
        if task == 'bert' or task == 'bert_distrib':
            app_ = '_chk_bert'  # bert result...
        elif task == 'lex':
            app_ = ''           # lexical overlap
        elif task == 'lex_rnn':
            app_ = '_seq_lex'     # lexical overlap filled with garbage collecting.
        elif task == 'bert_rnn':
            app_ = '_chk_bert'  # bert result...
        elif task == 'wb_global':
            app_ = '_fieldlex'  # WikiBio global pr feature


        if task == 'bert_distrib':
            app2_ = '_distrib.pkl'# soft version
        else:
            app2_ = ''


        if app2_ == '_distrib.pkl':
            self.minibatchify = self.minibatchify2

        self.app_ = app_
        if not self.wiki:
            # temp_test_name = 'test_full1'
            temp_test_name = 'test'
        else:
            temp_test_name = 'test1'


        if not self.wiki:
            # MARCC
            self.mem_eff_flag = True
            # self.mem_eff_flag = False
            print('labeled_data mem status is: ', self.mem_eff_flag)
        else:
            self.mem_eff_flag = True
            print('labeled_data mem status is: ', self.mem_eff_flag)


        train_src = os.path.join(path, "src_{}.txt".format(temp_name))
        sys.stdout.flush()
        if thresh > 0:
            self.get_vocabs(os.path.join(path, '{}{}.txt'.format(temp_name, app_)), train_src, thresh=thresh)
            self.ngen_types = len(self.genset) + 4 # assuming didn't encounter any special tokens
            add_to_dict = False
        else:
            add_to_dict = True
        print('finished _vocab section ')
        print ("using vocabulary of size:", len(self.dictionary))
        print('using field vocab of size:', len(self.field_names2idx))
        print (self.ngen_types, "gen word types")
        sys.stdout.flush()


        if option == 'train':
            trsents, trlabels, trfeats, trlocs, inps, tren_src, tren_targ, trsrc_wrd2fields = self.tokenize(
                os.path.join(path, '{}{}.txt'.format(temp_name, app_)), train_src, add_to_dict=add_to_dict,
                add_bos=add_bos, add_eos=add_eos)
            if app2_ == '_distrib.pkl':
                with open(os.path.join(path,'train' + app2_), 'rb') as f1:
                    trdistrib = pickle.load(f1)
                    print(len(trdistrib), len(trsents))
            else:
                trdistrib = None

            # self.reprint(trsents, trlabels,inps, tren_targ)

            self.train, self.train_mb2linenos = self.minibatchify(
                trsents, trlabels, trfeats, trlocs, inps, tren_src, tren_targ, trdistrib, bsz) # list of minibatches

        if (os.path.isfile(os.path.join(path, '{}{}.txt'.format(temp_val_name, app_)))
                or os.path.isfile(os.path.join(path, 'test.txt'))):
            if not test:
                val_src = os.path.join(path, "src_{}.txt".format(temp_val_name))
                vsents, vlabels, vfeats, vlocs, vinps, ven_src, ven_targ, vsrc_wrd2fields = self.tokenize(
                    os.path.join(path, '{}{}.txt'.format(temp_val_name, app_)), val_src, add_to_dict=False,
                    add_bos=add_bos, add_eos=add_eos)

                if app2_ == '_distrib.pkl':
                    with open(os.path.join(path, 'valid' + app2_), 'rb') as f1:
                        vdistrib = pickle.load(f1)
                        print(len(vdistrib), len(vsents))
                else:
                    vdistrib = None
            else:
                print ("using test data and whatnot....")
                test_src = os.path.join(path, "src_{}.txt".format(temp_test_name))
                vsents, vlabels, vfeats, vlocs, vinps, ven_src, ven_targ, vsrc_wrd2fields = self.tokenize(
                    os.path.join(path, '{}{}.txt'.format(temp_test_name, app_)), test_src, add_to_dict=False,
                    add_bos=add_bos, add_eos=add_eos)

                if app2_ == '_distrib.pkl':
                    with open(os.path.join(path, 'test' + app2_), 'rb') as f1:
                        vdistrib = pickle.load(f1)
                        print(len(vdistrib), len(vsents))
                else:
                    vdistrib = None

            # self.reprint(vsents, vlabels, vinps, ven_targ, 'test')
            self.valid, self.val_mb2linenos = self.minibatchify(
                vsents, vlabels, vfeats, vlocs, vinps, ven_src, ven_targ, vdistrib, bsz)
        else:
            print('no dev or test data presented. ')
        print('loaded dataset.')
        sys.stdout.flush()

    def reprint(self,trsents, trlabels,inps, en_targ, tag='train'):
        composed_map = {}
        # print(corpus.field_names2idx)
        for xx_k, xx_v in self.field_names2idx.items():
            if xx_k[1:] in e2e_key2idx:
                composed_map[xx_v] = e2e_key2idx[xx_k[1:]]

        full_lst = []
        for i in range(len(trsents)):
            cpp = [xx[0][1] for xx in inps[i]]
            lab = trlabels[i]
            known_lst = [9] * len(cpp)
            for (s1, ee, s2) in lab:
                known_lst[s1:ee] = [s2] * (ee - s1)
            for widx, word in enumerate(cpp):
                if known_lst[widx] == 9:
                    # use the copy check
                    if word in composed_map:
                        known_lst[widx] = composed_map[word]
            full_lst.append(known_lst)

        with open('../data/e2e_aligned/lex2_' +  tag, 'w') as f:
            for ii, ff in enumerate(full_lst):
                ff = [str(xx) for xx in ff]
                print( '{}|||{}'.format(' '.join(en_targ[ii]), ' '.join(ff)), file=f)



    def get_vocabs(self, path, src_path, thresh=2):
        """unks words occurring <= thresh times"""
        print(path)
        criteria = 'threshold'
        criteria = 'topK'
        assert os.path.exists(path)

        field_voc = Counter()
        tgt_voc = Counter()
        idx_voc = Counter()
        genwords = Counter()
        fieldvals_voc = Counter()


        with open(src_path, 'r') as src, open(path, 'r') as tgt:
            for line_src, line_tgt in zip(src, tgt):
                tokes_src = line_src.strip().split()
                if self.wiki:
                    fields = get_wikibio_poswrds(tokes_src) #key, pos -> wrd
                else:
                    fields = get_e2e_poswrds(tokes_src) # key, pos -> wrd
                fieldvals = fields.values()
                # update word counts.
                ##1
                fieldvals_voc.update(fieldvals)
                field_voc.update([k for k, idx, in fields])
                idx_voc.update([idx for k, idx in fields])
                set_src = set(wrd for wrd in fieldvals)

                words, spanlabels = line_tgt.strip().split('|||')
                words = words.split()
                genwords.update([wrd for wrd in words if wrd not in set_src])
                tgt_voc.update(words)

        # prune
        # N.B. it's possible a word appears enough times in total but not in genwords
        # so we need separate unking for generation
        if criteria == 'threshold':
            for cntr in [tgt_voc, genwords, field_voc, idx_voc]:
                del_lst = []
                for k in cntr.keys():
                    if cntr[k] < thresh:
                        del_lst.append(k)
                for k in del_lst:
                    del cntr[k]

            self.field_names = list(field_voc.keys())
            self.field_names2idx = {field_n: idx for idx, field_n in enumerate(self.field_names)}
            # for idx, ww in enumerate(['<unk_field>', '<pad>', '<bos>', '<eos>']):
            #     self.field_names2idx[ww] = idx
            self.field_names2idx['<fc_field>'] = len(self.field_names2idx)

            self.idx2idx = {ii: idx+4 for idx, ii in enumerate(list(idx_voc))}
            for idx, ww in enumerate(['<unk_idx>', '<pad>', '<bos>', '<eos>']):
                self.idx2idx[ww] = idx
            self.idx2idx['<fc_idx>'] = len(self.idx2idx)
            self.genset = list(genwords.keys())
            tgtkeys = list(tgt_voc.keys())
            # make sure gen stuff is first
            tgtkeys.sort(key=lambda x: -(x in self.genset))
            ##1
            self.dictionary.bulk_add(tgtkeys)
            # the previous things are [unk_word, "<pad>", "<bos>", "<eos>"] .
            # make sure we did everything right (assuming didn't encounter any special tokens)
            assert self.dictionary.idx2word[4 + len(self.genset) - 1] in self.genset
            assert self.dictionary.idx2word[4 + len(self.genset)] not in self.genset

        else:
            tgt_topK = 20000
            field_topK = 1480
            idx_topK = 30

            # tgt_topK = 200
            # field_topK = 148
            # idx_topK = 20

            tgt_voc = [x[0] for x in tgt_voc.most_common(tgt_topK)]
            field_voc = [x[0] for x in field_voc.most_common(field_topK)]
            idx_voc = [x[0] for x in idx_voc.most_common(idx_topK)]

            ''' find the intersection between the geneset and the tgt_voc'''
            self.genset = set(genwords.keys()).intersection(tgt_voc)

            self.field_names = field_voc
            self.field_names2idx = {field_n: idx for idx, field_n in enumerate(self.field_names)}
            # for idx, ww in enumerate(['<unk_field>', '<pad>', '<bos>', '<eos>']):
            #     self.field_names2idx[ww] = idx
            self.field_names2idx['<fc_field>'] = len(self.field_names2idx)
            assert len(self.field_names) + 1 == len(self.field_names2idx)

            self.idx2idx = {ii: idx + 4 for idx, ii in enumerate(idx_voc)}
            for idx, ww in enumerate(['<unk_idx>', '<pad>', '<bos>', '<eos>']):
                self.idx2idx[ww] = idx
            self.idx2idx['<fc_idx>'] = len(self.idx2idx)
            assert len(idx_voc) + 5 == len(self.idx2idx)

            tgtkeys = tgt_voc
            # make sure gen stuff is first
            tgtkeys.sort(key=lambda x: -(x in self.genset))
            self.dictionary.bulk_add(tgtkeys)
            # the previous things are [unk_word, "<pad>", "<bos>", "<eos>"] .
            # make sure we did everything right (assuming didn't encounter any special tokens)
            assert self.dictionary.idx2word[4 + len(self.genset) - 1] in self.genset
            assert self.dictionary.idx2word[4 + len(self.genset)] not in self.genset
        self.non_field = self.field_names2idx['<fc_field>']






    def minibatchify2(self, sents, labels, feats, locs, inps, en_src, en_targ, distrib, bsz, pre_compute=True):
        """
        this should result in there never being any padding.
        each minibatch is:
          (seqlen x bsz, bsz-length list of lists of (start, end, label) constraints,
           bsz x nfields x nfeats, seqlen x bsz x max_locs, seqlen x bsz x max_locs x nfeats)
        """

        # sort in ascending order
        sents, _, sorted_idxs = zip(*sorted(zip(sents, feats, range(len(sents))), key=lambda x: (len(x[0]), len(x[1]))))
        minibatches, mb2linenos = [], []
        curr_batch, curr_labels, curr_feats, curr_locs, curr_linenos, curr_ensrc, curr_entarg = \
            [], [], [], [], [], [], []
        # curr_src_wrd2fields = []
        curr_distrib = []
        curr_inps = []
        curr_len = len(sents[0])
        for i in range(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz: # we're done
                # minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                #                     curr_labels, self.padded_feat_mb(curr_feats),
                #                     self.padded_loc_mb(curr_locs),
                #                     self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous(),
                #                     curr_ensrc, curr_entarg, curr_src_wrd2fields ))

                if pre_compute:
                    special_inps = self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()
                    # special_pr_mask = gen_detailed_tgt_mask_pre(special_inps, self.L, self.non_field, self.maxseqlen)
                    seqlen = special_inps.size(0)
                    # special_pr_mask = make_bwd_idxs_pre2(self.L, seqlen, self.K, curr_labels)
                    minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                        curr_labels, self.padded_feat_mb(curr_feats),
                                        self.padded_loc_mb(curr_locs),
                                        special_inps,
                                        curr_ensrc, curr_entarg, curr_distrib))
                else:

                    minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                        curr_labels, self.padded_feat_mb(curr_feats),
                                        self.padded_loc_mb(curr_locs),
                                        self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous(),
                                        curr_ensrc, curr_entarg, None))

                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                curr_labels = [labels[sorted_idxs[i]]]
                curr_feats = [feats[sorted_idxs[i]]]
                curr_locs = [locs[sorted_idxs[i]]]
                curr_inps = [inps[sorted_idxs[i]]]
                curr_distrib = [distrib[sorted_idxs[i]]]
                if not self.mem_eff_flag:
                    pass
                    # curr_ensrc.append(en_src[sorted_idxs[i]])
                curr_entarg = [en_targ[sorted_idxs[i]]]
                # curr_src_wrd2fields = [src_wrd2fields[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                curr_labels.append(labels[sorted_idxs[i]])
                curr_feats.append(feats[sorted_idxs[i]])
                curr_locs.append(locs[sorted_idxs[i]])
                curr_inps.append(inps[sorted_idxs[i]])
                curr_distrib.append(distrib[sorted_idxs[i]])

                if not self.mem_eff_flag:
                    pass
                    # curr_ensrc.append(en_src[sorted_idxs[i]])
                curr_entarg.append(en_targ[sorted_idxs[i]])
                # curr_src_wrd2fields.append(src_wrd2fields[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
        # catch last
        if len(curr_batch) > 0:
            # minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
            #                     curr_labels, self.padded_feat_mb(curr_feats),
            #                     self.padded_loc_mb(curr_locs),
            #                     self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous(),
            #                     curr_ensrc, curr_entarg, curr_src_wrd2fields))

            if pre_compute:
                special_inps = self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()
                # special_pr_mask = gen_detailed_tgt_mask_pre(special_inps, self.L, self.non_field, self.maxseqlen)
                seqlen = special_inps.size(0)
                # special_pr_mask = make_bwd_idxs_pre2(self.L, seqlen, self.K, curr_labels)
                minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                    curr_labels, self.padded_feat_mb(curr_feats),
                                    self.padded_loc_mb(curr_locs),
                                    special_inps,
                                    curr_ensrc, curr_entarg, curr_distrib))
            else:

                minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                    curr_labels, self.padded_feat_mb(curr_feats),
                                    self.padded_loc_mb(curr_locs),
                                    self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous(),
                                    curr_ensrc, curr_entarg, None))
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos

    def tokenize(self, path, src_path, add_to_dict=False, add_bos=False, add_eos=False):
        """Assumes fmt is sentence|||s1,e1,k1 s2,e2,k2 ...."""
        print(path)
        assert os.path.exists(path)


        src_feats = []
        sents, labels, copylocs, inps = [], [], [], []
        en_src = []
        en_targ = []

        w2i = self.dictionary.word2idx
        unk_word = w2i['<unk>']
        # unk_field = self.field_names2idx['<unk_field>']
        unk_idx = self.idx2idx['<unk_idx>']
        place_holders =  [self.field_names2idx['<fc_field>'], self.idx2idx['<fc_idx>'], self.idx2idx['<fc_idx>']]

        # if not self.wiki:
        if False:
            e2e_keys = ["name", "eatType", "food", "priceRange", "customerrating", "area", "near"]
            e2e_key2idx = dict((key, i) for i, key in enumerate(e2e_keys))
            print(self.field_names)
            print(w2i)
            temp_dict = {e2e_key2idx[x[1:]]: w2i[x] for x in self.field_names if
                         x[1:] in e2e_key2idx}  # from e2ekey to index of the dictionary.


        with open(src_path, 'r') as src, open(path, 'r') as tgt:
            for tgtline, (line_src, line_tgt) in enumerate(zip(src, tgt)):
                tokes_src = line_src.strip().split()
                if self.wiki:
                    fields = get_wikibio_poswrds(tokes_src) #key, pos -> wrd
                else:
                    fields = get_e2e_poswrds(tokes_src) # key, pos -> wrd
                # wrd2things will be unordered
                feats, wrd2idxs, wrd2fields = [], defaultdict(list), defaultdict(list)
                fld_cntr = Counter([key for key, _ in fields])

                for (k, idx), wrd in fields.items():
                    if k in self.field_names2idx:
                        backward = fld_cntr[k] + 1 - idx
                        back_idx = self.idx2idx[backward] if backward in self.idx2idx else unk_idx
                        fore_idx = self.idx2idx[idx] if idx in self.idx2idx else unk_idx
                        word_idx = w2i[wrd] if wrd in w2i else unk_word
                        # field_idx = self.field_names2idx[k] if k in self.field_names2idx else unk_field
                        field_idx = self.field_names2idx[k]
                        featrow = [word_idx, field_idx, fore_idx, back_idx]
                        wrd2fields[wrd].append(featrow)
                        wrd2idxs[wrd].append(len(feats))
                        feats.append(featrow)
                src_feats.append(feats)

                #  the tgt side
                words, spanlabels = line_tgt.strip().split('|||')
                words = words.split()
                sent, copied, insent = [], [], []

                if add_bos:
                    sent.append(self.dictionary.add_word('<bos>', True))

                labetups = [tupstr.split(',') for tupstr in spanlabels.split()]
                if self.app_[:6] == '_field':
                    labelist = [(int(tup[0]), int(tup[1]), self.field_names2idx[tup[2]])  for tup in labetups]
                    # print(labelist)
                else:
                    labelist = [(int(tup[0]), int(tup[1]), int(tup[2])) for tup in labetups]

                for index1, word in enumerate(words):
                    # sent is just used for targets; we have separate inputs
                    if word in self.genset:
                        sent.append(w2i[word])
                    else:
                        sent.append(w2i["<unk>"])

                    if word in wrd2idxs:
                        copied.append(wrd2idxs[word])
                        winps = [[widx, kidx, idxidx, nidx]
                                 for widx, kidx, idxidx, nidx in wrd2fields[word]]
                        # consider cases that multiple source can contribute to some generation

                        # TODO: a bug here that involves how we pad copied, how we use combo_targs.
                        # if word in self.genset:
                        #     copied[-1].append(-1)
                        #     winps.append([sent[-1], place_holders[0], place_holders[1], place_holders[2]])

                        insent.append(winps)
                    else:
                        copied.append([-1])
                        insent.append([[sent[-1], place_holders[0], place_holders[1], place_holders[2]]])

                if add_eos:
                    sent.append(self.dictionary.add_word('<eos>', True))

                # if not self.wiki:
                if False:
                    self.pure_label_lst(labelist, copied, insent, w2i, temp_dict)

                sents.append(sent)
                labels.append(labelist)
                copylocs.append(copied)
                inps.append(insent)
                if self.mem_eff_flag:
                    en_targ.append(tgtline)
                else:
                    en_targ.append(words)

        print(len(sents))
        assert len(sents) == len(labels)
        assert len(src_feats) == len(sents)
        assert len(copylocs) == len(sents)
        if self.mem_eff_flag:
            pass
        else:
            print(len(en_targ), len(sents), len(en_src))
            assert len(en_targ) == len(sents)


        ''' sent is the indexed-sentence, label is the triplet, src_feats is the feature (field_name, index, word) as 
            marked in the source. copy locs means the location of the copied word in the target sentence. inps is the 
            4 element feature for each word in the target sentence. [word index, field, index, is this the last element
            in the source phrase]
        '''
        return sents, labels, src_feats, copylocs, inps, en_src, en_targ, None

    def featurize_tbl(self, fields):
        """
        fields are key, pos -> wrd maps
        returns: nrows x nfeats tensor
        """
        feats = []
        for (k, idx), wrd in fields.iteritems():
            if k in self.dictionary.word2idx:
                featrow = [self.dictionary.add_word(k, False),
                           self.dictionary.add_word(idx, False),
                           self.dictionary.add_word(wrd, False)]
                feats.append(featrow)
        return torch.LongTensor(feats)

    def padded_loc_mb(self, curr_locs):
        """
        curr_locs is a bsz-len list of tgt-len list of locations
        returns:
          a seqlen x bsz x max_locs tensor
        """
        max_locs = max(len(locs) for blocs in curr_locs for locs in blocs)
        for blocs in curr_locs:
            for locs in blocs:
                if len(locs) < max_locs:
                    locs.extend([-1]*(max_locs - len(locs)))
        return torch.LongTensor(curr_locs).transpose(0, 1).contiguous()

    def padded_feat_mb(self, curr_feats):
        """
        curr_feats is a bsz-len list of nrows-len list of features
        returns:
          a bsz x max_nrows x nfeats tensor
        """
        max_rows = max(len(feats) for feats in curr_feats)
        pad_idx = self.dictionary.word2idx["<pad>"]
        nfeats = len(curr_feats[0][0])
        for feats in curr_feats:
            if len(feats) < max_rows:
                [feats.append([pad_idx for _ in range(nfeats)])
                 for _ in range(max_rows - len(feats))]
        return torch.LongTensor(curr_feats)


    def padded_inp_mb(self, curr_inps):
        """
        curr_inps is a bsz-len list of seqlen-len list of nlocs-len list of features
        returns:
          a bsz x seqlen x max_nlocs x nfeats tensor
        """
        max_rows = max(len(feats) for seq in curr_inps for feats in seq)
        nfeats = len(curr_inps[0][0][0])
        for seq in curr_inps:
            for feats in seq:
                if len(feats) < max_rows:
                    # pick random rows
                    randidxs = [random.randint(0, len(feats)-1)
                                for _ in range(max_rows - len(feats))]
                    [feats.append(feats[ridx]) for ridx in randidxs]
        return torch.LongTensor(curr_inps)


    def minibatchify(self, sents, labels, feats, locs, inps, en_src, en_targ, temp1, bsz, pre_compute=True):
        """
        this should result in there never being any padding.
        each minibatch is:
          (seqlen x bsz, bsz-length list of lists of (start, end, label) constraints,
           bsz x nfields x nfeats, seqlen x bsz x max_locs, seqlen x bsz x max_locs x nfeats)
        """

        # sort in ascending order
        sents, _, sorted_idxs = zip(*sorted(zip(sents, feats, range(len(sents))), key=lambda x: (len(x[0]), len(x[1]))))
        minibatches, mb2linenos = [], []
        curr_batch, curr_labels, curr_feats, curr_locs, curr_linenos, curr_ensrc, curr_entarg = \
            [], [], [], [], [], [], []
        # curr_src_wrd2fields = []
        curr_inps = []
        curr_len = len(sents[0])
        for i in range(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz: # we're done
                # minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                #                     curr_labels, self.padded_feat_mb(curr_feats),
                #                     self.padded_loc_mb(curr_locs),
                #                     self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous(),
                #                     curr_ensrc, curr_entarg, curr_src_wrd2fields ))

                if pre_compute:
                    special_inps = self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()
                    # special_pr_mask = gen_detailed_tgt_mask_pre(special_inps, self.L, self.non_field, self.maxseqlen)
                    seqlen = special_inps.size(0)
                    special_pr_mask = make_bwd_idxs_pre2(self.L, seqlen, self.K, curr_labels)
                    minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                        curr_labels, self.padded_feat_mb(curr_feats),
                                        self.padded_loc_mb(curr_locs),
                                        special_inps,
                                        curr_ensrc, curr_entarg, special_pr_mask))
                else:

                    minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                        curr_labels, self.padded_feat_mb(curr_feats),
                                        self.padded_loc_mb(curr_locs),
                                        self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous(),
                                        curr_ensrc, curr_entarg, None))

                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                curr_labels = [labels[sorted_idxs[i]]]
                curr_feats = [feats[sorted_idxs[i]]]
                curr_locs = [locs[sorted_idxs[i]]]
                curr_inps = [inps[sorted_idxs[i]]]
                if not self.mem_eff_flag:
                    pass
                    # curr_ensrc.append(en_src[sorted_idxs[i]])
                curr_entarg = [en_targ[sorted_idxs[i]]]
                # curr_src_wrd2fields = [src_wrd2fields[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                curr_labels.append(labels[sorted_idxs[i]])
                curr_feats.append(feats[sorted_idxs[i]])
                curr_locs.append(locs[sorted_idxs[i]])
                curr_inps.append(inps[sorted_idxs[i]])
                if not self.mem_eff_flag:
                    pass
                    # curr_ensrc.append(en_src[sorted_idxs[i]])
                curr_entarg.append(en_targ[sorted_idxs[i]])
                # curr_src_wrd2fields.append(src_wrd2fields[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
        # catch last
        if len(curr_batch) > 0:
            # minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
            #                     curr_labels, self.padded_feat_mb(curr_feats),
            #                     self.padded_loc_mb(curr_locs),
            #                     self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous(),
            #                     curr_ensrc, curr_entarg, curr_src_wrd2fields))

            if pre_compute:
                special_inps = self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()
                # special_pr_mask = gen_detailed_tgt_mask_pre(special_inps, self.L, self.non_field, self.maxseqlen)
                seqlen = special_inps.size(0)
                special_pr_mask = make_bwd_idxs_pre2(self.L, seqlen, self.K, curr_labels)
                minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                    curr_labels, self.padded_feat_mb(curr_feats),
                                    self.padded_loc_mb(curr_locs),
                                    special_inps,
                                    curr_ensrc, curr_entarg, special_pr_mask))
            else:

                minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                    curr_labels, self.padded_feat_mb(curr_feats),
                                    self.padded_loc_mb(curr_locs),
                                    self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous(),
                                    curr_ensrc, curr_entarg, None))
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos









    def purity_check(self, insent, w2i):
        print('purify')
        for ind, x in enumerate(insent):
            if len(x) == 2:
                print(x)
                cand1, cand2 = x[0], x[1]
                # get respect positions.
                pos1 = self.dictionary.idx2word[cand1[2]]
                pos2 = self.dictionary.idx2word[cand2[2]]
                print('pos1=', pos1, 'pos2=', pos2)

                # get respective stop condition.
                if cand1[3] == w2i["<stop>"]:
                    stop1 = 1
                elif cand1[3] == w2i["<go>"]:
                    stop1 = 0
                else:
                    stop1 = -1
                if cand2[3] == w2i["<stop>"]:
                    stop2 = 1
                elif cand2[3] == w2i["<go>"]:
                    stop2 = 0
                else:
                    stop2 = -1

                if cand1[1] != cand2[1]:  # if the field name is different.
                    print('different field name')
                    # evaluate cand1.
                    # check left.
                    if pos1 >= 2 and stop1 == 1:  # have left and no right.
                        flag1 = False
                        for elem in insent[ind - 1]:
                            if elem[1] == cand1[1] and elem[2] == w2i[pos1 - 1]:
                                flag1 = True
                                break

                        flag2 = False
                        if ind == len(insent):
                            flag2 = True
                        else:
                            for elem in insent[ind + 1]:
                                if elem[1] != cand1[1]:
                                    flag2 = True
                                    break

                    elif stop1 == 1:  # no left and no right.
                        flag2 = False
                        if ind == len(insent):
                            flag2 = True
                        else:
                            for elem in insent[ind + 1]:
                                if elem[1] != cand1[1]:
                                    flag2 = True
                                    break

                    elif pos1 >= 2:  # have left and have right
                        flag1 = False
                        if pos1 >= 2 and stop1 == 1:  # have left and no right.
                            for elem in insent[ind - 1]:
                                if elem[1] == cand1[1] and elem[2] == w2i[pos1 - 1]:
                                    flag1 = True
                                    break

                    else:  # no left and have right
                        flag1 = False
                        for elem in insent[ind + 1]:
                            if elem[1] == cand1[1] and elem[2] == w2i[pos1 + 1]:
                                flag1 = True
                                break

                    if flag1 and flag2:
                        print('cand1 is good')
                        insent[ind] = [cand1]

                elif cand1[2] != cand2[2]:
                    print(cand1, cand2)
                    if ind < len(insent) - 1:
                        nxt = self.dictionary.idx2word[insent[ind + 1][0][2]]
                        print(nxt)
                        if nxt == pos1 + 1:
                            insent[ind] = [cand1]
                            print('pick cand1')
                        elif nxt == pos2 + 1:
                            insent[ind] = [cand2]
                            print('pick cand2')
                        else:
                            print('not sure')
                    print('soln', insent[ind], x)



    def pure_label_lst(self, label_lst, copied, inp_sent, w2i, temp_dict):
        # print(self.field_names)
        # print(temp_dict)
        # print(e2e_keys)
        # print(label_lst)
        # print(inp_sent)
        curr_state = -1
        curr_state_idx = -1
        idx_label = 0
        for idx, elem in enumerate(inp_sent):
            if idx >= label_lst[idx_label][1]:
                idx_label += 1
            if idx < label_lst[idx_label][0]:
                curr_state = -1
                curr_state_idx = -1
            else:
                assert idx < label_lst[idx_label][1]
                curr_state = label_lst[idx_label][2]
                if idx == label_lst[idx_label][0]:
                    curr_state_idx = 1
                else:
                    curr_state_idx += 1

            # print(idx, curr_state, curr_state_idx, label_lst[idx_label])

            if len(elem) == 1:
                pass
            else:
                flag=False
                # print('>= 2 elems')
                if curr_state == -1:
                    # print('curr state = -1')
                    for temp in elem:
                        # print(temp[1], w2i["<ncf1>"], 'the field is', self.dictionary.idx2word[temp[1]],
                        #       'the word is', self.dictionary.idx2word[temp[0]],
                        #       'the order is', self.dictionary.idx2word[temp[2]])
                        if temp[1] == w2i["<ncf1>"]:
                            inp_sent[idx] = [temp]
                            copied[idx] = [-1]
                            flag=True
                            break


                else:
                    # print('curr state = {}'.format(curr_state))
                    for temp in elem:
                        # print(temp, temp_dict[curr_state], temp[1], curr_state)
                        # print(temp[2], curr_state_idx, w2i[curr_state_idx])
                        if temp[1] == temp_dict[curr_state] and temp[2] == w2i[curr_state_idx]:
                            inp_sent[idx] = [temp]
                            flag = True
                            break
                if flag:
                    pass
                else:
                    inp_sent[idx] = [elem[0]]

        return inp_sent


