"""
this file modified from the word_language_model example
"""
import os
import torch
import sys

from collections import Counter, defaultdict

from utils_data import get_wikibio_poswrds, get_e2e_poswrds

import random
random.seed(1111)

#punctuation = set(['.', '!', ',', ';', ':', '?', '--', '-rrb-', '-lrb-'])
punctuation = set() # i don't know why i was so worried about punctuation

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
    def __init__(self, path, bsz, thresh=5, add_bos=False, add_eos=False,
                 test=False):
        self.dictionary = Dictionary()
        self.bsz = bsz
        self.wiki = "wb" in path

        temp_name = 'train'
        temp_val_name = 'valid'
        if not self.wiki:
            temp_test_name = 'test_full1'
        else:
            temp_test_name = 'test'

        if not self.wiki:
            self.mem_eff_flag = False
            print('labeled_data mem status is: ', self.mem_eff_flag)
        else:
            self.mem_eff_flag = True
            print('labeled_data mem status is: ', self.mem_eff_flag)


        train_src = os.path.join(path, "src_{}.txt".format(temp_name))
        sys.stdout.flush()
        if thresh > 0:
            self.get_vocabs(os.path.join(path, '{}.txt'.format(temp_name)), train_src, thresh=thresh)
            self.ngen_types = len(self.genset) + 4 # assuming didn't encounter any special tokens
            add_to_dict = False
        else:
            add_to_dict = True
        # trsent = training sentence : each word is replaced with the corresponding in dex.
        # trlabels = the human-annotated segmentation
        # trfeats = the training features from the src, each token is represented by [word, category K, index in the cat]
        # trlocs = the copied location, if a token is not copied, it's replaced by a -1
        # inps = the entire feature of the generated sentence. If it is copied from the src, then it will take on trfeats
        # if it is newly generated, then it will take on ixts own feature: < word-index, fnc1, fnc2, fnc3 > .
        print('finished _vocab section ')
        print ("using vocabulary of size:", len(self.dictionary))
        print (self.ngen_types, "gen word types")
        sys.stdout.flush()


        if not test:
            trsents, trlabels, trfeats, trlocs, inps, tren_src, tren_targ, trsrc_wrd2fields = self.tokenize(
                os.path.join(path, '{}.txt'.format(temp_name)), train_src, add_to_dict=add_to_dict,
                add_bos=add_bos, add_eos=add_eos)

            self.train, self.train_mb2linenos = self.minibatchify(
                trsents, trlabels, trfeats, trlocs, inps, tren_src, tren_targ, trsrc_wrd2fields, bsz) # list of minibatches

        if (os.path.isfile(os.path.join(path, '{}.txt'.format(temp_val_name)))
                or os.path.isfile(os.path.join(path, 'test.txt'))):
            if not test:
                val_src = os.path.join(path, "src_{}.txt".format(temp_val_name))
                vsents, vlabels, vfeats, vlocs, vinps, ven_src, ven_targ, vsrc_wrd2fields = self.tokenize(
                    os.path.join(path, '{}.txt'.format(temp_val_name)), val_src, add_to_dict=False,
                    add_bos=add_bos, add_eos=add_eos)
            else:
                print ("using test data and whatnot....")
                test_src = os.path.join(path, "src_{}.txt".format(temp_test_name))
                vsents, vlabels, vfeats, vlocs, vinps, ven_src, ven_targ, vsrc_wrd2fields = self.tokenize(
                    os.path.join(path, '{}.txt'.format(temp_test_name)), test_src, add_to_dict=False,
                    add_bos=add_bos, add_eos=add_eos)
            self.valid, self.val_mb2linenos = self.minibatchify(
                vsents, vlabels, vfeats, vlocs, vinps, ven_src, ven_targ,vsrc_wrd2fields, bsz)
        else:
            print('no dev or test data presented. ')
        print('loaded dataset.')
        sys.stdout.flush()


    def get_vocabs(self, path, src_path, thresh=2):
        """unks words occurring <= thresh times"""
        tgt_voc = Counter()
        print(path)
        assert os.path.exists(path)

        field_voc = Counter()
        linewords = []
        with open(src_path, 'r') as f:
            for line in f:
                tokes = line.strip().split()
                if self.wiki:
                    fields = get_wikibio_poswrds(tokes) #key, pos -> wrd
                else:
                    fields = get_e2e_poswrds(tokes) # key, pos -> wrd
                fieldvals = fields.values()
                tgt_voc.update(fieldvals)
                linewords.append(set(wrd for wrd in fieldvals
                                     if wrd not in punctuation))
                tgt_voc.update([k for k, idx in fields])
                field_voc.update([k for k, idx, in fields])
                tgt_voc.update([idx for k, idx in fields])


        genwords = Counter()
        # Add words to the dictionary
        with open(path, 'r') as f:
            for l, line in enumerate(f):
                words, spanlabels = line.strip().split('|||')
                words = words.split()
                genwords.update([wrd for wrd in words if wrd not in linewords[l]])
                tgt_voc.update(words)

        # prune
        # N.B. it's possible a word appears enough times in total but not in genwords
        # so we need separate unking for generation
        #print "comeon", "aerobatic" in genwords
        for cntr in [tgt_voc, genwords, field_voc]:
            del_lst = []
            for k in cntr.keys():
                if cntr[k] < thresh:
                    del_lst.append(k)
            for k in del_lst:
                del cntr[k]

        self.genset = list(genwords.keys())
        tgtkeys = list(tgt_voc.keys())
        self.field_names = list(field_voc.keys())
        self.field_names2idx = {field_n:idx for idx, field_n in enumerate(self.field_names)}
        # print(self.field_names2idx)
        # make sure gen stuff is first
        tgtkeys.sort(key=lambda x: -(x in self.genset))
        self.dictionary.bulk_add(tgtkeys)
        # the previous things are [unk_word, "<pad>", "<bos>", "<eos>"] .
        # make sure we did everything right (assuming didn't encounter any special tokens)
        assert self.dictionary.idx2word[4 + len(self.genset) - 1] in self.genset
        assert self.dictionary.idx2word[4 + len(self.genset)] not in self.genset
        self.dictionary.add_word("<ncf1>", train=True)
        self.dictionary.add_word("<ncf2>", train=True)
        self.dictionary.add_word("<ncf3>", train=True)
        self.dictionary.add_word("<go>", train=True)
        self.dictionary.add_word("<stop>", train=True)



    def tokenize(self, path, src_path, add_to_dict=False, add_bos=False, add_eos=False):
        """Assumes fmt is sentence|||s1,e1,k1 s2,e2,k2 ...."""
        print(path)
        assert os.path.exists(path)


        src_feats, src_wrd2idxs, src_wrd2fields = [], [], []
        en_src = []
        en_targ = []
        w2i = self.dictionary.word2idx
        stop_idx =  w2i["<stop>"]
        go_idx =  w2i["<go>"]
        count_src = 0

        if not self.wiki:
            e2e_keys = ["name", "eatType", "food", "priceRange", "customerrating", "area", "near"]
            e2e_key2idx = dict((key, i) for i, key in enumerate(e2e_keys))
            temp_dict = {e2e_key2idx[x[1:]]: w2i[x] for x in self.field_names if
                         x[1:] in e2e_key2idx}  # from e2ekey to index of the dictionary.


        with open(src_path, 'r') as f:
            for line in f:
                tokes = line.strip().split()
                #fields = get_e2e_fields(tokes, keys=self.e2e_keys) #keyname -> list of words
                if self.wiki:
                    fields = get_wikibio_poswrds(tokes) #key, pos -> wrd
                else:
                    fields = get_e2e_poswrds(tokes) # key, pos -> wrd
                # wrd2things will be unordered
                feats, wrd2idxs, wrd2fields = [], defaultdict(list), defaultdict(list)
                # get total number of words per field
                fld_cntr = Counter([key for key, _ in fields])
                for (k, idx), wrd in fields.items():
                    if k in w2i:
                        featrow = [self.dictionary.add_word(k, add_to_dict),
                                   self.dictionary.add_word(idx, add_to_dict),
                                   self.dictionary.add_word(wrd, add_to_dict)]
                        wrd2idxs[wrd].append(len(feats))
                        #nflds = self.dictionary.add_word(fld_cntr[k], add_to_dict)
                        cheatfeat = stop_idx if fld_cntr[k] == idx else go_idx
                        wrd2fields[wrd].append((featrow[2], featrow[0], featrow[1], cheatfeat))
                        feats.append((featrow[2], featrow[0], featrow[1], cheatfeat))
                src_wrd2idxs.append(wrd2idxs) # index dict.
                src_wrd2fields.append(wrd2fields) # 4 val-elem dict
                src_feats.append(feats) # 3 elem
                if self.mem_eff_flag:
                    en_src.append(count_src)
                else:
                    en_src.append(tokes)
                count_src += 1
        sents, labels, copylocs, inps = [], [], [], []

        # Add words to the dictionary
        tgtline = 0

        with open(path, 'r') as f:
            for line in f:
                words, spanlabels = line.strip().split('|||')
                words = words.split()
                sent, copied, insent = [], [], []

                if add_bos:
                    sent.append(self.dictionary.add_word('<bos>', True))

                labetups = [tupstr.split(',') for tupstr in spanlabels.split()]
                labelist = [(int(tup[0]), int(tup[1]), int(tup[2])) for tup in labetups]

                for index1, word in enumerate(words):
                    # sent is just used for targets; we have separate inputs
                    if word in self.genset:
                        sent.append(w2i[word])
                    else:
                        sent.append(w2i["<unk>"])
                    if word not in punctuation and word in src_wrd2idxs[tgtline]:
                        copied.append(src_wrd2idxs[tgtline][word])
                        winps = [[widx, kidx, idxidx, nidx]
                                 for widx, kidx, idxidx, nidx in src_wrd2fields[tgtline][word]]

                        # the new parts. check please.
                        if not self.wiki:
                            if word == 'of':
                                winps.append([sent[-1], w2i["<ncf1>"], w2i["<ncf2>"], w2i["<ncf3>"]])

                        insent.append(winps)
                    else:
                        #assert sent[-1] < self.ngen_types
                        copied.append([-1])
                         # 1 x wrd, tokennum, totalnum
                        #insent.append([[sent[-1], w2i["<ncf1>"], w2i["<ncf2>"]]])
                        insent.append([[sent[-1], w2i["<ncf1>"], w2i["<ncf2>"], w2i["<ncf3>"]]])
                #sent.extend([self.dictionary.add_word(word, add_to_dict) for word in words])
                if add_eos:
                    sent.append(self.dictionary.add_word('<eos>', True))


                ''' purify insent '''
                # What I will do is to purify the insent results.
                # using the labellist.
                if not self.wiki:
                    self.pure_label_lst(labelist, copied, insent, w2i,  temp_dict)


                sents.append(sent)
                labels.append(labelist)
                copylocs.append(copied)
                inps.append(insent)
                if self.mem_eff_flag:
                    en_targ.append(tgtline)
                else:
                    en_targ.append(words)
                tgtline += 1

        assert len(sents) == len(labels)
        assert len(src_feats) == len(sents)
        assert len(copylocs) == len(sents)
        print(len(sents))
        if self.mem_eff_flag:
            pass
        else:
            assert len(en_targ) == len(sents)
            assert len(en_src) == len(sents)

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


    def minibatchify(self, sents, labels, feats, locs, inps, en_src, en_targ, src_wrd2fields, bsz):
        """
        this should result in there never being any padding.
        each minibatch is:
          (seqlen x bsz, bsz-length list of lists of (start, end, label) constraints,
           bsz x nfields x nfeats, seqlen x bsz x max_locs, seqlen x bsz x max_locs x nfeats)
        """
        # sort in ascending order
        sents, sorted_idxs = zip(*sorted(zip(sents, range(len(sents))), key=lambda x: len(x[0])))
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
                curr_ensrc = [en_src[sorted_idxs[i]]]
                curr_entarg = [en_targ[sorted_idxs[i]]]
                # curr_src_wrd2fields = [src_wrd2fields[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                curr_labels.append(labels[sorted_idxs[i]])
                curr_feats.append(feats[sorted_idxs[i]])
                curr_locs.append(locs[sorted_idxs[i]])
                curr_inps.append(inps[sorted_idxs[i]])
                curr_ensrc.append(en_src[sorted_idxs[i]])
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


