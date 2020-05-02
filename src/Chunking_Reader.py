from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest
import itertools
from collections import Counter
import torch
import os

def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False


class Data_Loader_toy:
    def __init__(self, data, bsz, test=False, max_len = 50, thresh=1):
        self.data_path = data
        self.bsz = bsz
        self.test = test
        self.max_len = max_len
        self.thresh = thresh

        self.get_data()

    def get_data(self):
        base_path = self.data_path
        path_train = os.path.join(base_path, "ptb.train.txt")
        train_dataset = list(self.reader(path_train))

        '''get vocab'''
        self.get_vocab(train_dataset)

        if not self.test:
            path_dev = os.path.join(base_path, "ptb.valid.txt")
            validation_dataset = list(self.reader(path_dev))
        else:
            path_test = os.path.join(base_path, "ptb.test.txt")
            test_dataset = list(self.reader(path_test))

        '''index the tokens'''
        token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst = self.get_idx(train_dataset)

        print(len(token_lst), len(pos_lst), len(chunk_lst), len(tokenstr_lst))

        self.train, self.train_idx = self.batchify(token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst, self.bsz)


        if not self.test:
            token_lst_dev, pos_lst_dev, chunk_lst_dev, ner_lst_dev, tokenstr_lst_dev = self.get_idx(validation_dataset)

            self.valid, self.valid_idx = self.batchify(token_lst_dev, pos_lst_dev, chunk_lst_dev,
                                                              ner_lst_dev, tokenstr_lst_dev, self.bsz)
        else:
            token_lst_dev, pos_lst_dev, chunk_lst_dev, ner_lst_dev, tokenstr_lst_dev = self.get_idx(test_dataset)

            self.test, self.test_idx = self.batchify(token_lst_dev, pos_lst_dev, chunk_lst_dev,
                                                              ner_lst_dev, tokenstr_lst_dev, self.bsz)


        return

    def reader(self, file_path):
        with open(file_path, "r") as data_file:
            print("Reading instances from lines in file at: %s", file_path)

            for line in data_file:
                if not _is_divider(line):
                    fields = line.strip().split()
                    yield fields

            # # Group into alternative divider / sentence chunks.
            # for is_divider, lines in itertools.groupby(data_file, _is_divider):
            #     # Ignore the divider chunks, so that `lines` corresponds to the words
            #     # of a single sentence.
            #     if not is_divider:
            #         fields = [line.strip().split() for line in lines]
            #         # unzipping trick returns tuples, but our Fields need lists
            #         fields = [list(field) for field in zip(*fields)]
            #         tokens, pos_tags, chunk_tags, ner_tags = fields
            #         # TextField requires ``Token`` objects
            #         # tokens = [token for token in tokens_]
            #
            #         yield tokens, pos_tags, chunk_tags, ner_tags

    def batchify(self, token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst, bsz):

        sents, sorted_idxs = zip(*sorted(zip(token_lst, range(len(token_lst))), key=lambda x: len(x[0])))
        minibatches, mb2linenos = [], []

        curr_batch = []
        curr_str = []
        curr_pos, curr_chunk, curr_ner, curr_linenos = [], [], [], []

        curr_len = len(sents[0])
        for i in range(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz:  # we're done
                minibatches.append((torch.LongTensor(curr_batch),
                                    None, None, None,
                                    curr_str)
                                   )

                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                # curr_pos = [pos_lst[sorted_idxs[i]]]
                # curr_chunk = [chunk_lst[sorted_idxs[i]]]
                # curr_ner = [ner_lst[sorted_idxs[i]]]
                curr_str = [tokenstr_lst[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                # curr_pos.append(pos_lst[sorted_idxs[i]])
                # curr_chunk.append(chunk_lst[sorted_idxs[i]])
                # curr_ner.append(ner_lst[sorted_idxs[i]])
                curr_str.append(tokenstr_lst[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
        # catch last
        if len(curr_batch) > 0:
            minibatches.append((torch.LongTensor(curr_batch),
                                None, None, None,
                                curr_str)
                               )
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos

    def get_idx(self, dataset, add_eos=True):
        unk = self.vocab2idx['<unk>']
        eos = self.vocab2idx['<eos>']
        # chk_eos = self.chunk2idx['<eos_chk>']
        # ner_eos = self.ner2idx['<eos_ner>']
        # pos_eos = self.pos2idx['<eos_pos>']
        token_lst = []
        pos_lst = []
        chunk_lst = []
        ner_lst = []
        tokenstr_lst = []

        for elem in dataset:
            token  = elem
            # token, pos, chunk, ner = elem
            token_idx = [self.vocab2idx[x] if x in self.vocab2idx else unk for x in token]
            # pos_idx = [self.pos2idx[x] for x in pos]
            # chunk_idx = [self.chunk2idx[x] for x in chunk]
            # ner_idx = [self.ner2idx[x] for x in ner]

            if add_eos:
                token_idx = token_idx + [eos]
                # pos_idx = pos_idx + [pos_eos]
                # chunk_idx = chunk_idx + [chk_eos]
                # ner_idx = ner_idx + [ner_eos]
            token_str = token + ['<eos>']

            token_lst.append(token_idx)
            # pos_lst.append(pos_idx)
            # chunk_lst.append(chunk_idx)
            # ner_lst.append(ner_idx)
            tokenstr_lst.append(token_str)

        return token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst

    def get_vocab(self, train_dataset):
        vocab_counter = Counter()
        pos_counter = Counter()
        chunk_counter = Counter()
        ner_counter = Counter()
        for elem in train_dataset:
            for token in zip(*elem):
                vocab_counter[token] += 1
                # pos_counter[pos] += 1
                # chunk_counter[chunk] += 1
                # ner_counter[ner] += 1

        # filter based on the threshold.
        del_lst = []
        for key, val in vocab_counter.items():
            if val < self.thresh:
                del_lst.append(key)
        for elem in del_lst:
            del vocab_counter[elem]

        # build the vocab token2idx
        self.idx2vocab = ['<unk>', '<pad>', '<bos>', '<eos>'] + list(vocab_counter.keys())
        self.vocab2idx = {x: i for i, x in enumerate(self.idx2vocab)}


        return


class Data_Loader_ptb:
    def __init__(self, data, bsz, L, K, test=False, max_len = 50, thresh=1):
        self.data_path = data
        self.bsz = bsz
        self.test = test
        self.max_len = max_len
        self.thresh = thresh
        self.L = L
        self.K = K

        self.get_data()

    def get_data(self):
        base_path = self.data_path
        # path_train = os.path.join(base_path, "pr_ptb2.train")
        path_train = os.path.join(base_path, "ptb.tag.train.txt")
        train_dataset, label_dataset = self.reader_pr(path_train)

        '''get vocab'''
        self.get_vocab(train_dataset, label_dataset)

        if not self.test:
            # path_dev = os.path.join(base_path, "ptb.valid.txt")
            path_dev = os.path.join(base_path, "ptb.tag.dev.txt")

            validation_dataset, valid_labels = list(self.reader_pr(path_dev))
        else:
            # path_test = os.path.join(base_path, "ptb.test.txt")
            path_test = os.path.join(base_path, "ptb.tag.test.txt")
            test_dataset, test_labels = list(self.reader_pr(path_test))

        '''index the tokens'''
        token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst = self.get_idx(train_dataset)

        print(len(token_lst), len(pos_lst), len(chunk_lst), len(tokenstr_lst))

        self.train, self.train_idx = self.batchify_pr(token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst, self.bsz, label_dataset)


        if not self.test:
            token_lst_dev, pos_lst_dev, chunk_lst_dev, ner_lst_dev, tokenstr_lst_dev = self.get_idx(validation_dataset)

            self.valid, self.valid_idx = self.batchify_pr(token_lst_dev, pos_lst_dev, chunk_lst_dev,
                                                              ner_lst_dev, tokenstr_lst_dev, self.bsz, valid_labels)
        else:
            token_lst_dev, pos_lst_dev, chunk_lst_dev, ner_lst_dev, tokenstr_lst_dev = self.get_idx(test_dataset)

            self.test, self.test_idx = self.batchify_pr(token_lst_dev, pos_lst_dev, chunk_lst_dev,
                                                              ner_lst_dev, tokenstr_lst_dev, self.bsz, test_labels)


        return


    def reader_pr(self, file_path):
        fields_lst = []
        labels_lst = []
        with open(file_path, "r") as data_file:
            print("Reading instances from lines in file at: %s", file_path)

            for line_full in data_file:
                if not _is_divider(line_full):
                    line, labels = line_full.strip().split('|||')
                    fields = line.split()

                    if len(labels) == 0:
                        fields_lst.append(fields)
                        labels_lst.append([])
                        continue

                    labels = labels.split(' ')
                    # labels_new = []
                    # for label in labels:
                    #     a,b,c = label.split(',')
                    #     a = int(a)
                    #     b = int(b)
                    #     c = int(c)
                    #     labels_new.append((a,b,c))
                    # labels_lst.append(labels_new)
                    fields_lst.append(fields)
                    labels_lst.append(labels)

        return fields_lst, labels_lst

    def reader(self, file_path):
        with open(file_path, "r") as data_file:
            print("Reading instances from lines in file at: %s", file_path)

            for line in data_file:
                if not _is_divider(line):
                    fields = line.strip().split()
                    yield fields



            # # Group into alternative divider / sentence chunks.
            # for is_divider, lines in itertools.groupby(data_file, _is_divider):
            #     # Ignore the divider chunks, so that `lines` corresponds to the words
            #     # of a single sentence.
            #     if not is_divider:
            #         fields = [line.strip().split() for line in lines]
            #         # unzipping trick returns tuples, but our Fields need lists
            #         fields = [list(field) for field in zip(*fields)]
            #         tokens, pos_tags, chunk_tags, ner_tags = fields
            #         # TextField requires ``Token`` objects
            #         # tokens = [token for token in tokens_]
            #
            #         yield tokens, pos_tags, chunk_tags, ner_tags


    def get_pr_mask(self, sent, tr_sent):
        temp = torch.LongTensor(self.L, len(tr_sent[0]), len(tr_sent), self.K).fill_(0)

        for idx, x in enumerate(sent):
            for idx2, pos in enumerate(x):
                if pos not in self.pos_vocab:
                    print(pos)
                    temp[0, idx2, idx, self.pos_vocab['PUNCT']] = 1
                else:
                    temp[0, idx2, idx, self.pos_vocab[pos]] = 1
            temp[0, -1, idx, self.eos_pos] = 1

        return temp


    def get_pr_mask2(self, sent, tr_sent):
        # find the boundary.
        #  L x  seqlen  x  bsz x K

        temp = torch.LongTensor(self.L, len(tr_sent[0]), len(tr_sent), self.K).fill_(0)

        for idx, x in enumerate(sent):
            for (a, b, c) in x:
                if b - a >= self.L:
                    prev = a
                    remain_len = b - a
                    while remain_len > 0:
                        temp[min(self.L, remain_len)-1, prev, idx, c] = 1
                        prev = a + min(self.L, remain_len)
                        remain_len = remain_len - min(self.L, remain_len)
                else:
                    temp[b-a-1, a, idx, c] = 1


        return temp

    def batchify_pr(self, token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst, bsz, pr_labels):

        sents, sorted_idxs = zip(*sorted(zip(token_lst, range(len(token_lst))), key=lambda x: len(x[0])))
        minibatches, mb2linenos = [], []

        curr_batch = []
        curr_str = []
        curr_pos, curr_chunk, curr_ner, curr_linenos = [], [], [], []
        curr_labels = []

        curr_len = len(sents[0])
        for i in range(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz:  # we're done
                minibatches.append((torch.LongTensor(curr_batch),
                                    curr_labels, self.get_pr_mask(curr_labels, curr_batch), None,
                                    curr_str)
                                   )

                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                curr_labels = [pr_labels[sorted_idxs[i]]]
                # curr_chunk = [chunk_lst[sorted_idxs[i]]]
                # curr_ner = [ner_lst[sorted_idxs[i]]]
                curr_str = [tokenstr_lst[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                curr_labels.append(pr_labels[sorted_idxs[i]])
                # curr_pos.append(pos_lst[sorted_idxs[i]])
                # curr_chunk.append(chunk_lst[sorted_idxs[i]])
                # curr_ner.append(ner_lst[sorted_idxs[i]])
                curr_str.append(tokenstr_lst[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
        # catch last
        if len(curr_batch) > 0:
            minibatches.append((torch.LongTensor(curr_batch),
                               curr_labels,  self.get_pr_mask(curr_labels, curr_batch), None,
                                curr_str)
                               )
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos

    def batchify(self, token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst, bsz):

        sents, sorted_idxs = zip(*sorted(zip(token_lst, range(len(token_lst))), key=lambda x: len(x[0])))
        minibatches, mb2linenos = [], []

        curr_batch = []
        curr_str = []
        curr_pos, curr_chunk, curr_ner, curr_linenos = [], [], [], []

        curr_len = len(sents[0])
        for i in range(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz:  # we're done
                minibatches.append((torch.LongTensor(curr_batch),
                                    None, None, None,
                                    curr_str)
                                   )

                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                # curr_pos = [pos_lst[sorted_idxs[i]]]
                # curr_chunk = [chunk_lst[sorted_idxs[i]]]
                # curr_ner = [ner_lst[sorted_idxs[i]]]
                curr_str = [tokenstr_lst[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                # curr_pos.append(pos_lst[sorted_idxs[i]])
                # curr_chunk.append(chunk_lst[sorted_idxs[i]])
                # curr_ner.append(ner_lst[sorted_idxs[i]])
                curr_str.append(tokenstr_lst[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
        # catch last
        if len(curr_batch) > 0:
            minibatches.append((torch.LongTensor(curr_batch),
                                None, None, None,
                                curr_str)
                               )
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos

    def get_idx(self, dataset, add_eos=True):
        unk = self.vocab2idx['<unk>']
        eos = self.vocab2idx['<eos>']
        # chk_eos = self.chunk2idx['<eos_chk>']
        # ner_eos = self.ner2idx['<eos_ner>']
        # pos_eos = self.pos2idx['<eos_pos>']
        token_lst = []
        pos_lst = []
        chunk_lst = []
        ner_lst = []
        tokenstr_lst = []

        for elem in dataset:
            token  = elem
            # token, pos, chunk, ner = elem
            token_idx = [self.vocab2idx[x] if x in self.vocab2idx else unk for x in token]
            # pos_idx = [self.pos2idx[x] for x in pos]
            # chunk_idx = [self.chunk2idx[x] for x in chunk]
            # ner_idx = [self.ner2idx[x] for x in ner]

            if add_eos:
                token_idx = token_idx #+ [eos]
                # pos_idx = pos_idx + [pos_eos]
                # chunk_idx = chunk_idx + [chk_eos]
                # ner_idx = ner_idx + [ner_eos]
            token_str = token #+ ['<eos>']

            token_lst.append(token_idx)
            # pos_lst.append(pos_idx)
            # chunk_lst.append(chunk_idx)
            # ner_lst.append(ner_idx)
            tokenstr_lst.append(token_str)

        return token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst

    def get_vocab(self, train_dataset, tagset):
        vocab_counter = Counter()
        pos_counter = Counter()
        chunk_counter = Counter()
        ner_counter = Counter()
        self.pos_vocab = {}
        for elem in train_dataset:
            for token in elem:
                vocab_counter[token] += 1

        for elem in tagset:
            for pos in elem:
                if pos not in self.pos_vocab:
                    self.pos_vocab[pos] = len(self.pos_vocab)
        self.pos_vocab['EOS'] = len(self.pos_vocab)

        self.eos_pos = self.pos_vocab['EOS']
                # pos_counter[pos] += 1
                # chunk_counter[chunk] += 1
                # ner_counter[ner] += 1

        # filter based on the threshold.
        del_lst = []
        for key, val in vocab_counter.items():
            if val < self.thresh:
                del_lst.append(key)
        for elem in del_lst:
            del vocab_counter[elem]

        # build the vocab token2idx
        tempp = list(vocab_counter.keys())
        tempp.remove('<unk>')
        tempp.remove('<eos>')
        self.idx2vocab = ['<unk>', '<pad>', '<bos>', '<eos>'] + tempp
        self.vocab2idx = {x: i for i, x in enumerate(self.idx2vocab)}


        # # build other X2idx and idx2X maps
        # self.idx2pos = ['<unk_pos>', '<pad_pos>', '<bos_pos>', '<eos_pos>'] + list(pos_counter.keys())
        # self.idx2chunk = ['<unk_chk>', '<pad_chk>', '<bos_chk>', '<eos_chk>'] + list(chunk_counter.keys())
        # self.idx2ner = ['<unk_ner>', '<pad_ner>', '<bos_ner>', '<eos_ner>'] + list(ner_counter.keys())

        # self.pos2idx = {x: i for i, x in enumerate(self.idx2pos)}
        # self.chunk2idx = {x: i for i, x in enumerate(self.idx2chunk)}
        # self.ner2idx = {x: i for i, x in enumerate(self.idx2ner)}

        return



class Data_Loader_Chunk:

    def __init__(self, data, bsz, test=False, max_len=50, thresh=1 ):
        self.data_path = data
        self.bsz = bsz
        self.test = test
        self.max_len = max_len
        self.thresh = thresh

        self.get_data()

    def reader(self, file_path):
        with open(file_path, "r") as data_file:
            print("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens, pos_tags, chunk_tags, ner_tags = fields
                    # TextField requires ``Token`` objects
                    # tokens = [token for token in tokens_]

                    yield tokens, pos_tags, chunk_tags, ner_tags

    def get_data(self):
        base_path = self.data_path
        path_train = os.path.join(base_path, "eng.train")
        # path_train = '/Users/xiangli/Desktop/Sasha/FSA-RNN/data/conll2003/eng.train'
        train_dataset = list(self.reader(path_train))

        '''get vocab'''
        self.get_vocab(train_dataset)

        if not self.test:
            path_dev = os.path.join(base_path, "eng.testa")
            # path_dev = '/Users/xiangli/Desktop/Sasha/FSA-RNN/data/conll2003/eng.testa'
            validation_dataset = list(self.reader(path_dev))
        else:
            path_test = os.path.join(base_path, "eng.testb")
            # path_test = '/Users/xiangli/Desktop/Sasha/FSA-RNN/data/conll2003/eng.testb'
            test_dataset = list(self.reader(path_test))

        '''index the tokens'''
        token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst = self.get_idx(train_dataset)

        print(len(token_lst), len(pos_lst), len(chunk_lst), len(tokenstr_lst))

        self.train, self.train_idx = self.batchify(token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst, self.bsz)


        if not self.test:
            token_lst_dev, pos_lst_dev, chunk_lst_dev, ner_lst_dev, tokenstr_lst_dev = self.get_idx(validation_dataset)

            self.valid, self.valid_idx = self.batchify(token_lst_dev, pos_lst_dev, chunk_lst_dev,
                                                              ner_lst_dev, tokenstr_lst_dev, self.bsz)
        else:
            token_lst_dev, pos_lst_dev, chunk_lst_dev, ner_lst_dev, tokenstr_lst_dev = self.get_idx(test_dataset)

            self.test, self.test_idx = self.batchify(token_lst_dev, pos_lst_dev, chunk_lst_dev,
                                                              ner_lst_dev, tokenstr_lst_dev, self.bsz)


        return

        ''' put to batch '''

    def batchify(self, token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst, bsz):

        sents, sorted_idxs = zip(*sorted(zip(token_lst, range(len(token_lst))), key=lambda x: len(x[0])))
        minibatches, mb2linenos = [], []

        curr_batch, curr_pos, curr_chunk, curr_ner, curr_linenos, curr_str  = \
            [], [], [], [], [], []

        curr_len = len(sents[0])
        for i in range(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz:  # we're done
                minibatches.append((torch.LongTensor(curr_batch),
                                    torch.LongTensor(curr_pos),
                                    torch.LongTensor(curr_chunk),
                                    torch.LongTensor(curr_ner),
                                    curr_str)
                                   )

                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                curr_pos = [pos_lst[sorted_idxs[i]]]
                curr_chunk = [chunk_lst[sorted_idxs[i]]]
                curr_ner = [ner_lst[sorted_idxs[i]]]
                curr_str = [tokenstr_lst[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                curr_pos.append(pos_lst[sorted_idxs[i]])
                curr_chunk.append(chunk_lst[sorted_idxs[i]])
                curr_ner.append(ner_lst[sorted_idxs[i]])
                curr_str.append(tokenstr_lst[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
        # catch last
        if len(curr_batch) > 0:
            minibatches.append((torch.LongTensor(curr_batch),
                                torch.LongTensor(curr_pos),
                                torch.LongTensor(curr_chunk),
                                torch.LongTensor(curr_ner),
                                curr_str)
                               )
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos


    def get_idx(self, dataset, add_eos=True):
        unk = self.vocab2idx['<unk>']
        eos = self.vocab2idx['<eos>']
        chk_eos = self.chunk2idx['<eos_chk>']
        ner_eos = self.ner2idx['<eos_ner>']
        pos_eos = self.pos2idx['<eos_pos>']
        token_lst = []
        pos_lst = []
        chunk_lst = []
        ner_lst = []
        tokenstr_lst = []


        for elem in dataset:
            token, pos, chunk, ner = elem
            token_idx = [self.vocab2idx[x] if x in self.vocab2idx else unk for x in token]
            pos_idx = [self.pos2idx[x] for x in pos]
            chunk_idx = [self.chunk2idx[x] for x in chunk]
            ner_idx = [self.ner2idx[x] for x in ner]

            if add_eos:
                token_idx = token_idx + [eos]
                pos_idx = pos_idx + [pos_eos]
                chunk_idx = chunk_idx + [chk_eos]
                ner_idx = ner_idx + [ner_eos]
            token_str = token + ['<eos>']

            token_lst.append(token_idx)
            pos_lst.append(pos_idx)
            chunk_lst.append(chunk_idx)
            ner_lst.append(ner_idx)
            tokenstr_lst.append(token_str)

        return token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst



    def get_vocab(self, train_dataset, threshold=1):
        vocab_counter = Counter()
        pos_counter = Counter()
        chunk_counter = Counter()
        ner_counter = Counter()
        for elem in train_dataset:
            for token, pos, chunk, ner in zip(*elem):
                vocab_counter[token] += 1
                pos_counter[pos] += 1
                chunk_counter[chunk] += 1
                ner_counter[ner] += 1


        # filter based on the threshold.
        del_lst = []
        for key, val in vocab_counter.items():
            if val < threshold:
                del_lst.append(key)
        for elem in del_lst:
            del vocab_counter[elem]

        # build the vocab token2idx
        self.idx2vocab = ['<unk>', '<pad>', '<bos>', '<eos>'] + list(vocab_counter.keys())
        self.vocab2idx = {x:i for i, x in enumerate(self.idx2vocab)}

        # build other X2idx and idx2X maps
        self.idx2pos = ['<unk_pos>', '<pad_pos>', '<bos_pos>', '<eos_pos>'] + list(pos_counter.keys())
        self.idx2chunk = ['<unk_chk>', '<pad_chk>', '<bos_chk>', '<eos_chk>'] + list(chunk_counter.keys())
        self.idx2ner = ['<unk_ner>', '<pad_ner>', '<bos_ner>', '<eos_ner>'] + list(ner_counter.keys())

        self.pos2idx = {x:i for i, x in enumerate(self.idx2pos)}
        self.chunk2idx = {x: i for i, x in enumerate(self.idx2chunk)}
        self.ner2idx = {x: i for i, x in enumerate(self.idx2ner)}

        return





#
#
#
#
# lazy = False
# # coding_scheme = 'IOB1'
# # conll_reader = Conll2003DatasetReader(lazy=lazy, coding_scheme=coding_scheme)
#
# path_train = '/Users/xiangli/Desktop/Sasha/FSA-RNN/data/conll2003/eng.train'
# train_dataset = conll_reader.read(path_train)
#
# path_dev = '/Users/xiangli/Desktop/Sasha/FSA-RNN/data/conll2003/eng.testa'
# validation_dataset = conll_reader.read(path_dev)
#
# instances = ensure_list(train_dataset)
# print(len(instances))
#
#
# vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
#
#
# iterator = BucketIterator(batch_size=40, sorting_keys=[("tokens", "num_tokens")])
# iterator.index_with(vocab)
#
# batches = list(iterator._create_batches(train_dataset, shuffle=False))
#
# for batch in batches:
#     for x in batch.instances:
#         print(x['tokens'], x['tags'])
