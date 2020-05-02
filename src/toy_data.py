import numpy as np
import  sys
import os
import  torch

'''
Want to design a data generator that first generate a sequence of 5 Zs. And then, for each Z, it will generate a 
corresponding sequence of X drawn from a particular grp.  
'''

grp = {'0':['{}-{}'.format(0, x) for x in range(1,7)], '1' : ['{}-{}'.format(1, x) for x in range(1,7)],
       '2':['{}-{}'.format(2, x) for x in range(1,7)], '3':['{}-{}'.format(3, x) for x in range(1,7)],
       '4':['{}-{}'.format(4, x) for x in range(1,7)], '5':['{}-{}'.format(5, x) for x in range(1,7)]}
min_len = 1
max_len = 5
# encode the n-gram assumption for each given state
def get_grams(mini, maxi, lst):
    set_ = []
    result = {}
    for n in range(mini, maxi):
        set_.append([tuple(lst[i:i+n]) for i in range(len(lst) - n) ])
        result[n] =  list(set(set_[-1]))
    return result

result_ngram = {}
for key, val in grp.items():
    temp = sorted(val)
    temp = temp + temp
    result_ngram[key] = get_grams(min_len, max_len, temp)




p_trans = 1 / 6

caps_lst = [x for x in grp.keys()]
avg_prob = 1 / len(caps_lst)
transition_dict = {}
p_stop = 0.4
for key in grp.keys():

    ''' double for the immediate one, and else for the remaining. '''
    transition_param = np.ones(len(caps_lst)) * avg_prob
    transition_param[int(key)] = 0
    if int(key) + 1 < len(caps_lst):
        transition_param[int(key) + 1] = p_trans
        transition_param[:int(key)] = (1-p_trans)/ (len(grp)-2)
        transition_param[int(key) + 2:] = (1-p_trans) / (len(grp)-2)
    else:
        transition_param[0] = p_trans
        transition_param[1:-1] =(1-p_trans) / (len(grp) - 2)
    # print(transition_param)
    transition_dict[key] = transition_param


def generate_z(length=10):
    # lst = ['0']
    lst = []
    while(len(lst) < length):
        # print(transition_dict[lst[-1]])
        if len(lst) == 0:
            next_ = np.random.choice([x for x in grp.keys()], 1, p=np.ones(len(grp) )/len(grp) )[0]
        else:
            next_ = np.random.choice([x for x in grp.keys()], 1, p=transition_dict[lst[-1]])[0]
        lst.append(next_)
    return lst

def generate_x(z):
    stop_ = 0
    idx = 0
    lst = []
    while(stop_ == 0):
        # generate a string by picking
        next_ = np.random.choice(grp[z], 1)[0]
        lst.append(next_)
        # check if to continue
        if idx < 2:
            stop_ = 0
        else:
            stop_ = np.random.choice([0, 1], 1, p=[1-p_stop, p_stop])[0]
        idx += 1
        if len(lst) >= 5:
            break
    return lst

def generate_x_ngram(z):
    # generate the length
    len_ = np.random.choice(list(range(min_len, max_len)), 1)[0]
    # generate the ngram sequence of that particular length
    candid = result_ngram[z][len_]
    # print(candid)
    idx = np.random.choice(list(range(len(candid))), 1)[0]
    return candid[idx]


def generate_all(length=4):
    z_lst = generate_z(length)
    result = []
    state = []
    for z in z_lst:
        gen_x = generate_x_ngram(z)
        result += gen_x
        state += len(gen_x) * [z]
    return result, state


class Data_Loader_toy:
    def __init__(self, path, bsz, L, K, test=False):
        self.data_path = path
        self.bsz = bsz
        self.test = test
        self.L = L
        self.K = K
        self.get_data()

    def get_data(self):
        base_path = self.data_path
        path_train = os.path.join(base_path, "train-toy2.txt")
        train_dataset = self.read_toy(path_train)


        if not self.test:
            path_dev = os.path.join(base_path, "dev-toy2.txt")
            validation_dataset = self.read_toy(path_dev)
        else:
            path_test = os.path.join(base_path, "test-toy2.txt")
            test_dataset = self.read_toy(path_test)

        '''index the tokens'''

        train_dict = self.pre_process(train_dataset, self.bsz)

        self.vocab2idx = train_dict['data_vocab']
        self.state_vocab = train_dict['state_vocab']
        self.train = train_dict['bsz_processed_data']
        self.train_idx = train_dict['mb2lineno']

        # token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst = self.get_idx(train_dataset)
        # print(len(token_lst), len(pos_lst), len(chunk_lst), len(tokenstr_lst))
        # self.train, self.train_idx = self.batchify(token_lst, pos_lst, chunk_lst, ner_lst, tokenstr_lst, self.bsz)

        if not self.test:
            dev_dict = self.pre_process_dev(validation_dataset, train_dict, self.bsz)
            self.valid, self.valid_idx = dev_dict['bsz_processed_data'], dev_dict['mb2lineno']
        else:
            test_dict = self.pre_process_dev(test_dataset, train_dict, self.bsz)
            self.test, self.test_idx =test_dict['bsz_processed_data'], test_dict['mb2lineno']


        return

    def read_toy(self, path):
        '''
        the toy data is formated as SENT ||| STATE
        :return: a list of training dataset
        '''
        data_lst = []
        state_lst = []
        with open(path, 'r') as f:
            for line in f:
                sent, state = line.split('|||')
                sent = sent.split() + ['<eos>']
                state = state.split()
                data_lst.append(sent)
                state_lst.append(state)
        return data_lst, state_lst

    def get_vocab(self, data_lst):
        vocab = []
        for sent in data_lst:
            for letter in sent:
                if letter not in vocab:
                    vocab.append(letter)
        vocab =  ['<bos>','<pad>'] + vocab
        vocab = {x:i for i, x in enumerate(vocab)}
        return vocab

    def pre_process(self, data, bsz=20):
        data_lst, state_lst = data
        data_vocab = self.get_vocab(data_lst)
        state_vocab = self.get_vocab(state_lst)

        self.get_pr_dict(data_vocab)
        print(data_vocab)
        print(state_vocab)

        processed_data = []
        processed_state_lst = []
        for sent, state in zip(data_lst, state_lst):
            processed_data.append([data_vocab[x] for x in sent])
            processed_state_lst.append([state_vocab[x] for x in state])

        bsz_processed_data, mb2lineno = self.batchify(processed_data, processed_state_lst, data_lst, bsz)


        result = {'data_vocab':data_vocab, 'state_vocab':state_vocab,
                  'processed_data':processed_data, 'processed_state_lst':processed_state_lst,
                  'bsz_processed_data':bsz_processed_data, 'mb2lineno':mb2lineno}

        return result

    def get_pr_dict(self, vocab):
        grp_dict = {}
        self.vocab2field = {}
        for key, val in vocab.items():
            if '-' not in key:
                grp_dict[key] = len(grp_dict)
                self.vocab2field[val] = grp_dict[key]
                continue
            grp, num = key.split('-')
            if grp not in grp_dict:
                grp_dict[grp] = len(grp_dict)
            self.vocab2field[val] = grp_dict[grp]
        return

    def get_pr_mask(self, sent):
        # find the boundary.
        #  L x  seqlen  x  bsz x K
        translate = [self.vocab2field[val] for val in sent]
        prev = 0
        curr = 0
        curr_lab = -1
        temp = torch.LongTensor(self.L, len(translate), 1, self.K).fill_(0)
        seglen = 0
        while(curr < len(translate)):
            if translate[curr] != curr_lab or seglen >= self.L - 1:
                if curr_lab != -1:
                    temp[seglen, prev, 0, curr_lab] = 1
                seglen = 0
                curr_lab = translate[curr]
                prev = curr
            else:
                seglen += 1
            curr += 1
        # output the last segment.
        temp[seglen, prev, 0, curr_lab] = 1

        return temp



    def pre_process_dev(self, data, result_train, bsz=20):
        data_lst, state_lst = data
        data_vocab = result_train['data_vocab']
        state_vocab = result_train['state_vocab']
        print(data_vocab)
        print(state_vocab)

        processed_data = []
        processed_state_lst = []
        for sent, state in zip(data_lst, state_lst):
            processed_data.append([data_vocab[x] for x in sent])
            processed_state_lst.append([state_vocab[x] for x in state])

        bsz_processed_data, mb2lineno = self.batchify(processed_data, processed_state_lst, data_lst, bsz)

        result = {'data_vocab': data_vocab, 'state_vocab': state_vocab,
                  'processed_data': processed_data, 'processed_state_lst': processed_state_lst,
                  'bsz_processed_data':bsz_processed_data, 'mb2lineno':mb2lineno}


        return result

    def batchify(self, token_lst, processed_state_lst, data_lst, bsz):
        sents, sorted_idxs = zip(*sorted(zip(token_lst, range(len(token_lst))), key=lambda x: len(x[0])))
        minibatches, mb2linenos = [], []
        curr_batch = []
        curr_linenos = []
        curr_strbsz = []
        curr_statez = []
        mask_pr = []
        curr_len = len(sents[0])
        for i in range(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz:  # we're done
                minibatches.append((torch.LongTensor(curr_batch),
                                    curr_statez, mask_pr, None,
                                    curr_strbsz)
                                   )

                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                curr_strbsz = [data_lst[sorted_idxs[i]]]
                curr_statez = [processed_state_lst[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
                mask_pr = [self.get_pr_mask(sents[i])]
            else:
                curr_batch.append(sents[i])
                curr_strbsz.append(data_lst[sorted_idxs[i]])
                curr_statez.append(processed_state_lst[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
                mask_pr.append(self.get_pr_mask(sents[i]))
                # catch last
        if len(curr_batch) > 0:
            minibatches.append((torch.LongTensor(curr_batch),
                                curr_statez, mask_pr, None,
                                curr_strbsz)
                               )
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos





if __name__ == '__main__':

    ''' generate '''
    if sys.argv[1] == 'generate':

        total_data =int(sys.argv[2])
        for _ in range(total_data):
            data, state = generate_all(length=4)
            print(' '.join(data) + '|||' + ' '.join(state))
    else:

        data_lst, state_lst = read_toy('/Users/xiangli/Desktop/Sasha/FSA-RNN/data/toy/train-toy2.txt')
        print('training data length is {}'.format(len(data_lst)))
        result_train = pre_process(data_lst, state_lst, bsz=10)

        data_lst, state_lst = read_toy('/Users/xiangli/Desktop/Sasha/FSA-RNN/data/toy/dev-toy2.txt')
        print('dev data length is {}'.format(len(data_lst)))
        # print(result_train)
        result_dev = pre_process_dev(data_lst, state_lst, result_train, bsz=10)
        # print(result_dev)

