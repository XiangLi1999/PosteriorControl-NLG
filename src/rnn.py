import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import cuda
from utils import *
from hsmm import HSMM
from rnn_new import RNNLM2
from hsmm_chunk import HSMM_chunk
from Chunking_Gen import Chunking_RNN
from hsmm_gen import HSMM_generative
import  time
# import torch_struct

class RNN_cond_Gen(nn.Module):
    def __init__(self, opt):
        super(RNN_cond_Gen, self).__init__()

        # implement the parameter sharing of the embedding space.
        self.vocab_size = opt.vocab_size
        self.field_vocab_size = opt.field_vocab_size
        self.idx_vocab_size = opt.idx_vocab_size
        self.embedding_dim = opt.embedding_dim
        self.table_hidden_dim = opt.table_hidden_dim
        self.f_dim = opt.f_dim
        self.i_dim = opt.i_dim

        self.sample_size = torch.Size(torch.LongTensor([opt.sample_size]))
        assert opt.pad_idx == 1
        self.table_dim = self.f_dim + self.i_dim*2 + self.embedding_dim
        opt.table_dim = self.table_dim

        self.posterior_reg = opt.posterior_reg

        self.rnn_lm = RNNLM2(opt)
        self.hsmm_crf = HSMM_generative(opt)

        self.word_vecs = self.rnn_lm.word_vecs
        self.field_vecs = self.rnn_lm.field_vecs
        self.idx_vecs = self.rnn_lm.idx_vecs
        self.table_lstm = self.rnn_lm.table_lstm

        # if opt.decoder == 'crf':
        #     print('using the discriminative version of the CRF HSMM model')
        #     self.hsmm_crf = HSMM(opt, self.word_vecs)
        # elif opt.decoder == 'gen':
        #     print('using the discriminative version of the  generative HSMM model')
        #     self.hsmm_crf = HSMM_generative(opt, self.word_vecs)


    def get_src_embs(self, src):
        bsz, nfields, nfeats = src.size()
        w_embs = self.word_vecs(src[:, :, 0])
        f_embs = self.field_vecs(src[:, :, 1])
        i_embs = self.idx_vecs(src[:, :, 2:]).view(bsz, nfields, -1)
        embs_repr = torch.cat([w_embs, f_embs, i_embs], dim=-1)
        return embs_repr

    def get_tgt_embs(self, tgt, version=1):
        '''
        (1) is an version in which we average over the possible source of each word.

        (2) is an version in which we only assume the first occurence of each word in the table.
        :param tgt:
        :return:
        '''
        if version == 2:
            seqlen, bsz, maxlocs, nfeats = tgt.size()
            w_embs = self.word_vecs(tgt[:, :, 0,  0])
            f_embs = self.field_vecs(tgt[:, :, 0, 1])
            i_embs = self.idx_vecs(tgt[:, :, 0, 2:]).view(seqlen, bsz, -1)
            embs_repr = torch.cat([w_embs, f_embs, i_embs], dim=-1)
            return embs_repr
        else:
            seqlen, bsz, maxlocs, nfeats = tgt.size()
            w_embs = self.word_vecs(tgt[:, :, 0, 0])
            f_embs = self.field_vecs(tgt[:, :, :, 1]).mean(2)
            i_embs = self.idx_vecs(tgt[:, :, :, 2:]).mean(2).view(seqlen, bsz, -1)
            embs_repr = torch.cat([w_embs, f_embs, i_embs], dim=-1)
            return embs_repr

    def get_field_embs(self, inp):
        f_embs = self.field_vecs(inp)
        return f_embs

    def get_tgt_wembs(self, tgt):
        w_embs = self.word_vecs(tgt[:, :, 0, 0])
        return w_embs




    def encode_table(self, src, avgmask, uniqfields):
        """
        args:
          src - bsz x nfields x nfeats
          avgmask - bsz x nfields, with 0s for pad and 1/tru_nfields for rest
          uniqfields - bsz x maxfields
        returns bsz x emb_size, bsz x nfields x emb_size
        """
        # TODO: adapt to the LSTM version of encoding.
        bsz, nfields, nfeats = src.size()
        w_embs = self.word_vecs(src[:,:,0])
        f_embs = self.field_vecs(src[:,:,1])
        i_embs = self.idx_vecs(src[:,:,2:]).view(bsz, nfields, -1)
        embs_repr = torch.cat([w_embs, f_embs, i_embs], dim=-1)
        field_vecs =  torch.cat([f_embs, i_embs], dim=-1)

        src_repr, _ = self.table_lstm(embs_repr)
        srcenc = torch.cat([src_repr[:,0,:self.table_hidden_dim], src_repr[:,-1,self.table_hidden_dim:]], dim=-1)

        # print(src.shape)
        # print(srcenc.shape)
        uniqenc = torch.tanh(srcenc)  # bsz x nfields x emb_size -> bsz x emb_size


        return {"srcenc": srcenc, "srcfieldenc": src_repr, "uniqenc": uniqenc, 'src_unique_field': uniqfields,
                'field_vecs': field_vecs}


    def get_sample_lst_vtb(self, z_gold):
        samples_lst, samples_vtb = [], []
        for elem in z_gold:
            samples_vtb.append([elem])
            samples_lst.append([elem])
        return samples_lst, samples_vtb


    def sanity_check(self,sent, uniqenc, dict_computed, viterbi_lst):
        score_big = self.hsmm_crf.get_score(viterbi_lst, dict_computed)
        score_big =  score_big[0][0] - dict_computed['Z'].item()
        word_cmp, state_cmp = self.rnn_lm.forward_with_state2(uniqenc, sent, viterbi_lst)
        # print(score_big, state_cmp, score_big.item() > state_cmp[0][0].item())


    def forward_with_crf(self, sent, condi, sample_num, gold_z = None, indep_reg=True, timing=None):

        dict_computed, _ = self.hsmm_crf.get_weights(sent, condi)
        with torch.enable_grad():
            Z, entropy, pr_expected = self.hsmm_crf.get_entr(dict_computed)
        if self.posterior_reg:
            pr_term = self.hsmm_crf.posterior_reg_term(sent, condi, pr_expected)
        else:
            pr_term = -1
        with torch.no_grad():
            samples_lst, samples_vtb = self.hsmm_crf.get_sample(dict_computed, sample_num=sample_num)

        # if not gold_z:
        # else:
        #     samples_vtb, samples_lst = gold_z
        #     samples_lst = [[samples_lst]]
        #     samples_vtb = [samples_vtb]

        sample_score = self.hsmm_crf.get_score(samples_lst, dict_computed)
        target = [torch.stack(samples) for samples in sample_score]
        target = torch.stack(target, dim=0)
        bsz, num_sample = target.shape

        state_llq = (target - Z.expand(bsz, num_sample))

        result_dict = self.rnn_lm.forward(condi, samples_lst)
        word_ll, state_llp = result_dict['p(y)'], result_dict['p(z)']


        result = {"word_ll": word_ll, "state_llp": state_llp,
                  "state_llq": state_llq, "entropy": entropy,
                  "samples_state": samples_lst, 'samples_vtb_style': samples_vtb, 'q(x)':Z.mean(), 'pr':pr_term }



        return result, dict_computed


    def forward_with_crf2(self, sent, condi, sample_num, gold_z = None, indep_reg=True, timing=None):

        dict_computed, log_potentials = self.hsmm_crf.get_weights(sent, condi)
        # log_potentials = torch.randn(log_potentials.shape)
        dist = torch_struct.SemiMarkovCRF(log_potentials)


        entropy = dist.entropy
        pr_expected = dist.marginals
        # entropy, pr_expected = self.hsmm_crf.get_entr(dict_computed)
        if self.posterior_reg:
            pr_term = self.hsmm_crf.posterior_reg_term(sent, condi, pr_expected)
        else:
            pr_term = -1

        samples_lst = dist.sample(self.sample_size)
        # samples_lst, samples_vtb = self.hsmm_crf.get_sample(dict_computed, sample_num=sample_num)
        state_llq = dist.log_prob(samples_lst)
        # sample_score = self.hsmm_crf.get_score(samples_lst, dict_computed)
        # target = [torch.stack(samples) for samples in sample_score]
        # target = torch.stack(target, dim=0)
        # bsz, num_sample = target.shape

        # state_llq = (target - Z.expand(bsz, num_sample))

        viterb_lst = dist.argmax
        viterb_score = dist.log_prob(viterb_lst)

        result_dict = self.rnn_lm.forward(condi, samples_lst)
        word_ll, state_llp = result_dict['p(y)'], result_dict['p(z)']


        result = {"word_ll": word_ll, "state_llp": state_llp,
                  "state_llq": state_llq, "entropy": entropy,
                  "samples_state": samples_lst, 'pr':pr_term,
                  'viterb_lst': viterb_lst, 'viterb_score':viterb_score}



        return result, dict_computed

    def process_state_translate(self, samples_state):
        # print(len(samples_state))
        begin_lst, end_lst, state_lst = samples_state[0]
        # result = [None] * (end_lst[-1] + 1)
        result = torch.ones(end_lst[-1]) * -1
        # print(result.shape)
        for idx, (a, b, s) in enumerate(zip(begin_lst, end_lst, state_lst)):
            # print(s, result[a:b+1])
            result[a:b] = s
        return result

    def save_model(self, path = 'latest_model_full'):
        torch.save(self.state_dict(), path)
        # with open(path, 'bw') as f:
        # pickle.dump(self, f)

    def load_model(self, path = 'latest_model_full'):
        self.load_state_dict(torch.load(path))
        # self = torch.load(path)
        return self


###############################################################################################################################################333
###############################################################################################################################################333
###############################################################################################################################################333

class Chunking_Model(nn.Module):
    def __init__(self, opt):
        super(Chunking_Model, self).__init__()

        # implement the parameter sharing of the embedding space.
        self.vocab_size = opt.vocab_size
        self.embedding_dim = opt.embedding_dim

        assert opt.pad_idx == 1

        self.posterior_reg = opt.posterior_reg
        self.rnn_lm = Chunking_RNN(opt)

        self.hsmm_crf = HSMM_chunk(opt)



        self.word_vecs = self.rnn_lm.word_vecs
        self.state_vecs = self.rnn_lm.state_vecs
        # self.field_vecs = self.rnn_lm.field_vecs
        # self.idx_vecs = self.rnn_lm.idx_vecs
        # self.table_lstm = self.rnn_lm.table_lstm

        # if opt.decoder == 'crf':
        #     print('using the discriminative version of the CRF HSMM model')
        #     self.hsmm_crf = HSMM(opt, self.word_vecs)
        # elif opt.decoder == 'gen':
        #     print('using the discriminative version of the  generative HSMM model')
        #     self.hsmm_crf = HSMM_generative(opt, self.word_vecs)


    def get_src_embs(self, src):
        bsz, nfields, nfeats = src.size()
        w_embs = self.word_vecs(src[:, :, 0])
        f_embs = self.field_vecs(src[:, :, 1])
        i_embs = self.idx_vecs(src[:, :, 2:]).view(bsz, nfields, -1)
        embs_repr = torch.cat([w_embs, f_embs, i_embs], dim=-1)
        return embs_repr


    def forward_with_rnn(self, sent, condi, sample_num):

        samples_lst, samples_vtb, state_llq = self.hsmm_crf.get_sample(condi, sample_num=sample_num)
        entropy = self.hsmm_crf.get_entr(state_llq)
        result_dict = self.rnn_lm.forward(condi, samples_lst)
        word_ll, state_llp = result_dict['p(y)'], result_dict['p(z)']

        # print(word_ll, state_llp, state_llq, entropy, Z.mean())

        result = {"word_ll": word_ll, "state_llp": state_llp,
                  "state_llq": state_llq, "entropy": entropy,
                  "samples_state": samples_lst, 'samples_vtb_style': samples_vtb, 'q(x)': Z.mean(), 'pr': pr_term}

        return result


    def get_tgt_embs(self, tgt, version=1):
        '''
        (1) is an version in which we average over the possible source of each word.

        (2) is an version in which we only assume the first occurence of each word in the table.
        :param tgt:
        :return:
        '''
        if version == 2:
            seqlen, bsz, maxlocs, nfeats = tgt.size()
            w_embs = self.word_vecs(tgt[:, :, 0,  0])
            f_embs = self.field_vecs(tgt[:, :, 0, 1])
            i_embs = self.idx_vecs(tgt[:, :, 0, 2:]).view(seqlen, bsz, -1)
            embs_repr = torch.cat([w_embs, f_embs, i_embs], dim=-1)
            return embs_repr
        else:
            seqlen, bsz, maxlocs, nfeats = tgt.size()
            w_embs = self.word_vecs(tgt[:, :, 0, 0])
            f_embs = self.field_vecs(tgt[:, :, :, 1]).mean(2)
            i_embs = self.idx_vecs(tgt[:, :, :, 2:]).mean(2).view(seqlen, bsz, -1)
            embs_repr = torch.cat([w_embs, f_embs, i_embs], dim=-1)
            return embs_repr

    def get_tgt_wembs(self, tgt):
        w_embs = self.word_vecs(tgt)
        return w_embs

    def get_state_embs(self):
        s_embs = self.state_vecs
        return s_embs

    def get_word_embs(self):
        w_embs = self.word_vecs
        return w_embs




    def encode_table(self, src, avgmask, uniqfields):
        """
        args:
          src - bsz x nfields x nfeats
          avgmask - bsz x nfields, with 0s for pad and 1/tru_nfields for rest
          uniqfields - bsz x maxfields
        returns bsz x emb_size, bsz x nfields x emb_size
        """
        # TODO: adapt to the LSTM version of encoding.
        bsz, nfields, nfeats = src.size()
        w_embs = self.word_vecs(src[:,:,0])
        f_embs = self.field_vecs(src[:,:,1])
        i_embs = self.idx_vecs(src[:,:,2:]).view(bsz, nfields, -1)
        embs_repr = torch.cat([w_embs, f_embs, i_embs], dim=-1)
        field_vecs =  torch.cat([f_embs, i_embs], dim=-1)

        src_repr, _ = self.table_lstm(embs_repr)
        srcenc = torch.cat([src_repr[:,0,:self.table_hidden_dim], src_repr[:,-1,self.table_hidden_dim:]], dim=-1)

        return {"srcenc": srcenc, "srcfieldenc": src_repr, "uniqenc": None, 'src_unique_field': uniqfields,
                'field_vecs': field_vecs}


    def get_sample_lst_vtb(self, z_gold):
        samples_lst, samples_vtb = [], []
        for elem in z_gold:
            samples_vtb.append([elem])
            samples_lst.append([elem])
        return samples_lst, samples_vtb


    def sanity_check(self,sent, uniqenc, dict_computed, viterbi_lst):
        score_big = self.hsmm_crf.get_score(viterbi_lst, dict_computed)
        score_big =  score_big[0][0] - dict_computed['Z'].item()
        word_cmp, state_cmp = self.rnn_lm.forward_with_state2(uniqenc, sent, viterbi_lst)
        # print(score_big, state_cmp, score_big.item() > state_cmp[0][0].item())


    def forward_with_crf(self, sent, condi, sample_num, gold_z = None, indep_reg=True, timing=None, more_info=False):

        dict_computed = self.hsmm_crf.get_weights(sent, condi)
        with torch.enable_grad():
            Z, entropy, pr_expected = self.hsmm_crf.get_entr(dict_computed)

        if self.posterior_reg:
            inp_cond = condi if more_info else None
            pr_term = self.hsmm_crf.posterior_reg_term(sent, inp_cond, pr_expected)
        else:
            pr_term = -1

        # print(pr_expected[0][:,:,0,:].sum())
        # print(sent.shape)

        with torch.no_grad():
            samples_lst, samples_vtb = self.hsmm_crf.get_sample(dict_computed, sample_num=sample_num)

        sample_score = self.hsmm_crf.get_score(samples_lst, dict_computed)
        target = [torch.stack(samples) for samples in sample_score]
        target = torch.stack(target, dim=0)
        bsz, num_sample = target.shape


        state_llq = (target - Z.expand(bsz, num_sample))

        result_dict = self.rnn_lm.forward(condi, samples_lst)
        word_ll, state_llp = result_dict['p(y)'], result_dict['p(z)']

        # print(word_ll.mean(), state_llp.mean(), state_llq.mean(), entropy.mean(), Z.mean(), pr_term.mean())

        result = {"word_ll": word_ll, "state_llp": state_llp,
                  "state_llq": state_llq, "entropy": entropy,
                  "samples_state": samples_lst, 'samples_vtb_style': samples_vtb, 'q(x)':Z.mean(), 'pr':pr_term }


        return result, dict_computed

    def process_state_translate(self, samples_state):
        # print(len(samples_state))
        begin_lst, end_lst, state_lst = samples_state[0]
        # result = [None] * (end_lst[-1] + 1)
        result = torch.ones(end_lst[-1]) * -1
        # print(result.shape)
        for idx, (a, b, s) in enumerate(zip(begin_lst, end_lst, state_lst)):
            # print(s, result[a:b+1])
            result[a:b] = s
        return result

    def save_model(self, path = 'latest_model_full'):
        torch.save(self.state_dict(), path)
        # with open(path, 'bw') as f:
        # pickle.dump(self, f)

    def load_model(self, path = 'latest_model_full'):
        self.load_state_dict(torch.load(path))
        # self = torch.load(path)
        return self

