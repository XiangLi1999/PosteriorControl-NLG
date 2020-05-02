import torch
from torch import nn
import torch.nn.functional as F
import argparse
import numpy as np
from torch import cuda
from utils import *
from hsmm import HSMM

neginf = -1e38
from collections import defaultdict
from queue import PriorityQueue
from beam_search import BeamSearchNode
import operator
import time


class Chunking_RNN(nn.Module):
    def __init__(self, options):
        super(Chunking_RNN, self).__init__()
        self.h_dim = options.hidden_dim
        self.w_dim = options.embedding_dim
        self.s_dim = options.s_dim

        self.vocab_size = options.vocab_size
        self.dual_attn = (options.dual_attn == 'yes')
        self.self_attn = (options.self_attn == 'yes')

        self.tagset_size = options.tagset_size + 1  # the last element is the BOS state.
        self.device = options.device
        # self.unk = options.unk_idx

        # when we encode the fielded_words by the concatenation of [word_emb, field_emb, idx_emb]
        rnninsz = self.w_dim

        self.word_vecs = nn.Embedding(self.vocab_size, options.embedding_dim, padding_idx=options.pad_idx)
        self.state_vecs = nn.Embedding(self.tagset_size, self.s_dim)
        self.dropout = nn.Dropout(options.dropout)
        self.layers = options.layers
        self.num_layers = self.layers
        self.rnninsz = rnninsz

        '''attention and copy attention'''
        if self.self_attn:
            attn_tagset = self.tagset_size
            self.mid_dim = 100
            # input matrix 1.
            self.state_att_gates = nn.Parameter(torch.randn(attn_tagset, 1, self.h_dim, self.mid_dim))
            self.state_att_biases = nn.Parameter(torch.randn(attn_tagset, 1, self.mid_dim))
            self.attn_src = nn.Linear(self.conditional_dim, self.mid_dim)

            attn_dim = self.h_dim + self.conditional_dim
            self.sep_attn = options.sep_attn
            if self.sep_attn:
                # input matrix 2.
                self.state_att2_gates = nn.Parameter(torch.randn(attn_tagset, 1, self.h_dim, self.mid_dim))
                self.state_att2_biases = nn.Parameter(torch.randn(attn_tagset, 1, self.mid_dim))
            # output matrix 1.
            out_hid_sz = self.h_dim + self.conditional_dim
            self.state_out_gates = nn.Parameter(torch.randn(attn_tagset, 1, 1, out_hid_sz))
            self.state_out_biases = nn.Parameter(torch.randn(attn_tagset, 1, 1, out_hid_sz))

            if self.dual_attn:
                self.state_att_gates_dual = nn.Parameter(torch.randn(attn_tagset, 1, self.h_dim, self.mid_dim))
                self.state_att_biases_dual = nn.Parameter(torch.randn(attn_tagset, 1, self.mid_dim))
                self.attn_src_dual = nn.Linear(self.f_dim + self.i_dim + self.i_dim, self.mid_dim)
                if self.sep_attn:
                    self.state_att2_gates_dual = nn.Parameter(torch.randn(attn_tagset, 1, self.h_dim, self.mid_dim))
                    self.state_att2_biases_dual = nn.Parameter(torch.randn(attn_tagset, 1, self.mid_dim))


        self.zeros = torch.Tensor(1, 1).fill_(neginf).to(options.device)
        self.eos_idx = options.eos_idx
        self.bos_idx = options.bos_idx
        # self.idx2word = options.idx2word
        # self.temp_field = options.temp_field.cpu()

        self.full_independence = options.full_independence  # -1 if fully independence, 1 if fully RNN.

        # related to the type three of our full_independence model.

        attn_dim = self.h_dim + self.s_dim
        self.rnn = nn.LSTM(rnninsz + self.s_dim, self.h_dim, num_layers=self.num_layers,
                           batch_first=True, dropout=options.dropout)
        self.state_pick = nn.Sequential(nn.Linear(self.h_dim, self.h_dim // 2), nn.ReLU(),
                                        nn.Linear(self.h_dim // 2, self.tagset_size - 1))
        self.vocab_pick = nn.Sequential(nn.Linear(attn_dim, attn_dim), nn.ReLU(),
                                        nn.Linear(attn_dim, self.vocab_size))
        self.bos_emb = nn.Parameter(torch.zeros(1, 1, rnninsz + self.s_dim))


        # state attention
        if self.self_attn:
            self.state_att_gates2 = nn.Parameter(torch.randn(1, self.h_dim, self.mid_dim))
            self.state_att_biases2 = nn.Parameter(torch.randn(1, self.mid_dim))
            self.attn_src2 = nn.Linear(self.conditional_dim, self.mid_dim)
            self.state_out_gates2 = nn.Parameter(torch.randn(1, 1, out_hid_sz))
            self.state_out_biases2 = nn.Parameter(torch.randn(1, 1, out_hid_sz))
            self.decoder_constraint = (options.decoder_constraint == 'yes')
            if self.dual_attn:
                self.state_att_gates2_dual = nn.Parameter(torch.randn(1, self.h_dim, self.mid_dim))
                self.state_att_biases2_dual = nn.Parameter(torch.randn(1, self.mid_dim))
                self.attn_src2_dual = nn.Linear(self.f_dim + self.i_dim + self.i_dim, self.mid_dim)

            if self.dual_attn:
                self.forward = self.forward_easy_dual
                self.beam_forward = self.beam_forward_easy_dual
            else:
                self.forward = self.forward_easy
                self.beam_forward = self.beam_forward_easy
                self.beam_control = self.beam_cond_gen_easy
        else:
            self.forward = self.forward_easy
            self.beam_forward = self.beam_forward_easy
            self.beam_control = self.beam_cond_gen_easy


    def get_state_seq(self, sample_lst, seqlen):
        bsz = len(sample_lst)
        sample_size = len(sample_lst[0])
        result = torch.zeros(bsz, sample_size, seqlen).to(self.device)
        for b_idx, b in enumerate(sample_lst):
            for samp_idx, samp in enumerate(b):
                start, end, states = samp
                temp = sum([[state] * (t - s) for (s, t, state) in zip(start, end, states)], [])
                result[b_idx][samp_idx] = torch.LongTensor(temp)
        return result


    def forward_easy(self, condi, sample_lst):
        '''
            condi: the representation of x and y.
            sample_lst: the representation of z.
        '''

        # TODO: validate the unsupervised copy attention idea.
        # x = condi['inps'][:,:,0,:].unsqueeze(2)
        x = condi['inps']
        bsz_orig, seqlen = x.size()
        targs = condi['targs']
        layers, rnn_size = self.layers, self.h_dim
        self.sample_size = len(sample_lst[0])

        state_names = self.get_state_seq(sample_lst, seqlen).long()
        word_embs = condi['tgt']
        inpembs = self.dropout(word_embs).transpose(1, 0)

        inpembs = inpembs.repeat_interleave(self.sample_size, dim=1).transpose(0, 1)  # bsz * seqlen * dim

        state_embs = self.state_vecs(state_names).view(bsz_orig * self.sample_size, seqlen, -1)  # bsz * seqlen * dim
        state_embs = self.dropout(state_embs)
        rnn_inp = torch.cat([inpembs, state_embs], dim=-1)  # bsz x seqlen x nfeats*emb + state_embs
        rnn_inp = torch.cat([self.bos_emb.expand(bsz_orig * self.sample_size, 1, -1), rnn_inp], dim=1)

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim

        h, _ = self.rnn(rnn_inp, None)  # h.shape = (batch_size * sample_size, seqlen, h_dim)
        h = self.dropout(h)

        slps_k = F.log_softmax(self.state_pick(h[:, :-1]), dim=-1)
        targstate = state_names.view(bsz_orig * self.sample_size, seqlen, 1)
        logp_state_temp = torch.gather(slps_k, 2, targstate)
        # print(logp_state_temp.shape)
        logp_state = logp_state_temp.sum(-1).sum(-1)
        # print(logp_state.shape)


        h1 = torch.cat([h[:, :-1], state_embs], dim=-1)
        wlps_k = F.log_softmax(self.vocab_pick(h1), dim=-1)  # (bsz*seqlen, nfield)
        targs = torch.repeat_interleave(targs, self.sample_size, dim=0)
        logp_word_temp = torch.gather(wlps_k, 2, targs.unsqueeze(2))
        logp_word = torch.sum(logp_word_temp.squeeze(2), dim=-1)

        result_dict = {'p(y)': logp_word.view(bsz_orig, -1), 'p(z)': logp_state.view(bsz_orig, -1)}
        return result_dict

    def beam_cond_gen_easy(self, condi, bsz, beam_size):
        '''

        :param condi: condi contains a sequence of template.
        :param bsz: the batch_size.
        :return: the generated sequence.
        '''
        beam_width = 3
        beam_w_ = 1
        topk = beam_size  # how many sentence do you want to generate
        decoded_batch = []
        decoded_score = []
        decoded_states = []
        max_len = 60
        window_size = 3

        clean_ = False
        clean_size = 100

        # x = condi['inps']
        # seqlen, bsz_orig, maxlocs, nfeats = x.size()
        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        src_wrd2fields = condi["src"]
        template = condi['template']

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        layers, rnn_size = self.layers, self.h_dim
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        # print(inits.shape, self.hidden_dim, rnn_size)
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim
        c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim

        # the rnns have batch first.
        for idx in range(bsz):
            true_len = len(template[idx])
            decoder_hidden = (h_prev[:, idx, :].unsqueeze(1).contiguous(), c_prev[:, idx, :].unsqueeze(1).contiguous())
            # Start with the start of the sentence token

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length, state_id, word_feats, rnninp
            node = BeamSearchNode(decoder_hidden, None, self.bos_idx, 0, 1, -1, -1, rnninp=self.bos_emb, window=[])
            node.prev_cell = None
            node.stack = None

            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1
            nfeats = 4

            # start beam search
            break_flag = False
            while True:
                # give up when we have enough good results.
                if len(endnodes) >= number_required: break
                # give up when decoding takes too long
                if qsize > 3000:
                    print('break due to huge amount')
                    break
                # fetch the best node
                if nodes.qsize() > 0:
                    score_top, n_top = nodes.get()
                else:
                    return 'failed --> \n'

                if n_top.wordid == self.eos_idx and n_top.prevNode != None:
                    endnodes.append((score, n_top))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                h, decoder_hidden = self.rnn(n_top.rnninp.contiguous(), n_top.h)
                out_rnn, copy = self.attn_step(h, -1, srcfieldenc[idx:idx + 1], fieldmask[idx:idx + 1], mod='state')

                state_logp = nn.LogSoftmax(dim=-1)(self.state_pick(out_rnn))

                if n_top.leng > true_len:
                    decoded_ts = template[idx][-1]
                else:
                    decoded_ts = template[idx][n_top.leng - 1]

                log_ps = state_logp[0, 0, decoded_ts].item()

                out_rnn, copy = self.attn_step(h, decoded_ts.item(), srcfieldenc[idx:idx + 1],
                                               fieldmask[idx:idx + 1])
                temp_wordlp = nn.LogSoftmax(dim=-1)(torch.cat([self.vocab_pick(out_rnn), copy], dim=-1))
                log_prob_w, indexes_w = torch.topk(temp_wordlp, beam_width + window_size + 1)

                temp_ww = []
                temp_pp = []
                for elem1, elem2 in zip(indexes_w.view(-1), log_prob_w.view(-1)):
                    if elem1.item() in [x[0] for x in n_top.window[-window_size:]]:
                        pass
                    elif n_top.leng - 1 < true_len - 1 and elem1.item() == self.eos_idx:
                        # print('you shouldn t stop plz move on. ')
                        pass
                    else:
                        temp_ww.append(elem1)
                        temp_pp.append(elem2)
                    if len(temp_ww) >= beam_width:
                        break

                nextnodes = []
                for new_k_w in range(beam_width):
                    decoded_tw = temp_ww[new_k_w].view(-1)[0]
                    log_pw = temp_pp[new_k_w].item()
                    # decoded_tw = indexes_w[0][0][new_k_w].view(-1)[0]
                    # log_pw = log_prob_w[0][0][new_k_w].item()
                    window_new = n_top.window + [(decoded_tw.item(), decoded_ts.item())]

                    if decoded_tw > self.gen_vocab_size:
                        # generate out of gen_vocab: copy
                        temp_word_id = decoded_tw - self.gen_vocab_size - 1
                        decoded_feat = torch.LongTensor(src_wrd2fields[idx][temp_word_id]).to(
                            self.device)  # bsz x seqlen x 4
                        decoded_tw = decoded_feat[0]
                    else:
                        # generate from the gen vocab.
                        temp = self.temp_field
                        temp[0] = decoded_tw
                        decoded_feat = temp
                    # print('gen=',self.idx2word[decoded_t.item()])
                    # rnninp
                    temp_word_emb = self.inpmlp(self.word_vecs(decoded_feat).view(1, 1, -1))
                    temp_state_emb = self.state_vecs(decoded_ts).view(1, 1, -1)
                    # build rnninp
                    rnninp = torch.cat([temp_word_emb, temp_state_emb], dim=-1)
                    node = BeamSearchNode(decoder_hidden, n_top, decoded_tw, n_top.logp + log_pw + log_ps,
                                          n_top.leng + 1, state_id=decoded_ts.item(), rnninp=rnninp,
                                          window=window_new)
                    score = -node.eval()
                    if node.leng >= max_len:
                        endnodes.append((score, node))
                        if len(endnodes) >= number_required:
                            break_flag = True
                            break
                        else:
                            continue
                    else:
                        nextnodes.append((score, node))

                if break_flag:
                    break

                for i in range(len(nextnodes)):
                    score, nn_ = nextnodes[i]
                    nodes.put((score, nn_))
                    qsize += 1

                if clean_ and qsize >= clean_size:
                    tempnodes = [nodes.get() for _ in range(min(clean_size, qsize)) if not nodes.empty()]
                    nodes = PriorityQueue()
                    qsize = 0
                    for x in tempnodes:
                        nodes.put(x)
                        qsize += 1
                elif clean_:
                    pass

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk) if not nodes.empty()]

            # back tracing
            utterances_w = []
            utterances_s = []
            scores_result = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance_w = []
                utterance_s = []
                utterance_w.append(n.wordid)
                utterance_s.append(n.state_id)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance_w.append(n.wordid)
                    utterance_s.append(n.state_id)

                utterance_w = utterance_w[::-1]
                utterance_s = utterance_s[::-1]

                utterances_w.append(utterance_w)
                utterances_s.append(utterance_s)
                scores_result.append(score)

            decoded_batch.append(utterances_w)
            decoded_states.append(utterances_s)
            decoded_score.append(scores_result)
        temp_lst = idx2word(decoded_batch, self.idx2word)

        final_states = []
        len_marks = []
        for x in decoded_states:
            final_states += x
            len_marks += [y for y in range(len(x))]
        final_scores = []
        for x in decoded_score:
            final_scores += x

        result_str = ''
        for idx_, elem in enumerate(temp_lst):
            result_str += ' '.join(elem)
            result_str += '|||{}'.format(final_scores[idx_])
            result_str += '|||{}'.format(final_states[idx_])
            result_str += '|||{}'.format(len_marks[idx_])
            result_str += '\n'
        return result_str

    def get_single_embs(self, tgt, version=1):
        '''
        (1) is an version in which we average over the possible source of each word.

        (2) is an version in which we only assume the first occurence of each word in the table.
        :param tgt:
        :return:
        '''
        w_embs = self.word_vecs(tgt[0])
        f_embs = self.field_vecs(tgt[1])
        i_embs = self.idx_vecs(tgt[2:]).view(-1)
        embs_repr = torch.cat([w_embs, f_embs, i_embs], dim=-1)
        return embs_repr

    def get_word_embs(self, tgt):
        # print(self.word_vecs.device)
        w_embs = self.word_vecs(tgt)
        return w_embs

    def beam_forward_easy(self, condi, bsz):
        beam_width = 5
        beam_w_ = 1
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        decoded_score = []
        decoded_states = []
        decoded_copy = []
        max_len = 70
        window_size = 3

        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        src_wrd2fields = condi["src"]

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        layers, rnn_size = self.layers, self.h_dim
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim
        c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim

        # the rnns have batch first.
        for idx in range(bsz):
            decoder_hidden = (h_prev[:, idx, :].unsqueeze(1).contiguous(), c_prev[:, idx, :].unsqueeze(1).contiguous())
            # Start with the start of the sentence token

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length, state_id, word_feats, rnninp
            node = BeamSearchNode(decoder_hidden, None, self.bos_idx, 0, 1, -1, -1, rnninp=self.bos_emb, window=[])
            node.record = -1
            node.prev_cell = None
            node.stack = None
            prev_nodes = []
            prev_nodes.append((-node.eval(), node))
            # start beam search
            break_flag = False
            while True:
                # give up when we have enough good results.
                if len(endnodes) >= number_required: break

                # fetch the best node
                nextnodes = []
                for elem in range(min(len(prev_nodes), beam_width)):
                    score_top, n_top = prev_nodes[elem]

                    if n_top.wordid == self.eos_idx and n_top.prevNode != None:
                        endnodes.append((score, n_top))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    h, decoder_hidden = self.rnn(n_top.rnninp.contiguous(), n_top.h)
                    out_rnn, copy = self.attn_step(h, -1, srcfieldenc[idx:idx + 1], fieldmask[idx:idx + 1], mod='state')
                    state_logp = nn.LogSoftmax(dim=-1)(self.state_pick(out_rnn))
                    log_prob, indexes = torch.topk(state_logp, beam_w_)
                    for new_k in range(beam_w_):
                        decoded_ts = indexes[0][0][new_k].view(-1)
                        log_ps = log_prob[0][0][new_k].item()

                        out_rnn, copy = self.attn_step(h, decoded_ts.item(), srcfieldenc[idx:idx + 1],
                                                       fieldmask[idx:idx + 1])
                        temp_wordlp = nn.LogSoftmax(dim=-1)(torch.cat([self.vocab_pick(out_rnn), copy], dim=-1))
                        log_prob_w, indexes_w = torch.topk(temp_wordlp, beam_width + window_size)

                        temp_ww = []
                        temp_pp = []
                        for elem1, elem2 in zip(indexes_w.view(-1), log_prob_w.view(-1)):
                            if elem1.item() in [x[0] for x in n_top.window[-window_size:]]:
                                pass
                            else:
                                temp_ww.append(elem1)
                                temp_pp.append(elem2)
                            if len(temp_ww) >= beam_width:
                                break
                        for new_k_w in range(beam_width):
                            decoded_tw = temp_ww[new_k_w].view(-1)[0]
                            log_pw = temp_pp[new_k_w].item()
                            window_new = n_top.window + [(decoded_tw.item(), decoded_ts.item())]

                            if decoded_tw > self.gen_vocab_size:
                                # generate out of gen_vocab: copy
                                temp_word_id = decoded_tw - self.gen_vocab_size - 1
                                decoded_feat = src_wrd2fields[idx][temp_word_id]  # bsz x seqlen x 4
                                # decoded_feat = torch.LongTensor(src_wrd2fields[idx][temp_word_id]) # bsz x seqlen x 4
                                decoded_tw = decoded_feat[0]
                                record = temp_word_id.item()
                            else:
                                # generate from the gen vocab.
                                temp = self.temp_field
                                temp[0] = decoded_tw
                                decoded_feat = temp
                                record = -1

                            # rnninp
                            temp_word_emb = self.get_single_embs(decoded_feat).view(1, 1, -1)
                            temp_state_emb = self.state_vecs(decoded_ts).view(1, 1, -1)
                            rnninp = torch.cat([temp_word_emb, temp_state_emb], dim=-1)
                            node = BeamSearchNode(decoder_hidden, n_top, decoded_tw, n_top.logp + log_ps + log_pw,
                                                  n_top.leng + 1, state_id=decoded_ts.item(), rnninp=rnninp,
                                                  window=window_new)
                            node.record = record
                            score = -node.eval()

                            if node.leng >= max_len:
                                endnodes.append((score, node))
                                if len(endnodes) >= number_required:
                                    break_flag = True
                                    break
                                else:
                                    continue
                            else:
                                nextnodes.append((score, node))

                        if break_flag:
                            break

                    if break_flag:
                        break

                next_nodes = []
                for i in range(len(nextnodes)):
                    score, nn_ = nextnodes[i]
                    next_nodes.append((score, nn_))
                    # qsize += 1
                # sort next_nodes.
                next_nodes = sorted(next_nodes, key=lambda x: x[0])
                prev_nodes = next_nodes

            # back tracing
            utterances_w = []
            utterances_s = []
            scores_result = []
            copys = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance_w = []
                utterance_s = []
                copy = []
                utterance_w.append(n.wordid)
                utterance_s.append(n.state_id)
                copy.append(n.record)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance_w.append(n.wordid)
                    utterance_s.append(n.state_id)
                    copy.append(n.record)

                utterance_w = utterance_w[::-1]
                utterance_s = utterance_s[::-1]
                copy = copy[::-1]
                utterances_w.append(utterance_w)
                utterances_s.append(utterance_s)
                scores_result.append(score)
                copys.append(copy)

            decoded_batch.append(utterances_w)
            decoded_states.append(utterances_s)
            decoded_score.append(scores_result)
            decoded_copy.append(copys)

        temp_lst = idx2word(decoded_batch, self.idx2word)

        final_states = []
        len_marks = []
        for x in decoded_states:
            final_states += x
            len_marks += [y for y in range(len(x))]
        final_scores = []
        for x in decoded_score:
            final_scores += x
        final_copy = []
        for x in decoded_copy:
            final_copy += x

        result_str = ''
        for idx_, elem in enumerate(temp_lst):
            result_str += ' '.join(elem)
            result_str += '|||{}'.format(final_scores[idx_])
            result_str += '|||{}'.format(final_states[idx_])
            result_str += '|||{}'.format(len_marks[idx_])
            result_str += '|||{}'.format(final_copy[idx_])
            result_str += '\n'
        return result_str

    def beam_forward_easy_true(self, condi, bsz):
        beam_width = 5
        beam_w_ = 1
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        decoded_score = []
        decoded_states = []
        max_len = 60
        window_size = 3
        clean_ = False
        clean_size = 100

        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        src_wrd2fields = condi["src"]

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        layers, rnn_size = self.layers, self.h_dim
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        # print(inits.shape, self.hidden_dim, rnn_size)
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim
        c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim

        # the rnns have batch first.
        for idx in range(bsz):
            decoder_hidden = (h_prev[:, idx, :].unsqueeze(1).contiguous(), c_prev[:, idx, :].unsqueeze(1).contiguous())
            # Start with the start of the sentence token

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length, state_id, word_feats, rnninp
            node = BeamSearchNode(decoder_hidden, None, self.bos_idx, 0, 1, -1, -1, rnninp=self.bos_emb, window=[])
            node.prev_cell = None
            node.stack = None
            prev_nodes = []
            prev_nodes.append((-node.eval(), node))
            # qsize = 1
            # start beam search
            break_flag = False
            while True:
                # give up when we have enough good results.
                if len(endnodes) >= number_required: break
                # give up when decoding takes too long
                # if qsize > 2000:
                #     print('break due to huge amount')
                #     break

                # fetch the best node
                nextnodes = []
                for elem in range(min(len(prev_nodes), beam_width)):
                    score_top, n_top = prev_nodes[elem]

                    if n_top.wordid == self.eos_idx and n_top.prevNode != None:
                        endnodes.append((score, n_top))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    h, decoder_hidden = self.rnn(n_top.rnninp.contiguous(), n_top.h)
                    out_rnn, copy = self.attn_step(h, -1, srcfieldenc[idx:idx + 1], fieldmask[idx:idx + 1], mod='state')
                    state_logp = nn.LogSoftmax(dim=-1)(self.state_pick(out_rnn))
                    log_prob, indexes = torch.topk(state_logp, beam_w_)
                    for new_k in range(beam_w_):
                        decoded_ts = indexes[0][0][new_k].view(-1)
                        log_ps = log_prob[0][0][new_k].item()

                        out_rnn, copy = self.attn_step(h, decoded_ts.item(), srcfieldenc[idx:idx + 1],
                                                       fieldmask[idx:idx + 1])
                        temp_wordlp = nn.LogSoftmax(dim=-1)(torch.cat([self.vocab_pick(out_rnn), copy], dim=-1))
                        log_prob_w, indexes_w = torch.topk(temp_wordlp, beam_width + window_size)

                        temp_ww = []
                        temp_pp = []
                        for elem1, elem2 in zip(indexes_w.view(-1), log_prob_w.view(-1)):
                            if elem1.item() in [x[0] for x in n_top.window[-window_size:]]:
                                pass
                            else:
                                temp_ww.append(elem1)
                                temp_pp.append(elem2)
                            if len(temp_ww) >= beam_width:
                                break
                        for new_k_w in range(beam_width):
                            decoded_tw = temp_ww[new_k_w].view(-1)[0]
                            log_pw = temp_pp[new_k_w].item()
                            # decoded_tw = indexes_w[0][0][new_k_w].view(-1)[0]
                            # log_pw = log_prob_w[0][0][new_k_w].item()
                            window_new = n_top.window + [(decoded_tw.item(), decoded_ts.item())]

                            if decoded_tw > self.gen_vocab_size:
                                # generate out of gen_vocab: copy
                                temp_word_id = decoded_tw - self.gen_vocab_size - 1
                                decoded_feat = torch.LongTensor(src_wrd2fields[idx][temp_word_id]).to(
                                    self.device)  # bsz x seqlen x 4
                                decoded_tw = decoded_feat[0]
                            else:
                                # generate from the gen vocab.
                                temp = self.temp_field
                                temp[0] = decoded_tw
                                decoded_feat = temp
                            # print('gen=',self.idx2word[decoded_t.item()])
                            # rnninp
                            temp_word_emb = self.inpmlp(self.word_vecs(decoded_feat).view(1, 1, -1))
                            temp_state_emb = self.state_vecs(decoded_ts).view(1, 1, -1)
                            # build rnninp
                            rnninp = torch.cat([temp_word_emb, temp_state_emb], dim=-1)
                            node = BeamSearchNode(decoder_hidden, n_top, decoded_tw, n_top.logp + log_ps + log_pw,
                                                  n_top.leng + 1, state_id=decoded_ts.item(), rnninp=rnninp,
                                                  window=window_new)
                            score = -node.eval()

                            if node.leng >= max_len:
                                endnodes.append((score, node))
                                if len(endnodes) >= number_required:
                                    break_flag = True
                                    break
                                else:
                                    continue
                            else:
                                nextnodes.append((score, node))

                        if break_flag:
                            break

                    if break_flag:
                        break

                next_nodes = []
                for i in range(len(nextnodes)):
                    score, nn_ = nextnodes[i]
                    next_nodes.append((score, nn_))
                    # qsize += 1
                # sort next_nodes.
                next_nodes = sorted(next_nodes, key=lambda x: x[0])
                prev_nodes = next_nodes

            # back tracing
            utterances_w = []
            utterances_s = []
            scores_result = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance_w = []
                utterance_s = []
                utterance_w.append(n.wordid)
                utterance_s.append(n.state_id)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance_w.append(n.wordid)
                    utterance_s.append(n.state_id)

                utterance_w = utterance_w[::-1]
                utterance_s = utterance_s[::-1]
                utterances_w.append(utterance_w)
                utterances_s.append(utterance_s)
                scores_result.append(score)

            decoded_batch.append(utterances_w)
            decoded_states.append(utterances_s)
            decoded_score.append(scores_result)
        temp_lst = idx2word(decoded_batch, self.idx2word)

        final_states = []
        len_marks = []
        for x in decoded_states:
            final_states += x
            len_marks += [y for y in range(len(x))]
        final_scores = []
        for x in decoded_score:
            final_scores += x

        result_str = ''
        for idx_, elem in enumerate(temp_lst):
            result_str += ' '.join(elem)
            result_str += '|||{}'.format(final_scores[idx_])
            result_str += '|||{}'.format(final_states[idx_])
            result_str += '|||{}'.format(len_marks[idx_])
            result_str += '\n'
        return result_str

    def beam_forward_easy2(self, condi, bsz):
        beam_width = 2
        beam_w_ = 1
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        decoded_score = []
        decoded_states = []
        max_len = 60
        window_size = 3

        clean_ = False
        clean_size = 100

        # x = condi['inps']
        # seqlen, bsz_orig, maxlocs, nfeats = x.size()
        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        src_wrd2fields = condi["src"]

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        layers, rnn_size = self.layers, self.h_dim
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        # print(inits.shape, self.hidden_dim, rnn_size)
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim
        c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim

        # the rnns have batch first.
        for idx in range(bsz):
            decoder_hidden = (h_prev[:, idx, :].unsqueeze(1).contiguous(), c_prev[:, idx, :].unsqueeze(1).contiguous())
            # Start with the start of the sentence token

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length, state_id, word_feats, rnninp
            node = BeamSearchNode(decoder_hidden, None, self.bos_idx, 0, 1, -1, -1, rnninp=self.bos_emb, window=[])
            node.prev_cell = None
            node.stack = None

            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1
            nfeats = 4

            # start beam search
            break_flag = False
            while True:
                # give up when we have enough good results.
                if len(endnodes) >= number_required: break
                # give up when decoding takes too long
                if qsize > 2000:
                    print('break due to huge amount')
                    break

                # fetch the best node
                if nodes.qsize() > 0:
                    score_top, n_top = nodes.get()
                else:
                    return 'failed --> \n'

                if n_top.wordid == self.eos_idx and n_top.prevNode != None:
                    endnodes.append((score, n_top))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                h, decoder_hidden = self.rnn(n_top.rnninp.contiguous(), n_top.h)
                out_rnn, copy = self.attn_step(h, -1, srcfieldenc[idx:idx + 1], fieldmask[idx:idx + 1], mod='state')

                state_logp = nn.LogSoftmax(dim=-1)(self.state_pick(out_rnn))
                log_prob, indexes = torch.topk(state_logp, beam_w_)
                nextnodes = []
                for new_k in range(beam_w_):
                    decoded_ts = indexes[0][0][new_k].view(-1)
                    log_ps = log_prob[0][0][new_k].item()

                    # give nthe current state, pick the vocab.
                    # print(fieldmask.shape)
                    out_rnn, copy = self.attn_step(h, decoded_ts.item(), srcfieldenc[idx:idx + 1],
                                                   fieldmask[idx:idx + 1])
                    temp_wordlp = nn.LogSoftmax(dim=-1)(torch.cat([self.vocab_pick(out_rnn), copy], dim=-1))
                    log_prob_w, indexes_w = torch.topk(temp_wordlp, beam_width + window_size)

                    temp_ww = []
                    temp_pp = []
                    for elem1, elem2 in zip(indexes_w.view(-1), log_prob_w.view(-1)):
                        if elem1.item() in [x[0] for x in n_top.window[-window_size:]]:
                            pass
                        else:
                            temp_ww.append(elem1)
                            temp_pp.append(elem2)
                        if len(temp_ww) >= beam_width:
                            break

                    nextnodes = []
                    for new_k_w in range(beam_width):
                        decoded_tw = temp_ww[new_k_w].view(-1)[0]
                        log_pw = temp_pp[new_k_w].item()
                        # decoded_tw = indexes_w[0][0][new_k_w].view(-1)[0]
                        # log_pw = log_prob_w[0][0][new_k_w].item()
                        window_new = n_top.window + [(decoded_tw.item(), decoded_ts.item())]

                        if decoded_tw > self.gen_vocab_size:
                            # generate out of gen_vocab: copy
                            temp_word_id = decoded_tw - self.gen_vocab_size - 1
                            decoded_feat = torch.LongTensor(src_wrd2fields[idx][temp_word_id]).to(
                                self.device)  # bsz x seqlen x 4
                            decoded_tw = decoded_feat[0]
                        else:
                            # generate from the gen vocab.
                            temp = self.temp_field
                            temp[0] = decoded_tw
                            decoded_feat = temp
                        # print('gen=',self.idx2word[decoded_t.item()])
                        # rnninp
                        temp_word_emb = self.inpmlp(self.word_vecs(decoded_feat).view(1, 1, -1))
                        temp_state_emb = self.state_vecs(decoded_ts).view(1, 1, -1)
                        # build rnninp
                        rnninp = torch.cat([temp_word_emb, temp_state_emb], dim=-1)
                        node = BeamSearchNode(decoder_hidden, n_top, decoded_tw, n_top.logp + log_ps + log_pw,
                                              n_top.leng + 1, state_id=decoded_ts.item(), rnninp=rnninp,
                                              window=window_new)
                        score = -node.eval()

                        if node.leng >= max_len:
                            endnodes.append((score, node))
                            if len(endnodes) >= number_required:
                                break_flag = True
                                break
                            else:
                                continue
                        else:
                            nextnodes.append((score, node))

                    if break_flag:
                        break

                for i in range(len(nextnodes)):
                    score, nn_ = nextnodes[i]
                    nodes.put((score, nn_))
                    qsize += 1

                if clean_ and qsize >= clean_size:
                    tempnodes = [nodes.get() for _ in range(min(clean_size, qsize)) if not nodes.empty()]
                    nodes = PriorityQueue()
                    qsize = 0
                    for x in tempnodes:
                        nodes.put(x)
                        qsize += 1
                elif clean_:
                    pass

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk) if not nodes.empty()]

            # back tracing
            utterances_w = []
            utterances_s = []
            scores_result = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance_w = []
                utterance_s = []
                utterance_w.append(n.wordid)
                utterance_s.append(n.state_id)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance_w.append(n.wordid)
                    utterance_s.append(n.state_id)

                utterance_w = utterance_w[::-1]
                utterance_s = utterance_s[::-1]

                utterances_w.append(utterance_w)
                utterances_s.append(utterance_s)
                scores_result.append(score)
            decoded_batch.append(utterances_w)
            decoded_states.append(utterances_s)
            decoded_score.append(scores_result)
        temp_lst = idx2word(decoded_batch, self.idx2word)
        result_str = ''
        for idx_, elem in enumerate(temp_lst):
            result_str += ' '.join(elem)
            # print(idx_)
            result_str += '|||{}'.format(decoded_score[idx_])
            result_str += '|||{}'.format(decoded_states[idx_])
            result_str += '\n'
        return result_str

    def beam_sample_template(self, condi, bsz):
        beam_width = 2
        beam_w_ = 3
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        decoded_score = []
        decoded_states = []
        max_len = 60
        window_size = 3

        clean_ = False
        clean_size = 100

        # x = condi['inps']
        # seqlen, bsz_orig, maxlocs, nfeats = x.size()
        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        src_wrd2fields = condi["src"]
        targs = condi['targs'][:, 0].unsqueeze(1)
        targs = targs.view(bsz, -1, 1)

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        layers, rnn_size = self.layers, self.h_dim
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        # print(inits.shape, self.hidden_dim, rnn_size)
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim
        c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim

        # the rnns have batch first.
        for idx in range(bsz):
            decoder_hidden = (h_prev[:, idx, :].unsqueeze(1).contiguous(), c_prev[:, idx, :].unsqueeze(1).contiguous())
            # Start with the start of the sentence token

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length, state_id, word_feats, rnninp
            node = BeamSearchNode(decoder_hidden, None, self.bos_idx, 0, 1, -1, -1, rnninp=self.bos_emb, window=[])
            node.prev_cell = None
            node.stack = None

            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1
            nfeats = 4

            # start beam search
            break_flag = False
            while True:
                # give up when we have enough good results.
                if len(endnodes) >= number_required: break
                # give up when decoding takes too long
                if qsize > 3000:
                    print('break due to huge amount')
                    break

                # fetch the best node
                if nodes.qsize() > 0:
                    score_top, n_top = nodes.get()
                    qsize -= 1
                else:
                    return 'failed --> \n'

                if n_top.wordid == self.eos_idx and n_top.prevNode != None:
                    endnodes.append((score, n_top))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                h, decoder_hidden = self.rnn(n_top.rnninp.contiguous(), n_top.h)
                out_rnn, copy = self.attn_step(h, -1, srcfieldenc[idx:idx + 1], fieldmask[idx:idx + 1], mod='state')

                state_logp = nn.LogSoftmax(dim=-1)(self.state_pick(out_rnn))
                log_prob, indexes = torch.topk(state_logp, beam_w_)
                nextnodes = []
                for new_k in range(beam_w_):
                    decoded_ts = indexes[0][0][new_k].view(-1)
                    log_ps = log_prob[0][0][new_k].item()

                    # give nthe current state, pick the vocab.
                    # print(fieldmask.shape)
                    out_rnn, copy = self.attn_step(h, decoded_ts.item(), srcfieldenc[idx:idx + 1],
                                                   fieldmask[idx:idx + 1])
                    temp_wordlp = nn.LogSoftmax(dim=-1)(torch.cat([self.vocab_pick(out_rnn), copy], dim=-1))
                    decoded_tw = targs[idx][n_top.leng - 1]
                    log_pw = temp_wordlp[0, 0, decoded_tw].item()

                    window_new = n_top.window + [(decoded_tw.item(), decoded_ts.item())]

                    if decoded_tw > self.gen_vocab_size:
                        # generate out of gen_vocab: copy
                        temp_word_id = decoded_tw - self.gen_vocab_size - 1
                        decoded_feat = torch.LongTensor(src_wrd2fields[idx][temp_word_id]).to(
                            self.device)  # bsz x seqlen x 4
                        decoded_tw = decoded_feat[0]
                    else:
                        # generate from the gen vocab.
                        temp = self.temp_field
                        temp[0] = decoded_tw
                        decoded_feat = temp
                    # rnninp
                    temp_word_emb = self.inpmlp(self.word_vecs(decoded_feat).view(1, 1, -1))
                    temp_state_emb = self.state_vecs(decoded_ts).view(1, 1, -1)
                    # build rnninp
                    rnninp = torch.cat([temp_word_emb, temp_state_emb], dim=-1)
                    node = BeamSearchNode(decoder_hidden, n_top, decoded_tw, n_top.logp + log_ps + log_pw,
                                          n_top.leng + 1, state_id=decoded_ts.item(), rnninp=rnninp,
                                          window=window_new)
                    score = -node.eval()

                    if node.leng >= max_len:
                        endnodes.append((score, node))
                        if len(endnodes) >= number_required:
                            break_flag = True
                            break
                        else:
                            continue
                    else:
                        nextnodes.append((score, node))

                if break_flag:
                    break

                for i in range(len(nextnodes)):
                    score, nn_ = nextnodes[i]
                    nodes.put((score, nn_))
                    qsize += 1

                if clean_ and qsize >= clean_size:
                    tempnodes = [nodes.get() for _ in range(min(clean_size, qsize)) if not nodes.empty()]
                    nodes = PriorityQueue()
                    qsize = 0
                    for x in tempnodes:
                        nodes.put(x)
                        qsize += 1
                elif clean_:
                    pass

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk) if not nodes.empty()]

            # back tracing
            utterances_w = []
            utterances_s = []
            scores_result = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance_w = []
                utterance_s = []
                utterance_w.append(n.wordid)
                utterance_s.append(n.state_id)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance_w.append(n.wordid)
                    utterance_s.append(n.state_id)

                utterance_w = utterance_w[::-1]
                utterance_s = utterance_s[::-1]

                utterances_w.append(utterance_w)
                utterances_s.append(utterance_s)
                scores_result.append(score)
            decoded_batch.append(utterances_w)
            decoded_states.append(utterances_s)
            decoded_score.append(scores_result)

        temp_lst = idx2word(decoded_batch, self.idx2word)
        result_str = ''
        for idx_, elem in enumerate(temp_lst):
            result_str += ' '.join(elem)
            # print(idx_)
            result_str += '|||{}'.format(decoded_score[idx_])
            result_str += '|||{}'.format(decoded_states[idx_])
            result_str += '\n'
        return result_str

    def attn_step(self, h, k, srcfieldenc, fieldmask, mod='vocab'):
        '''
            h is the output of the RNN: bsz * seglen * hiddem_dim
            know the current state name is k
            Thus, we need to pay attention to the srcfieldenc

            return: states_k: the output of the attention layer: size (bsz*sample_size, seglen, w_dim+h_dim)
                    ascrores: the copy attention score (bsz*sample_size, seglen,  src_field_len)
        '''
        if mod == 'vocab':
            bsz, seglen, cond_dim = h.shape
            mid_dim = self.mid_dim
            state_att_gates = self.state_att_gates[k]
            state_att_biases = self.state_att_biases[k]  # (bsz*sample, seglen, h_dim)

            attnin1 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
            attnin1 = F.tanh(attnin1).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
            srcfieldenc_ = F.tanh(self.attn_src(srcfieldenc).transpose(1,
                                                                       2))  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
            # TODO: consider also use the state Z to influcence the src side. [But this would involve more computation]
            ascores = torch.bmm(attnin1, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)

            fieldmask = fieldmask.unsqueeze(1).expand_as(ascores)
            ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)

            aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)
            ctx = torch.bmm(aprobs, srcfieldenc)  # (bsz*sample_size, seglen, w_dim)
            cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)

            state_out_gates = self.state_out_gates[k]
            state_out_biases = self.state_out_biases[k]
            states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)

            if self.sep_attn:
                state_att2_gates = self.state_att2_gates[k]  # (bsz*sample, seglen, h_dim)
                state_att2_biases = self.state_att2_biases[k]  # (bsz*sample, seglen, h_dim)
                attnin2 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                    state_att2_gates) + state_att2_biases  # (bsz*sample, seglen, h_dim)
                attnin2 = F.tanh(attnin2).view(bsz, seglen, mid_dim)
                ascores = torch.bmm(attnin2, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)
                ascores[fieldmask] = neginf

            return states_k, ascores

        else:
            bsz, seglen, cond_dim = h.shape
            mid_dim = self.mid_dim
            state_att_gates = self.state_att_gates2.expand(bsz * seglen, cond_dim, mid_dim)
            state_att_biases = self.state_att_biases2.expand(bsz * seglen, 1, mid_dim)  # (bsz*sample, seglen, h_dim)

            attnin1 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
            attnin1 = F.tanh(attnin1).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)

            srcfieldenc_ = F.tanh(
                self.attn_src2(srcfieldenc))  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)

            ascores = torch.bmm(attnin1, srcfieldenc_.transpose(1, 2))  # (bsz*sample_size, seglen,  src_field_len)

            fieldmask = fieldmask.unsqueeze(1).expand_as(ascores)
            ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
            aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)
            ctx = torch.bmm(aprobs, srcfieldenc)  # (bsz*sample_size, seglen, w_dim)
            cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)

            state_out_gates = self.state_out_gates2
            state_out_biases = self.state_out_biases2
            states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)  # (bsz*sample_size, seglen, w_dim+h_dim)

            return states_k, None

    def get_attn2(self, h, srcfieldenc, fieldmask):
        # batch first
        '''
        Implemnet the copy attention mechanism as well as the local attention mechanism.
        :param h: (bsz*sample_size, seqlen+state_len ,hidden_dim)
        :param k: (bsz*sample_size, seqlen+state_len)
        :param srcfieldenc: (bsz, src_field_len, w_dim)
        :param fieldmask: (bsz, src_field_len)
        :return: the output after we apply the attention layer. (bsz*sample_size, seqlen+state_len ,hidden_dim*2)
        '''
        # print(k.shape, h.shape, srcfieldenc.shape, fieldmask.shape)
        bsz, seglen, cond_dim = h.shape
        mid_dim = self.mid_dim
        state_att_gates = self.state_att_gates2.expand(bsz * seglen, cond_dim, mid_dim)
        state_att_biases = self.state_att_biases2.expand(bsz * seglen, 1, mid_dim)  # (bsz*sample, seglen, h_dim)
        attnin1 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                            state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
        attnin1 = F.tanh(attnin1).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)

        srcfieldenc_ = F.tanh(self.attn_src2(srcfieldenc))  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
        srcfieldenc_ = torch.repeat_interleave(srcfieldenc_, self.sample_size,
                                               dim=0).transpose(1, 2)  # (bsz*sample_size, src_field_len, m_dim)

        ascores = torch.bmm(attnin1, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)

        fieldmask = torch.repeat_interleave(fieldmask, self.sample_size, dim=0)  # (bsz*sample_size, src_field_len)
        fieldmask = fieldmask.unsqueeze(1).expand_as(ascores)
        ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
        aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)
        srcfieldenc = torch.repeat_interleave(srcfieldenc, self.sample_size, dim=0)
        ctx = torch.bmm(aprobs, srcfieldenc)  # (bsz*sample_size, seglen, w_dim)
        cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)

        # TODO: gated version -> matrix version.
        state_out_gates = self.state_out_gates2.expand(bsz, seglen, -1)
        state_out_biases = self.state_out_biases2.expand(bsz, seglen, -1)
        states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)  # (bsz*sample_size, seglen, w_dim+h_dim)

        return states_k

    def force_pr(self, copy_attn, state_names, src_detail_mask):
        '''

        :param copy_attn: the unnormalized log score of the copy attention. (bsz*sample_size, seqlen, src_field_len)
        :param state_names: (bsz, sample_size, seqlen)
        :param src_detail_mask: (bsz, tagset_size, field_len)
        :return: A masked version of copy_attn. (bsz*sample_size, seqlen, src_field_len)
        '''
        bsz, sample_size, seqlen = state_names.size()
        state_names = state_names.view(bsz, -1)  # bsz x (sample_size*seqlen)
        # src_state_mask = torch.repeat_interleave(src_detail_mask, self.sample_size,
        #                                       dim=0)  # (bsz*sample_size, src_field_len)
        # print(src_state_mask.shape)
        # temp = torch.index_select(src_state_mask, 1, state_names.view(-1))
        lst = []
        for b in range(bsz):
            temp = torch.index_select(src_detail_mask[b], 0, state_names[b].view(-1))
            lst.append(temp)
        lst = torch.stack(lst, dim=0).view(bsz * sample_size, seqlen, -1)  # (bsz*sample_size) x seqlen x field_len
        return copy_attn + lst  # (bsz*sample_size) x seqlen x field_len

    def get_attn(self, h, k, srcfieldenc, fieldmask):
        # batch first
        '''
        Implemnet the copy attention mechanism as well as the local attention mechanism.
        :param h: (bsz*sample_size, seqlen+state_len ,hidden_dim)
        :param k: (bsz*sample_size, seqlen+state_len)
        :param srcfieldenc: (bsz, src_field_len, w_dim)
        :param fieldmask: (bsz, src_field_len)
        :return: the output after we apply the attention layer. (bsz*sample_size, seqlen+state_len ,hidden_dim*2)
        '''
        # print(k.shape, h.shape, srcfieldenc.shape, fieldmask.shape)
        bsz, seglen, cond_dim = h.shape
        mid_dim = self.mid_dim
        state_att_gates = torch.index_select(self.state_att_gates, 0, k.view(-1)).view(bsz * seglen, cond_dim,
                                                                                       mid_dim)  # (bsz*sample, seglen, h_dim)
        state_att_biases = torch.index_select(self.state_att_biases, 0, k.view(-1)).view(bsz * seglen, 1,
                                                                                         mid_dim)  # (bsz*sample, seglen, h_dim)
        attnin1 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                            state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
        attnin1 = F.tanh(attnin1).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)

        srcfieldenc_ = F.tanh(self.attn_src(srcfieldenc))  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
        srcfieldenc_ = torch.repeat_interleave(srcfieldenc_, self.sample_size, dim=0).transpose(1,
                                                                                                2)  # (bsz*sample_size, src_field_len, m_dim)

        # TODO: consider also use the state Z to influcence the src side. [But this would involve more computation]
        # state_att_src_gates = torch.index_select(self.state_att_gates, 0, k.view(-1)).view(bsz * seglen, cond_dim, mid_dim)  # (bsz*sample, seglen, h_dim)
        # state_att_src_biases = torch.index_select(self.state_att_biases, 0, k.view(-1)).view(bsz * seglen, 1,  mid_dim)  # (bsz*sample, seglen, h_dim)
        # attnin2 = torch.bmm(h.view(bsz * seglen, 1, cond_dim),
        #                     state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
        # attnin2 = F.tanh(attnin2)  # (bsz*sample, seglen, h_dim)

        ascores = torch.bmm(attnin1, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)

        fieldmask = torch.repeat_interleave(fieldmask, self.sample_size, dim=0)  # (bsz*sample_size, src_field_len)
        fieldmask = fieldmask.unsqueeze(1).expand_as(ascores)
        ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)

        aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)
        srcfieldenc = torch.repeat_interleave(srcfieldenc, self.sample_size, dim=0)
        ctx = torch.bmm(aprobs, srcfieldenc)  # (bsz*sample_size, seglen, w_dim)
        cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)
        # TODO: gated version -> matrix version.
        state_out_gates = torch.index_select(self.state_out_gates, 0, k.view(-1)).view(bsz, seglen, -1)
        state_out_biases = torch.index_select(self.state_out_biases, 0, k.view(-1)).view(bsz, seglen, -1)

        states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)  # (bsz*sample_size, seglen, w_dim+h_dim)

        if self.sep_attn:
            state_att2_gates = torch.index_select(self.state_att2_gates, 0, k.view(-1)).view(bsz * seglen, cond_dim,
                                                                                             mid_dim)  # (bsz*sample, seglen, h_dim)
            state_att2_biases = torch.index_select(self.state_att2_biases, 0, k.view(-1)).view(bsz * seglen, 1,
                                                                                               mid_dim)  # (bsz*sample, seglen, h_dim)
            attnin2 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                state_att2_gates) + state_att2_biases  # (bsz*sample, seglen, h_dim)
            attnin2 = F.tanh(attnin2).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)

            ascores = torch.bmm(attnin2, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)
            ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)

        return states_k, ascores

    def forward_easy_dual(self, condi, sample_lst):
        '''
            condi: the representation of x and y.
            sample_lst: the representation of z.
        '''

        # TODO: validate the unsupervised copy attention idea.
        x = condi['inps'][:, :, 0, 0]
        seqlen, bsz_orig = x.size()
        targs = condi['targs']
        targs = targs.view(bsz_orig, seqlen, -1)
        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        field_vecs = condi['field_vecs']
        fieldmask = condi["fmask"]

        bsz_orig, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.h_dim
        self.sample_size = len(sample_lst[0])
        state_names = self.get_state_seq(sample_lst, seqlen).long()

        # create input to RNN by [word, next state]
        # TODO: tgt embeddings be replaced.
        ##1
        # word_embs = self.word_vecs(x)
        # word_embs = self.dropout(word_embs)
        # inpembs = self.inpmlp(word_embs.view(seqlen, bsz_orig, maxlocs, -1)).mean(2)

        word_embs = condi['tgt_word']
        inpembs = self.dropout(word_embs)

        # inpembs = torch.cat([self.bos_emb.expand(1, bsz_orig, -1), inpembs], dim=0)
        inpembs = inpembs.repeat_interleave(self.sample_size, dim=1).transpose(0, 1)  # bsz * seqlen * dim
        state_embs = self.state_vecs(state_names).view(bsz_orig * self.sample_size, seqlen, -1)  # bsz * seqlen * dim
        state_embs = self.dropout(state_embs)

        rnn_inp = torch.cat([inpembs, state_embs], dim=-1)  # bsz x seqlen x nfeats*emb + state_embs
        rnn_inp = torch.cat([self.bos_emb.expand(bsz_orig * self.sample_size, 1, -1), rnn_inp], dim=1)
        # print(rnn_inp.shape)

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
        h_prev = torch.repeat_interleave(h_prev, self.sample_size, dim=1)  # layer * (bsz*sample_size) * dim
        c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
        c_prev = torch.repeat_interleave(c_prev, self.sample_size, dim=1)  # layer * (bsz*sample_size) * dim

        h, _ = self.rnn(rnn_inp, (h_prev, c_prev))  # h.shape = (batch_size * sample_size, seqlen, h_dim)
        h = self.dropout(h)
        h1, copy_attn = self.get_attn_dual(h[:, :-1], state_names, srcfieldenc, fieldmask, field_vecs)

        # LLISA
        if self.decoder_constraint:
            detailed_src_mask = condi['detailed_src_mask']
            copy_attn = self.force_pr(copy_attn, state_names, detailed_src_mask)

        # pick vocab
        wlps_k = F.log_softmax(torch.cat([self.vocab_pick(h1), copy_attn], -1), dim=-1)  # (bsz*seqlen, nfield)
        wlps_k = torch.cat([wlps_k, self.zeros.expand(wlps_k.size(0), seqlen, -1)], -1)
        targs = torch.repeat_interleave(targs, self.sample_size, dim=0)
        logp_word_temp = torch.gather(wlps_k, 2, targs)
        logp_word = torch.sum(logsumexp2(logp_word_temp), dim=-1)

        # pick state
        h2 = self.get_attn2_dual(h[:, :-1], srcfieldenc, fieldmask, field_vecs)
        slps_k = F.log_softmax(self.state_pick(h2), dim=-1)
        targstate = state_names.view(bsz_orig * self.sample_size, seqlen, 1)
        logp_state_temp = torch.gather(slps_k, 2, targstate)
        # print(logp_state_temp.shape)
        logp_state = logp_state_temp.sum(-1).sum(-1)
        # print(logp_state.shape)

        result_dict = {'p(y)': logp_word.view(bsz_orig, -1), 'p(z)': logp_state.view(bsz_orig, -1)}
        # print(result_dict)
        return result_dict

    def get_attn2_dual(self, h, srcfieldenc, fieldmask, field_vecs):
        # batch first
        '''
        Implemnet the copy attention mechanism as well as the local attention mechanism.
        :param h: (bsz*sample_size, seqlen+state_len ,hidden_dim)
        :param k: (bsz*sample_size, seqlen+state_len)
        :param srcfieldenc: (bsz, src_field_len, w_dim)
        :param fieldmask: (bsz, src_field_len)
        :return: the output after we apply the attention layer. (bsz*sample_size, seqlen+state_len ,hidden_dim*2)
        '''
        # print(k.shape, h.shape, srcfieldenc.shape, fieldmask.shape)
        bsz, seglen, cond_dim = h.shape
        mid_dim = self.mid_dim
        state_att_gates = self.state_att_gates2.expand(bsz * seglen, cond_dim, mid_dim)
        state_att_biases = self.state_att_biases2.expand(bsz * seglen, 1, mid_dim)  # (bsz*sample, seglen, h_dim)
        attnin1 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                            state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
        attnin1 = F.tanh(attnin1).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)

        srcfieldenc_ = F.tanh(self.attn_src2(srcfieldenc))  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
        srcfieldenc_ = torch.repeat_interleave(srcfieldenc_, self.sample_size,
                                               dim=0).transpose(1, 2)  # (bsz*sample_size, src_field_len, m_dim)

        ascores = torch.bmm(attnin1, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)

        fieldmask = torch.repeat_interleave(fieldmask, self.sample_size, dim=0)  # (bsz*sample_size, src_field_len)
        fieldmask = fieldmask.unsqueeze(1).expand_as(ascores)
        ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
        aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

        # get the beta scores.
        state_att_gates_dual = self.state_att_gates2_dual.expand(bsz * seglen, cond_dim, mid_dim)
        state_att_biases_dual = self.state_att_biases2_dual.expand(bsz * seglen, 1,
                                                                   mid_dim)  # (bsz*sample, seglen, h_dim)
        attnin1_dual = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                 state_att_gates_dual) + state_att_biases_dual  # (bsz*sample, seglen, h_dim)
        attnin1_dual = F.tanh(attnin1_dual).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
        srcfieldenc_dual = F.tanh(
            self.attn_src2_dual(field_vecs))  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
        srcfieldenc_dual = torch.repeat_interleave(srcfieldenc_dual, self.sample_size,
                                                   dim=0).transpose(1, 2)  # (bsz*sample_size, src_field_len, m_dim)
        ascores_dual = torch.bmm(attnin1_dual, srcfieldenc_dual)  # (bsz*sample_size, seglen,  src_field_len)
        ascores_dual[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
        aprobs_dual = F.softmax(ascores_dual, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

        # combine the alphas and betas.
        probs_total = aprobs * aprobs_dual
        probs_total = probs_total / (probs_total.sum(2).unsqueeze(2).expand(probs_total.shape) + 1e-6)

        srcfieldenc = torch.repeat_interleave(srcfieldenc, self.sample_size, dim=0)
        ctx = torch.bmm(probs_total, srcfieldenc)  # (bsz*sample_size, seglen, w_dim)
        cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)

        # TODO: gated version -> matrix version.
        state_out_gates = self.state_out_gates2.expand(bsz, seglen, -1)
        state_out_biases = self.state_out_biases2.expand(bsz, seglen, -1)
        states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)  # (bsz*sample_size, seglen, w_dim+h_dim)

        return states_k

    def get_attn_dual(self, h, k, srcfieldenc, fieldmask, field_vecs):
        # batch first
        '''
        Implemnet the copy attention mechanism as well as the local attention mechanism.
        :param h: (bsz*sample_size, seqlen+state_len ,hidden_dim)
        :param k: (bsz*sample_size, seqlen+state_len)
        :param srcfieldenc: (bsz, src_field_len, w_dim)
        :param fieldmask: (bsz, src_field_len)
        :return: the output after we apply the attention layer. (bsz*sample_size, seqlen+state_len ,hidden_dim*2)
        '''
        # print(k.shape, h.shape, srcfieldenc.shape, fieldmask.shape)
        bsz, seglen, cond_dim = h.shape
        mid_dim = self.mid_dim
        state_att_gates = torch.index_select(self.state_att_gates, 0, k.view(-1)).view(bsz * seglen, cond_dim, mid_dim)
        # (bsz*sample, seglen, h_dim)
        state_att_biases = torch.index_select(self.state_att_biases, 0, k.view(-1)).view(bsz * seglen, 1, mid_dim)
        attnin1 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                            state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
        attnin1 = F.tanh(attnin1).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
        srcfieldenc_ = F.tanh(self.attn_src(srcfieldenc))  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
        # (bsz*sample_size, src_field_len, m_dim)
        srcfieldenc_ = torch.repeat_interleave(srcfieldenc_, self.sample_size, dim=0).transpose(1, 2)
        ascores = torch.bmm(attnin1, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)
        fieldmask = torch.repeat_interleave(fieldmask, self.sample_size, dim=0)  # (bsz*sample_size, src_field_len)
        fieldmask = fieldmask.unsqueeze(1).expand_as(ascores)
        ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
        aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

        # the beta scores.
        state_att_gates_dual = torch.index_select(self.state_att_gates_dual, 0, k.view(-1)).view(bsz * seglen, cond_dim,
                                                                                                 mid_dim)
        state_att_biases_dual = torch.index_select(self.state_att_biases_dual, 0, k.view(-1)).view(bsz * seglen, 1,
                                                                                                   mid_dim)
        attnin1_dual = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                 state_att_gates_dual) + state_att_biases_dual  # (bsz*sample, seglen, h_dim)
        attnin1_dual = F.tanh(attnin1_dual).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
        srcfieldenc_dual = F.tanh(
            self.attn_src_dual(field_vecs))  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
        srcfieldenc_dual = torch.repeat_interleave(srcfieldenc_dual, self.sample_size, dim=0).transpose(1, 2)

        ascores_dual = torch.bmm(attnin1_dual, srcfieldenc_dual)  # (bsz*sample_size, seglen,  src_field_len)
        ascores_dual[fieldmask] = neginf
        aprobs_dual = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

        # combine alphas and betas from dual.
        final_probs = aprobs_dual * aprobs
        final_probs = final_probs / (final_probs.sum(2).unsqueeze(2).expand(final_probs.shape) + 1e-6)

        srcfieldenc = torch.repeat_interleave(srcfieldenc, self.sample_size, dim=0)
        ctx = torch.bmm(final_probs, srcfieldenc)  # (bsz*sample_size, seglen, w_dim)
        cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)

        # TODO: gated version -> matrix version.
        state_out_gates = torch.index_select(self.state_out_gates, 0, k.view(-1)).view(bsz, seglen, -1)
        state_out_biases = torch.index_select(self.state_out_biases, 0, k.view(-1)).view(bsz, seglen, -1)

        states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)  # (bsz*sample_size, seglen, w_dim+h_dim)

        if self.sep_attn:
            state_att2_gates = torch.index_select(self.state_att2_gates, 0, k.view(-1)).view(bsz * seglen, cond_dim,
                                                                                             mid_dim)  # (bsz*sample, seglen, h_dim)
            state_att2_biases = torch.index_select(self.state_att2_biases, 0, k.view(-1)).view(bsz * seglen, 1,
                                                                                               mid_dim)  # (bsz*sample, seglen, h_dim)
            attnin2 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                state_att2_gates) + state_att2_biases  # (bsz*sample, seglen, h_dim)
            attnin2 = F.tanh(attnin2).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
            ascores = torch.bmm(attnin2, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)

            # TODO: unsure about this part: do we need copy attention for dual.
            state_att_gates_dual = torch.index_select(self.state_att2_gates_dual, 0, k.view(-1)).view(bsz * seglen,
                                                                                                      cond_dim, mid_dim)
            state_att_biases_dual = torch.index_select(self.state_att2_biases_dual, 0, k.view(-1)).view(bsz * seglen, 1,
                                                                                                        mid_dim)
            attnin2_dual = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                     state_att_gates_dual) + state_att_biases_dual  # (bsz*sample, seglen, h_dim)
            attnin2_dual = F.tanh(attnin2_dual).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)

            ascores_dual = torch.bmm(attnin2_dual, srcfieldenc_dual)  # (bsz*sample_size, seglen,  src_field_len)
            ascores = ascores + ascores_dual
            ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)

        return states_k, ascores

    def beam_forward_easy_dual3(self, condi, bsz, timing=None):
        beam_width = 5
        beam_w_ = 1
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        decoded_score = []
        decoded_states = []
        decoded_copy = []
        max_len = 60
        window_size = 3

        time_a = time.time()

        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        field_vecs = condi['field_vecs']
        fieldmask = condi["fmask"]
        src_wrd2fields = condi["src"].cpu()

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        layers, rnn_size = self.layers, self.h_dim
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        # print(inits.shape, self.hidden_dim, rnn_size)
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim
        c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim

        time_b = time.time()
        timing['pre_comp'] += time_b - time_a

        # the rnns have batch first.
        for idx in range(bsz):
            decoder_hidden = (h_prev[:, idx, :].unsqueeze(1).contiguous(), c_prev[:, idx, :].unsqueeze(1).contiguous())
            # Start with the start of the sentence token

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            time_c = time.time()

            srcfieldenc_w = F.tanh(self.attn_src(srcfieldenc[idx:idx + 1]))
            srcfieldenc_s = F.tanh(self.attn_src2(srcfieldenc[idx:idx + 1]))

            srcfieldenc_dual_w = F.tanh(self.attn_src_dual(field_vecs[idx:idx + 1]))
            srcfieldenc_dual_s = F.tanh(self.attn_src2_dual(field_vecs[idx:idx + 1]))

            time_d = time.time()
            timing['pre_comp'] += time_d - time_c

            # starting node -  hidden vector, previous node, word id, logp, length, state_id, word_feats, rnninp
            node = BeamSearchNode(decoder_hidden, None, self.bos_idx, 0, 1, -1, -1, rnninp=self.bos_emb, window=[])
            node.record = -1
            node.prev_cell = None
            node.stack = None
            prev_nodes = []
            prev_nodes.append((-node.eval(), node))
            # qsize = 1
            # start beam search
            break_flag = False
            while True:
                # give up when we have enough good results.
                if len(endnodes) >= number_required: break

                # fetch the best node
                nextnodes = []
                for elem in range(min(len(prev_nodes), beam_width)):
                    score_top, n_top = prev_nodes[elem]

                    if n_top.wordid == self.eos_idx and n_top.prevNode != None:
                        endnodes.append((score, n_top))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    time_e = time.time()

                    h, decoder_hidden = self.rnn(n_top.rnninp.contiguous(), n_top.h)
                    out_rnn, copy = self.attn_step_dual(h, -1, srcfieldenc[idx:idx + 1], fieldmask[idx:idx + 1],
                                                        srcfieldenc_s, srcfieldenc_dual_s, mod='state')
                    state_logp = nn.LogSoftmax(dim=-1)(self.state_pick(out_rnn))
                    log_prob, indexes = torch.topk(state_logp, beam_w_)

                    time_f = time.time()

                    timing['rnnstate_comp'] += time_f - time_e

                    for new_k in range(beam_w_):

                        time_aa = time.time()
                        decoded_ts = indexes.view(-1)[new_k]
                        log_ps = log_prob[0][0][new_k].item()

                        out_rnn, copy = self.attn_step_dual(h, decoded_ts.item(), srcfieldenc[idx:idx + 1],
                                                            fieldmask[idx:idx + 1], srcfieldenc_w, srcfieldenc_dual_w)
                        temp_wordlp = nn.LogSoftmax(dim=-1)(torch.cat([self.vocab_pick(out_rnn), copy], dim=-1))
                        log_prob_w, indexes_w = torch.topk(temp_wordlp, beam_width + window_size)

                        time_bb = time.time()
                        timing['word_comp'] += time_bb - time_aa

                        temp_ww = []
                        temp_pp = []

                        time_cc = time.time()
                        ''' this section should be moved to CPU. '''
                        for elem1, elem2 in zip(indexes_w.cpu().view(-1), log_prob_w.cpu().view(-1)):
                            if elem1.item() in [x[0] for x in n_top.window[-window_size:]]:
                                pass
                            else:
                                temp_ww.append(elem1)
                                temp_pp.append(elem2)
                            if len(temp_ww) >= beam_width:
                                break

                        for new_k_w in range(beam_width):

                            decoded_tw = temp_ww[new_k_w].view(-1)[0]
                            log_pw = temp_pp[new_k_w].item()
                            window_new = n_top.window + [(decoded_tw.item(), decoded_ts.item())]

                            if decoded_tw > self.gen_vocab_size:
                                # generate out of gen_vocab: copy
                                temp_word_id = decoded_tw - self.gen_vocab_size - 1
                                decoded_feat = src_wrd2fields[idx][temp_word_id]  # bsz x seqlen x 4
                                # decoded_feat = torch.LongTensor(src_wrd2fields[idx][temp_word_id]) # bsz x seqlen x 4
                                decoded_tw = decoded_feat[0]
                                record = temp_word_id.item()
                            else:
                                # generate from the gen vocab.
                                temp = self.temp_field
                                temp[0] = decoded_tw
                                decoded_feat = temp
                                record = -1

                            temp_word_emb = self.get_word_embs(decoded_feat.to(self.device)).view(1, 1, -1)
                            temp_state_emb = self.state_vecs(decoded_ts).view(1, 1, -1)
                            rnninp = torch.cat([temp_word_emb, temp_state_emb], dim=-1)
                            node = BeamSearchNode(decoder_hidden, n_top, decoded_tw, n_top.logp + log_ps + log_pw,
                                                  n_top.leng + 1, state_id=decoded_ts.item(), rnninp=rnninp,
                                                  window=window_new)
                            node.record = record
                            score = -node.eval()

                            if node.leng >= max_len:
                                endnodes.append((score, node))
                                if len(endnodes) >= number_required:
                                    break_flag = True
                                    break
                                else:
                                    continue
                            else:
                                nextnodes.append((score, node))

                        time_dd = time.time()
                        timing['word_grp'] += time_dd - time_cc
                        if break_flag:
                            break

                    if break_flag:
                        break

                next_nodes = []
                for i in range(len(nextnodes)):
                    score, nn_ = nextnodes[i]
                    next_nodes.append((score, nn_))
                    # qsize += 1
                # sort next_nodes.
                next_nodes = sorted(next_nodes, key=lambda x: x[0])
                prev_nodes = next_nodes

            # back tracing
            time_ee = time.time()
            utterances_w = []
            utterances_s = []
            scores_result = []
            copys = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance_w = []
                utterance_s = []
                copy = []
                utterance_w.append(n.wordid)
                utterance_s.append(n.state_id)
                copy.append(n.record)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance_w.append(n.wordid)
                    utterance_s.append(n.state_id)
                    copy.append(n.record)

                utterance_w = utterance_w[::-1]
                utterance_s = utterance_s[::-1]
                copy = copy[::-1]
                utterances_w.append(utterance_w)
                utterances_s.append(utterance_s)
                scores_result.append(score)
                copys.append(copy)

            decoded_batch.append(utterances_w)
            decoded_states.append(utterances_s)
            decoded_score.append(scores_result)
            decoded_copy.append(copys)
            time_ff = time.time()
            timing['back_trace'] += time_ff - time_ee

        time_gg = time.time()
        temp_lst = idx2word(decoded_batch, self.idx2word)

        final_states = []
        len_marks = []
        for x in decoded_states:
            final_states += x
            len_marks += [y for y in range(len(x))]
        final_scores = []
        for x in decoded_score:
            final_scores += x
        final_copy = []
        for x in decoded_copy:
            final_copy += x

        result_str = ''
        for idx_, elem in enumerate(temp_lst):
            result_str += ' '.join(elem)
            result_str += '|||{}'.format(final_scores[idx_])
            result_str += '|||{}'.format(final_states[idx_])
            result_str += '|||{}'.format(len_marks[idx_])
            result_str += '|||{}'.format(final_copy[idx_])
            result_str += '\n'
        time_ii = time.time()

        timing['finalstep'] += time_ii - time_gg
        return result_str

    def beam_forward_easy_dual(self, condi, bsz, timing=None):
        beam_width = 3
        beam_w_ = 2
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        decoded_score = []
        decoded_states = []
        decoded_copy = []
        max_len = 60
        window_size = 3

        time_a = time.time()

        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        field_vecs = condi['field_vecs']
        fieldmask = condi["fmask"]
        src_wrd2fields = condi["src"].cpu()

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        layers, rnn_size = self.layers, self.h_dim
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        # print(inits.shape, self.hidden_dim, rnn_size)
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim
        c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim

        srcfieldenc_w_full = F.tanh(self.attn_src(srcfieldenc))
        srcfieldenc_s_full = F.tanh(self.attn_src2(srcfieldenc))

        srcfieldenc_dual_w_full = F.tanh(self.attn_src_dual(field_vecs))
        srcfieldenc_dual_s_full = F.tanh(self.attn_src2_dual(field_vecs))

        time_b = time.time()
        timing['pre_comp'] += time_b - time_a

        # the rnns have batch first.
        for idx in range(bsz):
            decoder_hidden = (h_prev[:, idx:idx + 1, :].contiguous(), c_prev[:, idx:idx + 1, :].contiguous())
            # Start with the start of the sentence token
            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            time_c = time.time()

            srcfieldenc_w = srcfieldenc_w_full[idx:idx + 1]
            srcfieldenc_s = srcfieldenc_s_full[idx:idx + 1]

            srcfieldenc_dual_w = srcfieldenc_dual_w_full[idx:idx + 1]
            srcfieldenc_dual_s = srcfieldenc_dual_s_full[idx:idx + 1]

            time_d = time.time()
            timing['pre_comp'] += time_d - time_c
            # starting node -  hidden vector, previous node, word id, logp, length, state_id, word_feats, rnninp
            node = {'h': decoder_hidden, 'prevNode': None, 'wordid': self.bos_idx, 'logp': 0, 'leng': 1,
                    'state_id': -1, 'word_feats': -1, 'rnninp': self.bos_emb, 'window': [], 'record': -1}
            # node = BeamSearchNode(decoder_hidden, None, self.bos_idx, 0, 1, -1, -1, rnninp=self.bos_emb, window=[])
            prev_nodes = []
            prev_nodes.append((-node['logp'], node))
            # start beam search
            break_flag = False
            while True:
                # give up when we have enough good results.
                if len(endnodes) >= number_required: break

                # fetch the best node
                nextnodes = []
                for elem in range(min(len(prev_nodes), beam_width)):
                    score_top, n_top = prev_nodes[elem]

                    time_e = time.time()

                    h, decoder_hidden = self.rnn(n_top['rnninp'].contiguous(), n_top['h'])
                    out_rnn, copy = self.attn_step_dual(h, -1, srcfieldenc[idx:idx + 1], fieldmask[idx:idx + 1],
                                                        srcfieldenc_s, srcfieldenc_dual_s, mod='state')
                    state_logp = torch.log_softmax(self.state_pick(out_rnn), dim=-1)
                    log_prob, indexes = torch.topk(state_logp, beam_w_)

                    time_f = time.time()

                    timing['rnnstate_comp'] += time_f - time_e

                    for new_k in range(beam_w_):

                        time_aa = time.time()
                        # print(indexes)
                        # print(log_prob)
                        decoded_ts = indexes.view(-1)[new_k]
                        log_ps = log_prob[0][0][new_k].item()

                        out_rnn, copy = self.attn_step_dual(h, decoded_ts.item(), srcfieldenc[idx:idx + 1],
                                                            fieldmask[idx:idx + 1], srcfieldenc_w, srcfieldenc_dual_w)

                        temp_wordlp = torch.log_softmax(torch.cat([self.vocab_pick(out_rnn), copy], dim=-1), dim=-1)

                        log_prob_w, indexes_w = torch.topk(temp_wordlp, beam_width + window_size + 1)

                        time_bb = time.time()
                        timing['word_comp'] += time_bb - time_aa

                        temp_ww = []
                        temp_pp = []

                        time_cc = time.time()
                        ''' this section should be moved to CPU. '''
                        for elem1, elem2 in zip(indexes_w.cpu().view(-1), log_prob_w.cpu().view(-1)):
                            if elem1.item() in [x[0] for x in
                                                n_top['window'][-window_size:]] or elem1.item() == self.unk:
                                continue
                            else:
                                temp_ww.append(elem1)
                                temp_pp.append(elem2)

                            if len(temp_ww) >= beam_width:
                                break

                        for new_k_w in range(beam_width):

                            decoded_tw = temp_ww[new_k_w].view(-1)[0]
                            log_pw = temp_pp[new_k_w].item()
                            window_new = n_top['window'] + [(decoded_tw.item(), decoded_ts.item())]

                            if decoded_tw > self.gen_vocab_size:
                                # generate out of gen_vocab: copy
                                temp_word_id = decoded_tw - self.gen_vocab_size - 1
                                decoded_feat = src_wrd2fields[idx][temp_word_id]  # bsz x seqlen x 4
                                # decoded_feat = torch.LongTensor(src_wrd2fields[idx][temp_word_id]) # bsz x seqlen x 4
                                decoded_tw = decoded_feat[0]
                                record = temp_word_id.item()
                            else:
                                # generate from the gen vocab.
                                # temp = self.temp_field
                                # temp[0] = decoded_tw
                                # decoded_feat = temp
                                record = -1
                                decoded_feat = -1

                            # rnninp
                            temp_word_emb = self.get_word_embs(decoded_tw.to(self.device)).view(1, 1, -1)
                            temp_state_emb = self.state_vecs(decoded_ts).view(1, 1, -1)
                            rnninp = torch.cat([temp_word_emb, temp_state_emb], dim=-1)
                            node = {'h': decoder_hidden, 'prevNode': n_top, 'wordid': decoded_tw,
                                    'logp': n_top['logp'] + log_ps + log_pw, 'leng': n_top['leng'] + 1,
                                    'state_id': decoded_ts.item(), 'word_feats': -1, 'rnninp': rnninp,
                                    'window': window_new, 'record': record}
                            # node = BeamSearchNode(decoder_hidden, n_top, decoded_tw, n_top.logp + log_ps + log_pw,
                            #                       n_top.leng + 1, state_id=decoded_ts.item(), rnninp=rnninp,
                            #                       window=window_new)
                            # node.record = record
                            # score = -node.eval()
                            score = -node['logp']

                            if n_top['wordid'] == self.eos_idx and n_top['prevNode'] != None:
                                endnodes.append((score, node))
                                # if we reached maximum # of sentences required
                                if len(endnodes) >= number_required:
                                    break
                                else:
                                    continue

                            if node['leng'] >= max_len:
                                endnodes.append((score, node))
                                if len(endnodes) >= number_required:
                                    break_flag = True
                                    break
                                else:
                                    continue
                            else:
                                nextnodes.append((score, node))

                        time_dd = time.time()
                        timing['word_grp'] += time_dd - time_cc
                        if break_flag:
                            break

                    if break_flag:
                        break

                next_nodes = []
                for i in range(len(nextnodes)):
                    score, nn_ = nextnodes[i]
                    next_nodes.append((score, nn_))
                    # qsize += 1
                # sort next_nodes.
                next_nodes = sorted(next_nodes, key=lambda x: x[0])
                prev_nodes = next_nodes

            # back tracing
            time_ee = time.time()
            utterances_w = []
            utterances_s = []
            scores_result = []
            copys = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance_w = []
                utterance_s = []
                copy = []
                utterance_w.append(n['wordid'])
                utterance_s.append(n['state_id'])
                copy.append(n['record'])
                # back trace
                while n['prevNode'] != None:
                    n = n['prevNode']
                    utterance_w.append(n['wordid'])
                    utterance_s.append(n['state_id'])
                    copy.append(n['record'])

                utterance_w = utterance_w[::-1]
                utterance_s = utterance_s[::-1]
                copy = copy[::-1]
                utterances_w.append(utterance_w)
                utterances_s.append(utterance_s)
                scores_result.append(score)
                copys.append(copy)

            decoded_batch.append(utterances_w)
            decoded_states.append(utterances_s)
            decoded_score.append(scores_result)
            decoded_copy.append(copys)
            time_ff = time.time()
            timing['back_trace'] += time_ff - time_ee

        time_gg = time.time()
        temp_lst = idx2word(decoded_batch, self.idx2word)

        final_states = []
        len_marks = []
        for x in decoded_states:
            final_states += x
            len_marks += [y for y in range(len(x))]
        final_scores = []
        for x in decoded_score:
            final_scores += x
        final_copy = []
        for x in decoded_copy:
            final_copy += x

        result_str = ''
        for idx_, elem in enumerate(temp_lst):
            result_str += ' '.join(elem)
            result_str += '|||{}'.format(final_scores[idx_])
            result_str += '|||{}'.format(final_states[idx_])
            result_str += '|||{}'.format(len_marks[idx_])
            result_str += '|||{}'.format(final_copy[idx_])
            result_str += '\n'
        time_ii = time.time()

        timing['finalstep'] += time_ii - time_gg
        return result_str

    def beam_forward_easy_dual4(self, condi, bsz, timing=None):
        beam_width = 3
        beam_w_ = 3
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        decoded_score = []
        decoded_states = []
        decoded_copy = []
        max_len = 60
        window_size = 3

        time_a = time.time()

        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        field_vecs = condi['field_vecs']
        fieldmask = condi["fmask"]
        src_wrd2fields = condi["src"].cpu()

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        layers, rnn_size = self.layers, self.h_dim
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        # print(inits.shape, self.hidden_dim, rnn_size)
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim
        c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim

        srcfieldenc_w_full = F.tanh(self.attn_src(srcfieldenc))
        srcfieldenc_s_full = F.tanh(self.attn_src2(srcfieldenc))

        srcfieldenc_dual_w_full = F.tanh(self.attn_src_dual(field_vecs))
        srcfieldenc_dual_s_full = F.tanh(self.attn_src2_dual(field_vecs))

        time_b = time.time()
        timing['pre_comp'] += time_b - time_a

        # the rnns have batch first.
        for idx in range(bsz):
            decoder_hidden = (h_prev[:, idx:idx + 1, :].contiguous(), c_prev[:, idx:idx + 1, :].contiguous())
            # Start with the start of the sentence token
            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            time_c = time.time()

            srcfieldenc_w = srcfieldenc_w_full[idx:idx + 1]
            srcfieldenc_s = srcfieldenc_s_full[idx:idx + 1]

            srcfieldenc_dual_w = srcfieldenc_dual_w_full[idx:idx + 1]
            srcfieldenc_dual_s = srcfieldenc_dual_s_full[idx:idx + 1]

            time_d = time.time()
            timing['pre_comp'] += time_d - time_c

            # starting node -  hidden vector, previous node, word id, logp, length, state_id, word_feats, rnninp
            node = {'h': decoder_hidden, 'prevNode': None, 'wordid': self.bos_idx, 'logp': 0., 'leng': 1,
                    'state_id': -1, 'word_feats': -1, 'rnninp': self.bos_emb, 'window': [], 'record': -1}
            # node = BeamSearchNode(decoder_hidden, None, self.bos_idx, 0, 1, -1, -1, rnninp=self.bos_emb, window=[])
            prev_nodes = []
            prev_nodes.append((-node['logp'], node))
            # start beam search
            break_flag = False
            while True:
                # give up when we have enough good results.
                if len(endnodes) >= number_required: break

                # fetch the best node
                nextnodes = []
                time_rnninp = []
                time_hidden0 = []
                time_hidden1 = []
                prev_scores = []
                time_windows = []

                for elem in range(min(len(prev_nodes))):
                    score_top, n_top = prev_nodes[elem]

                    time_rnninp.append(n_top['rnninp'])
                    time_hidden0.append(n_top['h'][0])
                    time_hidden1.append(n_top['h'][1])
                    prev_scores.append(n_top['logp'])
                    time_windows.append(n_top['window'])
                    print('aaa', n_top['rnninp'].shape)

                # batch the computation.
                time_rnninp = torch.cat(time_rnninp, dim=0)
                time_hidden0 = torch.cat(time_hidden0, dim=1)

                print(time_rnninp.shape, time_hidden0.shape)
                time_hidden1 = torch.cat(time_hidden1, dim=1)
                prev_scores = torch.tensor(prev_scores).view(-1, 1, 1).repeat_interleave(beam_w_, dim=0)
                print('prev scores = ', prev_scores)

                time_e = time.time()
                print(time_hidden0.shape)
                print(time_rnninp.shape)
                print(self.rnn)
                h, decoder_hidden = self.rnn(time_rnninp, (time_hidden0, time_hidden1))
                print(srcfieldenc[idx:idx + 1].shape, h.shape, srcfieldenc_s.shape)
                out_rnn, copy = self.attn_step_dual(h, -1, srcfieldenc[idx:idx + 1], fieldmask[idx:idx + 1],
                                                    srcfieldenc_s, srcfieldenc_dual_s, mod='state')
                state_logp = torch.log_softmax(self.state_pick(out_rnn), dim=-1)
                log_prob, indexes = torch.topk(state_logp, beam_w_, )
                time_f = time.time()

                timing['rnnstate_comp'] += time_f - time_e

                print(')' * 30)

                out_rnn, copy = self.attn_step_dual(torch.repeat_interleave(h, beam_w_, dim=0), indexes,
                                                    srcfieldenc[idx:idx + 1],
                                                    fieldmask[idx:idx + 1], srcfieldenc_w, srcfieldenc_dual_w)

                print(out_rnn.shape)

                temp_wordlp = torch.log_softmax(torch.cat([self.vocab_pick(out_rnn), copy], dim=-1), dim=-1)

                print(temp_wordlp.shape)

                log_prob_w, indexes_w = torch.topk(temp_wordlp, beam_width + window_size)
                print('lisa is here ')
                print(log_prob_w)
                print(indexes_w)
                print(indexes_w.shape, log_prob_w.shape)
                print(log_prob.shape, indexes.shape)

                print(prev_scores.shape, log_prob_w.shape, log_prob.shape)
                ''' first filter the set of wanted tokens . finally make each node object '''
                print(prev_scores.shape)
                print(log_prob.shape)
                compare_scores = -(prev_scores.expand_as(log_prob_w) + log_prob.view(-1, 1, 1).expand_as(log_prob_w) + \
                                   log_prob_w)

                print(compare_scores)

                sorted_scores, ordering = torch.sort(compare_scores.view(-1))
                # create compartment.
                comp_w = indexes_w.view(-1)
                comp_s = indexes.view(-1, 1, 1).expand_as(indexes_w).contiguous().view(-1)
                parent_order = np.repeat(np.arange(len(prev_nodes)), (beam_width + window_size) * beam_w_)
                print(parent_order)
                # comp_s = indexes.expand(-1, -1, beam_width + window_size).view(-1)
                # comp_n =

                comp_i = 0
                nextnodes = []
                while len(nextnodes) < beam_width * beam_w_:
                    score = sorted_scores[comp_i]
                    order = ordering[comp_i]
                    related_w = comp_w[comp_i]
                    related_s = comp_s[comp_i]
                    parent_ = parent_order[comp_i]
                    parent_node = prev_nodes[parent_][1]
                    decoder_hidden_temp = (decoder_hidden[0][parent_], decoder_hidden[1][parent_])
                    print(comp_i)
                    print(parent_)

                    print(score, order, related_w, related_s)

                    temp_window = prev_nodes[parent_][1]['window']
                    if related_w.item() in [x[0] for x in temp_window[-window_size:]]:
                        comp_i += 1
                        continue
                    else:
                        window_new = temp_window + [(related_w.item(), related_s.item())]
                        comp_i += 1

                        if related_w > self.gen_vocab_size:
                            # generate out of gen_vocab: copy
                            temp_word_id = related_w - self.gen_vocab_size - 1
                            decoded_feat = src_wrd2fields[idx][temp_word_id]  # bsz x seqlen x 4
                            # decoded_feat = torch.LongTensor(src_wrd2fields[idx][temp_word_id]) # bsz x seqlen x 4
                            decoded_tw = decoded_feat[0]
                            record = temp_word_id.item()
                        else:
                            # generate from the gen vocab.
                            # temp = self.temp_field
                            # temp[0] = decoded_tw
                            # decoded_feat = temp
                            record = -1
                            decoded_tw = related_w
                            decoded_feat = -1

                        # rnninp
                        temp_word_emb = self.word_vecs(related_w.to(self.device)).view(1, 1, -1)
                        temp_state_emb = self.state_vecs(related_s).view(1, 1, -1)
                        print(temp_word_emb.shape)
                        print(temp_state_emb.shape)
                        rnninp = torch.cat([temp_word_emb, temp_state_emb], dim=-1)
                        node = {'h': decoder_hidden, 'prevNode': parent_node, 'wordid': decoded_tw,
                                'logp': -score, 'leng': parent_node['leng'] + 1,
                                'state_id': related_s.item(), 'word_feats': -1, 'rnninp': rnninp,
                                'window': window_new, 'record': record}

                        if (node['wordid'] == self.eos_idx and node['prevNode'] != None) or node['leng'] >= max_len:
                            endnodes.append((score, node))
                            # if we reached maximum # of sentences required
                            if len(endnodes) >= number_required:
                                break
                            else:
                                continue
                        else:
                            nextnodes.append((score, node))

                if break_flag:
                    break

                # next_nodes = []
                # for i in range(len(nextnodes)):
                #     score, nn_ = nextnodes[i]
                #     next_nodes.append((score, nn_))
                # qsize += 1
                # sort next_nodes.
                next_nodes = sorted(nextnodes, key=lambda x: x[0])

                prev_nodes = next_nodes
                print(prev_nodes)

            # back tracing
            time_ee = time.time()
            utterances_w = []
            utterances_s = []
            scores_result = []
            copys = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance_w = []
                utterance_s = []
                copy = []
                utterance_w.append(n['wordid'])
                utterance_s.append(n['state_id'])
                copy.append(n['record'])
                # back trace
                while n['prevNode'] != None:
                    n = n['prevNode']
                    utterance_w.append(n['wordid'])
                    utterance_s.append(n['state_id'])
                    copy.append(n['record'])

                utterance_w = utterance_w[::-1]
                utterance_s = utterance_s[::-1]
                copy = copy[::-1]
                utterances_w.append(utterance_w)
                utterances_s.append(utterance_s)
                scores_result.append(score)
                copys.append(copy)

            decoded_batch.append(utterances_w)
            decoded_states.append(utterances_s)
            decoded_score.append(scores_result)
            decoded_copy.append(copys)
            time_ff = time.time()
            timing['back_trace'] += time_ff - time_ee

        time_gg = time.time()
        temp_lst = idx2word(decoded_batch, self.idx2word)

        final_states = []
        len_marks = []
        for x in decoded_states:
            final_states += x
            len_marks += [y for y in range(len(x))]
        final_scores = []
        for x in decoded_score:
            final_scores += x
        final_copy = []
        for x in decoded_copy:
            final_copy += x

        result_str = ''
        for idx_, elem in enumerate(temp_lst):
            result_str += ' '.join(elem)
            result_str += '|||{}'.format(final_scores[idx_])
            result_str += '|||{}'.format(final_states[idx_])
            result_str += '|||{}'.format(len_marks[idx_])
            result_str += '|||{}'.format(final_copy[idx_])
            result_str += '\n'
        time_ii = time.time()

        timing['finalstep'] += time_ii - time_gg
        return result_str

    def attn_step_dual(self, h, k, srcfieldenc, fieldmask, srcfieldenc_, srcfieldenc_dual, mod='vocab'):
        '''
            h is the output of the RNN: bsz * seglen * hiddem_dim
            know the current state name is k
            Thus, we need to pay attention to the srcfieldenc

            return: states_k: the output of the attention layer: size (bsz*sample_size, seglen, w_dim+h_dim)
                    ascrores: the copy attention score (bsz*sample_size, seglen,  src_field_len)
        '''
        if mod == 'vocab':
            bsz, seglen, cond_dim = h.shape
            mid_dim = self.mid_dim
            state_att_gates = self.state_att_gates[k]
            state_att_biases = self.state_att_biases[k]  # (bsz*sample, seglen, h_dim)

            attnin1 = torch.bmm(h.contiguous().view(1, 1, cond_dim),
                                state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
            attnin1 = F.tanh(attnin1).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
            srcfieldenc_ = srcfieldenc_.transpose(1, 2)  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
            ascores = torch.bmm(attnin1, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)
            fieldmask = fieldmask.unsqueeze(1).expand_as(ascores)
            ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
            aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

            ######################################################################
            state_att_gates_dual = self.state_att_gates_dual[k]
            state_att_biases_dual = self.state_att_biases_dual[k]

            attnin1_dual = torch.bmm(h.contiguous().view(1, 1, cond_dim), state_att_gates_dual) + state_att_biases_dual
            attnin1_dual = F.tanh(attnin1_dual).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
            srcfieldenc_dual = srcfieldenc_dual.transpose(1, 2)
            ascores_dual = torch.bmm(attnin1_dual, srcfieldenc_dual)  # (bsz*sample_size, seglen,  src_field_len)
            ascores_dual[fieldmask] = neginf
            aprobs_dual = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

            # combine alphas and betas from dual.
            final_probs = aprobs_dual * aprobs
            final_probs = final_probs / (final_probs.sum(2).unsqueeze(2).expand(final_probs.shape) + 1e-6)

            ctx = torch.bmm(final_probs, srcfieldenc)  # (bsz*sample_size, seglen, w_dim)
            cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)

            state_out_gates = self.state_out_gates[k]
            state_out_biases = self.state_out_biases[k]
            states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)

            if self.sep_attn:
                state_att2_gates = self.state_att2_gates[k]  # (bsz*sample, seglen, h_dim)
                state_att2_biases = self.state_att2_biases[k]  # (bsz*sample, seglen, h_dim)
                attnin2 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                    state_att2_gates) + state_att2_biases  # (bsz*sample, seglen, h_dim)
                attnin2 = F.tanh(attnin2).view(bsz, seglen, mid_dim)
                ascores = torch.bmm(attnin2, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)
                ascores[fieldmask] = neginf

                ######################################
                state_att_gates_dual = self.state_att2_gates_dual[k]
                state_att_biases_dual = self.state_att2_biases_dual[k]
                attnin2_dual = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                         state_att_gates_dual) + state_att_biases_dual  # (bsz*sample, seglen, h_dim)
                attnin2_dual = F.tanh(attnin2_dual).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
                ascores_dual = torch.bmm(attnin2_dual, srcfieldenc_dual)  # (bsz*sample_size, seglen,  src_field_len)
                ascores = ascores + ascores_dual
                ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
            return states_k, ascores

        else:
            bsz, seglen, cond_dim = h.shape
            mid_dim = self.mid_dim
            state_att_gates = self.state_att_gates2
            state_att_biases = self.state_att_biases2  # (bsz*sample, seglen, h_dim)

            attnin1 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
            attnin1 = F.tanh(attnin1).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)

            # srcfieldenc_ = srcfieldenc_  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
            ascores = torch.bmm(attnin1, srcfieldenc_.transpose(1, 2))  # (bsz*sample_size, seglen,  src_field_len)

            fieldmask = fieldmask.unsqueeze(1).expand_as(ascores)
            ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
            aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

            ###################################################3
            state_att_gates_dual = self.state_att_gates2_dual
            state_att_biases_dual = self.state_att_biases2_dual
            attnin1_dual = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                     state_att_gates_dual) + state_att_biases_dual  # (bsz*sample, seglen, h_dim)
            attnin1_dual = F.tanh(attnin1_dual).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
            # srcfieldenc_dual = self.attn_src2_dual(field_vecs)
            ascores_dual = torch.bmm(attnin1_dual, srcfieldenc_dual.transpose(1, 2))
            ascores_dual[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
            aprobs_dual = F.softmax(ascores_dual, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

            # combine the alphas and betas.
            probs_total = aprobs * aprobs_dual
            probs_total = probs_total / (probs_total.sum(2).unsqueeze(2).expand(probs_total.shape) + 1e-6)
            ##############################################
            ctx = torch.bmm(probs_total, srcfieldenc)  # (bsz*sample_size, seglen, w_dim)
            cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)
            state_out_gates = self.state_out_gates2
            state_out_biases = self.state_out_biases2
            states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)  # (bsz*sample_size, seglen, w_dim+h_dim)

            return states_k, None

    def attn_step_dual4(self, h, k, srcfieldenc, fieldmask, srcfieldenc_, srcfieldenc_dual, mod='vocab'):
        '''
            h is the output of the RNN: bsz * seglen * hiddem_dim
            know the current state name is k
            Thus, we need to pay attention to the srcfieldenc

            return: states_k: the output of the attention layer: size (bsz*sample_size, seglen, w_dim+h_dim)
                    ascrores: the copy attention score (bsz*sample_size, seglen,  src_field_len)
        '''
        if mod == 'vocab':
            bsz, seglen, cond_dim = h.shape
            mid_dim = self.mid_dim

            state_att_gates = torch.index_select(self.state_att_gates, 0, k.view(-1)).view(bsz * seglen, cond_dim,
                                                                                           mid_dim)
            state_att_biases = torch.index_select(self.state_att_biases, 0, k.view(-1)).view(bsz * seglen, 1, mid_dim)

            print(state_att_biases.shape)
            print(state_att_gates.shape)
            print(k)

            attnin1 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
            attnin1 = F.tanh(attnin1).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
            srcfieldenc_ = srcfieldenc_.transpose(1, 2).expand(bsz, -1,
                                                               -1)  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
            ascores = torch.bmm(attnin1, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)
            fieldmask = fieldmask.unsqueeze(1).expand_as(ascores)
            ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
            aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

            ######################################################################
            state_att_gates_dual = torch.index_select(self.state_att_gates_dual, 0, k.view(-1)).view(bsz * seglen,
                                                                                                     cond_dim,
                                                                                                     mid_dim)
            state_att_biases_dual = torch.index_select(self.state_att_biases_dual, 0, k.view(-1)).view(bsz * seglen, 1,
                                                                                                       mid_dim)

            # state_att_gates_dual = self.state_att_gates_dual[k]
            # state_att_biases_dual = self.state_att_biases_dual[k]

            attnin1_dual = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                     state_att_gates_dual) + state_att_biases_dual
            attnin1_dual = F.tanh(attnin1_dual).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
            srcfieldenc_dual = srcfieldenc_dual.transpose(1, 2).expand(bsz, -1, -1)
            ascores_dual = torch.bmm(attnin1_dual, srcfieldenc_dual)  # (bsz*sample_size, seglen,  src_field_len)
            ascores_dual[fieldmask] = neginf
            aprobs_dual = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

            # combine alphas and betas from dual.
            final_probs = aprobs_dual * aprobs
            final_probs = final_probs / (final_probs.sum(2).unsqueeze(2).expand(final_probs.shape) + 1e-6)

            ctx = torch.bmm(final_probs, srcfieldenc.expand(bsz, -1, -1))  # (bsz*sample_size, seglen, w_dim)
            cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)

            state_out_gates = torch.index_select(self.state_out_gates, 0, k.view(-1)).view(bsz, seglen, -1)
            state_out_biases = torch.index_select(self.state_out_biases, 0, k.view(-1)).view(bsz, seglen, -1)

            states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)

            if self.sep_attn:
                state_att2_gates = torch.index_select(self.state_att2_gates, 0, k.view(-1)).view(bsz * seglen, cond_dim,
                                                                                                 mid_dim)
                state_att2_biases = torch.index_select(self.state_att2_biases, 0, k.view(-1)).view(bsz * seglen, 1,
                                                                                                   mid_dim)

                attnin2 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                    state_att2_gates) + state_att2_biases  # (bsz*sample, seglen, h_dim)
                attnin2 = F.tanh(attnin2).view(bsz, seglen, mid_dim)
                ascores = torch.bmm(attnin2, srcfieldenc_)  # (bsz*sample_size, seglen,  src_field_len)
                ascores[fieldmask] = neginf

                ######################################
                state_att2_gates_dual = torch.index_select(self.state_att2_gates_dual, 0, k.view(-1)).view(bsz * seglen,
                                                                                                           cond_dim,
                                                                                                           mid_dim)
                state_att2_biases_dual = torch.index_select(self.state_att2_biases_dual, 0, k.view(-1)).view(
                    bsz * seglen,
                    1, mid_dim)

                attnin2_dual = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                         state_att_gates_dual) + state_att_biases_dual  # (bsz*sample, seglen, h_dim)
                attnin2_dual = F.tanh(attnin2_dual).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
                ascores_dual = torch.bmm(attnin2_dual, srcfieldenc_dual)  # (bsz*sample_size, seglen,  src_field_len)
                ascores = ascores + ascores_dual
                ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
            return states_k, ascores

        else:
            bsz, seglen, cond_dim = h.shape
            print(bsz)
            mid_dim = self.mid_dim
            state_att_gates = self.state_att_gates2.expand(bsz * seglen, cond_dim, mid_dim)
            state_att_biases = self.state_att_biases2.expand(bsz * seglen, 1, mid_dim)  # (bsz*sample, seglen, h_dim)

            attnin1 = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                state_att_gates) + state_att_biases  # (bsz*sample, seglen, h_dim)
            attnin1 = F.tanh(attnin1).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)

            # srcfieldenc_ = srcfieldenc_  # (bsz, src_field_len, w_dim) -> (bsz, src_field_len, m_dim)
            ascores = torch.bmm(attnin1, srcfieldenc_.transpose(1, 2).expand(bsz, -1,
                                                                             -1))  # (bsz*sample_size, seglen,  src_field_len)

            fieldmask = fieldmask.unsqueeze(1).expand_as(ascores)
            ascores[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
            aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

            ###################################################3
            state_att_gates_dual = self.state_att_gates2_dual.expand(bsz * seglen, cond_dim, mid_dim)
            state_att_biases_dual = self.state_att_biases2_dual.expand(bsz * seglen, 1, mid_dim)
            attnin1_dual = torch.bmm(h.contiguous().view(bsz * seglen, 1, cond_dim),
                                     state_att_gates_dual) + state_att_biases_dual  # (bsz*sample, seglen, h_dim)
            attnin1_dual = F.tanh(attnin1_dual).view(bsz, seglen, mid_dim)  # (bsz*sample, seglen, h_dim)
            # srcfieldenc_dual = self.attn_src2_dual(field_vecs)
            ascores_dual = torch.bmm(attnin1_dual, srcfieldenc_dual.transpose(1, 2).expand(bsz, -1, -1))
            ascores_dual[fieldmask] = neginf  # (bsz*sample_size, seglen,  src_field_len)
            aprobs_dual = F.softmax(ascores_dual, dim=2)  # (bsz*sample_size, seglen,  src_field_len)

            # combine the alphas and betas.
            probs_total = aprobs * aprobs_dual
            probs_total = probs_total / (probs_total.sum(2).unsqueeze(2).expand(probs_total.shape) + 1e-6)
            ##############################################
            ctx = torch.bmm(probs_total, srcfieldenc.expand(bsz, -1, -1))  # (bsz*sample_size, seglen, w_dim)
            cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)
            state_out_gates = self.state_out_gates2
            state_out_biases = self.state_out_biases2
            states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)  # (bsz*sample_size, seglen, w_dim+h_dim)

            return states_k, None

    def beam_forward_easy_dual2(self, condi, bsz, timing=None):
        beam_width = 5
        beam_w_ = 1
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        decoded_score = []
        decoded_states = []
        decoded_copy = []
        max_len = 60
        window_size = 3

        time_a = time.time()

        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        field_vecs = condi['field_vecs']
        fieldmask = condi["fmask"]
        src_wrd2fields = condi["src"]

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        layers, rnn_size = self.layers, self.h_dim
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        # print(inits.shape, self.hidden_dim, rnn_size)
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim
        c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()  # layer * bsz * dim

        time_b = time.time()
        timing['pre_comp'] += time_b - time_a

        # the rnns have batch first.
        for idx in range(bsz):
            decoder_hidden = (h_prev[:, idx, :].unsqueeze(1).contiguous(), c_prev[:, idx, :].unsqueeze(1).contiguous())
            # Start with the start of the sentence token

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            time_c = time.time()

            srcfieldenc_w = F.tanh(self.attn_src(srcfieldenc[idx:idx + 1]))
            srcfieldenc_s = F.tanh(self.attn_src2(srcfieldenc[idx:idx + 1]))

            srcfieldenc_dual_w = F.tanh(self.attn_src_dual(field_vecs[idx:idx + 1]))
            srcfieldenc_dual_s = F.tanh(self.attn_src2_dual(field_vecs[idx:idx + 1]))

            time_d = time.time()
            timing['pre_comp'] += time_d - time_c

            # starting node -  hidden vector, previous node, word id, logp, length, state_id, word_feats, rnninp
            node = BeamSearchNode(decoder_hidden, None, self.bos_idx, 0, 1, -1, -1, rnninp=self.bos_emb, window=[])
            node.record = -1
            node.prev_cell = None
            node.stack = None
            prev_nodes = []
            prev_nodes.append((-node.eval(), node))
            # qsize = 1
            # start beam search
            break_flag = False
            while True:
                # give up when we have enough good results.
                if len(endnodes) >= number_required: break

                # fetch the best node
                nextnodes = []
                for elem in range(min(len(prev_nodes), beam_width)):
                    score_top, n_top = prev_nodes[elem]

                    if n_top.wordid == self.eos_idx and n_top.prevNode != None:
                        endnodes.append((score, n_top))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    time_e = time.time()

                    h, decoder_hidden = self.rnn(n_top.rnninp.contiguous(), n_top.h)
                    out_rnn, copy = self.attn_step_dual(h, -1, srcfieldenc[idx:idx + 1], fieldmask[idx:idx + 1],
                                                        srcfieldenc_s, srcfieldenc_dual_s, mod='state')
                    state_logp = nn.LogSoftmax(dim=-1)(self.state_pick(out_rnn))
                    log_prob, indexes = torch.topk(state_logp, beam_w_)

                    time_f = time.time()

                    timing['rnnstate_comp'] += time_f - time_e

                    for new_k in range(beam_w_):

                        time_aa = time.time()
                        decoded_ts = indexes[0][0][new_k].view(-1)
                        log_ps = log_prob[0][0][new_k].item()

                        out_rnn, copy = self.attn_step_dual(h, decoded_ts.item(), srcfieldenc[idx:idx + 1],
                                                            fieldmask[idx:idx + 1], srcfieldenc_w, srcfieldenc_dual_w)
                        temp_wordlp = nn.LogSoftmax(dim=-1)(torch.cat([self.vocab_pick(out_rnn), copy], dim=-1))
                        log_prob_w, indexes_w = torch.topk(temp_wordlp, beam_width + window_size)

                        time_bb = time.time()
                        timing['word_comp'] += time_bb - time_aa

                        temp_ww = []
                        temp_pp = []

                        time_cc = time.time()
                        ''' this section should be moved to CPU. '''
                        for elem1, elem2 in zip(indexes_w.view(-1), log_prob_w.view(-1)):

                            if elem1.item() in [x[0] for x in n_top.window[-window_size:]]:
                                pass
                            else:
                                temp_ww.append(elem1)
                                temp_pp.append(elem2)
                            if len(temp_ww) >= beam_width:
                                break

                        for new_k_w in range(beam_width):

                            decoded_tw = temp_ww[new_k_w].view(-1)[0]
                            log_pw = temp_pp[new_k_w].item()
                            window_new = n_top.window + [(decoded_tw.item(), decoded_ts.item())]

                            if decoded_tw > self.gen_vocab_size:
                                # generate out of gen_vocab: copy
                                temp_word_id = decoded_tw - self.gen_vocab_size - 1
                                decoded_feat = src_wrd2fields[idx][temp_word_id]  # bsz x seqlen x 4
                                # decoded_feat = torch.LongTensor(src_wrd2fields[idx][temp_word_id]) # bsz x seqlen x 4
                                decoded_tw = decoded_feat[0]
                                record = temp_word_id.item()
                            else:
                                # generate from the gen vocab.
                                temp = self.temp_field
                                temp[0] = decoded_tw
                                decoded_feat = temp
                                record = -1

                            # rnninp
                            temp_word_emb = self.get_word_embs(decoded_feat).view(1, 1, -1)
                            temp_state_emb = self.state_vecs(decoded_ts).view(1, 1, -1)
                            rnninp = torch.cat([temp_word_emb, temp_state_emb], dim=-1)
                            node = BeamSearchNode(decoder_hidden, n_top, decoded_tw, n_top.logp + log_ps + log_pw,
                                                  n_top.leng + 1, state_id=decoded_ts.item(), rnninp=rnninp,
                                                  window=window_new)
                            node.record = record
                            score = -node.eval()

                            if node.leng >= max_len:
                                endnodes.append((score, node))
                                if len(endnodes) >= number_required:
                                    break_flag = True
                                    break
                                else:
                                    continue
                            else:
                                nextnodes.append((score, node))

                        time_dd = time.time()
                        timing['word_grp'] += time_dd - time_cc
                        if break_flag:
                            break

                    if break_flag:
                        break

                next_nodes = []
                for i in range(len(nextnodes)):
                    score, nn_ = nextnodes[i]
                    next_nodes.append((score, nn_))
                    # qsize += 1
                # sort next_nodes.
                next_nodes = sorted(next_nodes, key=lambda x: x[0])
                prev_nodes = next_nodes

            # back tracing
            time_ee = time.time()
            utterances_w = []
            utterances_s = []
            scores_result = []
            copys = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance_w = []
                utterance_s = []
                copy = []
                utterance_w.append(n.wordid)
                utterance_s.append(n.state_id)
                copy.append(n.record)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance_w.append(n.wordid)
                    utterance_s.append(n.state_id)
                    copy.append(n.record)

                utterance_w = utterance_w[::-1]
                utterance_s = utterance_s[::-1]
                copy = copy[::-1]
                utterances_w.append(utterance_w)
                utterances_s.append(utterance_s)
                scores_result.append(score)
                copys.append(copy)

            decoded_batch.append(utterances_w)
            decoded_states.append(utterances_s)
            decoded_score.append(scores_result)
            decoded_copy.append(copys)
            time_ff = time.time()
            timing['back_trace'] += time_ff - time_ee

        time_gg = time.time()
        temp_lst = idx2word(decoded_batch, self.idx2word)

        final_states = []
        len_marks = []
        for x in decoded_states:
            final_states += x
            len_marks += [y for y in range(len(x))]
        final_scores = []
        for x in decoded_score:
            final_scores += x
        final_copy = []
        for x in decoded_copy:
            final_copy += x

        result_str = ''
        for idx_, elem in enumerate(temp_lst):
            result_str += ' '.join(elem)
            result_str += '|||{}'.format(final_scores[idx_])
            result_str += '|||{}'.format(final_states[idx_])
            result_str += '|||{}'.format(len_marks[idx_])
            result_str += '|||{}'.format(final_copy[idx_])
            result_str += '\n'
        time_ii = time.time()

        timing['finalstep'] += time_ii - time_gg
        return result_str