import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import cuda
from utils import *
neginf = -1e38

class HSMM(nn.Module):
    def __init__(self, args, START_TAG=1, STOP_TAG=2):
        '''
        initialization step.
        '''
        # embedding_dim, hidden_dim, vocab_size, tag_set_size,  q_dim=20, dropout=0,
        super(HSMM, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.vocab_size = args.vocab_size
        self.tagset_size = args.tagset_size
        self.START_TAG = START_TAG
        self.STOP_TAG = STOP_TAG
        self.pad1 = args.vocab['<s>']  # idx for <s> token from ptb.dict
        self.pad2 = args.vocab['<\s>']  # idx for </s> token from ptb.dict
        self.dropout = nn.Dropout(args.dropout)
        self.w_dim = args.embedding_dim
        self.conditional_dim = args.conditional_dim
        self.q_dim = 20
        self.smaller_cond_dim = 20
        self.cond_A_dim = 20
        self.Kmul = 1
        self.A_dim = 20
        self.L = args.L

        # real dataset encoding
        self.max_pool = args.max_pool
        self.src_bias = nn.Parameter(torch.Tensor(1, self.embedding_dim))
        self.uniq_bias = nn.Parameter(torch.Tensor(1, self.embedding_dim))


        self.init_lin = nn.Linear(self.conditional_dim, self.tagset_size * self.Kmul)
        self.lsm = nn.LogSoftmax(dim=1)
        self.A_from = nn.Parameter(torch.randn(self.tagset_size * self.Kmul, self.A_dim))
        self.A_to = nn.Parameter(torch.randn(self.A_dim, self.tagset_size * self.Kmul))
        self.q_binary = nn.Sequential(nn.Linear(self.q_dim * 2, self.q_dim), nn.Tanh(),
                                     nn.Linear(self.q_dim, self.tagset_size))
        self.cond_trans_lin = nn.Sequential(
            nn.Linear(self.conditional_dim, self.smaller_cond_dim),
            nn.ReLU(),
            nn.Linear(self.smaller_cond_dim, self.tagset_size * self.Kmul * self.cond_A_dim * 2))

        self.unif_lenps = True
        if self.unif_lenps:
            self.len_scores = nn.Parameter(torch.ones(1, self.L))
            self.len_scores.requires_grad = False
            # self.len_scores[:, 0] = -2

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.w_dim, self.q_dim,
                            num_layers=1, bidirectional=True)

        self.yes_self_trans = False
        if not self.yes_self_trans:
            selfmask = torch.Tensor(self.Kmul * self.tagset_size).fill_(neginf)
            self.register_buffer('selfmask', Variable(torch.diag(selfmask), requires_grad=False))


    def get_span_scores(self, x):
        # produces the span scores s_ij
        # mask = torch.ones(x.size(1), x.size(1)).tril() * -1e20
        bos = x.new(x.size(0), 1).fill_(self.pad1)
        eos = x.new(x.size(0), 1).fill_(self.pad2)
        # print(bos.shape)
        # print(eos.shape)
        x = torch.cat([bos, x, eos], 1)
        x_vec = self.word_embeds(x) # add dropout maybe.
        # pos = torch.arange(0, x.size(1)).unsqueeze(0).expand_as(x).long().cuda()
        # pos = torch.arange(0, x.size(1)).unsqueeze(0).expand_as(x).long()
        # x_vec = x_vec + self.dropout(self.q_pos_emb(pos))
        q_h, _ = self.lstm(x_vec)
        fwd = q_h[:, 1:, :self.q_dim]
        bwd = q_h[:, :-1, self.q_dim:]
        fwd_diff = fwd[:, 1:].unsqueeze(1) - fwd[:, :-1].unsqueeze(2)
        bwd_diff = bwd[:, :-1].unsqueeze(2) - bwd[:, 1:].unsqueeze(1)
        # as a result fwd_diff[i,j] = the difference ends at j and begins at i (inclusive).
        concat = torch.cat([fwd_diff, bwd_diff], 3) # bsz, T, T, tagset_dim

        scores = self.q_binary(concat).squeeze(3)
        batch_size, T, _, tag_dim = scores.shape
        # scores * mask.unsqueeze(0).unsqueeze(3).expand(batch_size, T, T, tag_dim)
        return scores

    def bilstm_minus(self, x):
        # produces the span scores s_ij
        bos = x.new(x.size(0), 1).fill_(self.pad1)
        eos = x.new(x.size(0), 1).fill_(self.pad2)

        x = torch.cat([bos, x, eos], 1)
        x_vec = self.word_embeds(x)  # add dropout maybe.

        q_h, _ = self.lstm(x_vec)
        fwd = q_h[:, 1:, :self.q_dim]
        bwd = q_h[:, :-1, self.q_dim:]
        fwd_diff = fwd[:, 1:].unsqueeze(1) - fwd[:, :-1].unsqueeze(2)
        bwd_diff = bwd[:, :-1].unsqueeze(2) - bwd[:, 1:].unsqueeze(1)
        # as a result fwd_diff[i,j] = the difference ends at j and begins at i (inclusive).
        concat = torch.cat([fwd_diff, bwd_diff], 3)  # bsz, T, T, tagset_dim

        return concat


    def get_span_context_score(self, x, srcenc, srcfieldenc, mask, combotarg, bsz):
        seqlen, bsz, maxlocs, nfeats = x.size()
        inits = self.h0_lin(srcenc) # bsz x 2*dim
        bsz, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.hid_size
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:] # (bsz x dim, bsz x dim)
        h0 = F.tanh(h0).unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
            -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
        c0 = c0.unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
            -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()

        spans = self.get_span_scores(x)

        attnin1 = (states * self.state_att_gates[k].expand_as(states)
                   + self.state_att_biases[k].expand_as(states)).view(
            Lp1, bsz, seqlen, -1)

        # L+1 x bsz x seqlen x rnn_size -> bsz x (L+1)seqlen x rnn_size
        attnin1 = attnin1.transpose(0, 1).contiguous().view(bsz, Lp1 * seqlen, -1)
        attnin1 = F.tanh(attnin1)
        ascores = torch.bmm(attnin1, srcfieldenc.transpose(1, 2))  # bsz x (L+1)slen x nfield
        ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)
        aprobs = F.softmax(ascores, dim=2)
        # bsz x (L+1)seqlen x nfields * bsz x nfields x dim -> bsz x (L+1)seqlen x dim
        ctx = torch.bmm(aprobs, srcfieldenc)
        # concatenate states and ctx to get L+1 x bsz x seqlen x rnn_size + encdim
        cat_ctx = torch.cat([states.view(Lp1, bsz, seqlen, -1),
                             ctx.view(bsz, Lp1, seqlen, -1).transpose(0, 1)], 3)
        out_hid_sz = rnn_size + encdim
        cat_ctx = cat_ctx.view(Lp1, -1, out_hid_sz)
        # now linear to get L+1 x bsz*seqlen x rnn_size
        states_k = F.tanh(cat_ctx * self.state_out_gates[k].expand_as(cat_ctx)
                          + self.state_out_biases[k].expand_as(cat_ctx)).view(
            Lp1, -1, out_hid_sz)

        if self.sep_attn:
            attnin2 = (states * self.state_att2_gates[k].expand_as(states)
                       + self.state_att2_biases[k].expand_as(states)).view(
                Lp1, bsz, seqlen, -1)
            # L+1 x bsz x seqlen x rnn_size -> bsz x (L+1)seqlen x emb_size
            attnin2 = attnin2.transpose(0, 1).contiguous().view(bsz, Lp1 * seqlen, -1)
            attnin2 = F.tanh(attnin2)
            ascores = torch.bmm(attnin2, srcfieldenc.transpose(1, 2))  # bsz x (L+1)slen x nfield
            ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)

        normfn = F.log_softmax if self.lse_obj else F.softmax
        wlps_k = normfn(torch.cat([self.decoder(states_k.view(-1, out_hid_sz)),  # L+1*bsz*sl x V
                                   ascores.view(bsz, Lp1, seqlen, nfields).transpose(
                                       0, 1).contiguous().view(-1, nfields)], 1), dim=1)
        # concatenate on dummy column for when only a single answer...
        wlps_k = torch.cat([wlps_k, Variable(self.zeros.expand(wlps_k.size(0), 1))], 1)
        # get scores for predicted next-words (but not for last words in each segment as usual)
        psk = wlps_k.narrow(0, 0, self.L * bszsl).gather(1, combotargs.view(self.L * bszsl, -1))
        if self.lse_obj:
            lls_k = logsumexp1(psk)
        else:
            lls_k = psk.sum(1).log()

        # sum up log probs of words in each segment
        seglls_k = lls_k.view(self.L, -1).cumsum(0)  # L x bsz*seqlen
        # need to add end-of-phrase prob too
        eop_lps = wlps_k.narrow(0, bszsl, self.L * bszsl)[:, self.eop_idx]  # L*bsz*seqlen
        if self.lse_obj:
            seglls_k = seglls_k + eop_lps.contiguous().view(self.L, -1)
        else:
            seglls_k = seglls_k + eop_lps.log().view(self.L, -1)
        seg_lls.append(seglls_k)

        #  K x L x bsz x seqlen -> seqlen x L x bsz x K -> L x seqlen x bsz x K

        obslps = torch.stack(seg_lls).view(self.K, self.L, bsz, -1).transpose(
            0, 3).transpose(0, 1)
        if self.Kmul > 1:
            obslps = obslps.repeat(1, 1, 1, self.Kmul)

        pass

    def get_emission_score(self, score_temp):
        '''
        get phi(y_{t+1:t+l}, z_{t+1}).
        Since we have a CRF, we can score each span by LSTM-minus.
        :param x: the input database (in Long Tensor)
        :param y: the output sentence (in Long Tensor)
        :return: L x T x bsz x K: length = l, end=t, batch_size = bsz, and state size = k
        '''
        L = self.L
        bsz, T, _, K = score_temp.size()
        result = torch.ones(L, T, bsz, K) * -1e30
        score_temp = score_temp.permute(1, 2, 0, 3)
        for start in range(T):
            result[:min(L, T - start), start, :, :] = score_temp[start, start:min(start+L, T), :, :]
        return result

    def get_emission_score_for_alpha(self, score_temp):
        # TEMP
        '''
        get phi(y_{t+1:t+l}, z_{t+1}).
        Since we have a CRF, we can score each span by LSTM-minus.
        :param x: the input database (in Long Tensor)
        :param y: the output sentence (in Long Tensor)
        :return: L x T x bsz x K: length = l, end=t, batch_size = bsz, and state size = k
        '''
        L = self.L
        bsz, T, _, K = score_temp.size()
        result = torch.ones(L, T, bsz, K) * -1e30
        score_temp = score_temp.permute(1, 2, 0, 3)
        # print(score_temp.shape)
        for end in range(0, T):
            result[-min(L+1, end+1):, end, :, :] = score_temp[max(end-L+1, 0):end+1, end, :, :]
        return result

    def  get_length_score(self):
        """
        returns:
           [1xK tensor, 2 x K tensor, .., L-1 x K tensor, L x K tensor] of logprobs
        """
        K = self.tagset_size * self.Kmul

        state_embs = torch.cat([self.A_from, self.A_to.t()], 1)  # K x 2*A_dim
        if self.unif_lenps:
            len_scores = self.len_scores.expand(K, self.L)
        else:
            len_scores = self.len_decoder(state_embs)  # K x L
        lplist = [len_scores.data.new(1, K).zero_()]
        # lplist = [Variable(len_scores.data.new(1, K).zero_())]
        # print(lplist[-1])
        for l in range(2, self.L + 1):
            lplist.append(nn.LogSoftmax(dim=1)(len_scores.narrow(1, 0, l)).t())
        return lplist, len_scores

    def get_transition_score(self, uniqenc, seqlen=10):
        """
        args:
          uniqenc - bsz x emb_size
        returns:
          1 x K tensor and seqlen-1 x bsz x K x K tensor of log probabilities,
                           where lps[i] is p(q_{i+1} | q_i)
        """
        uniqenc = uniqenc.squeeze(1)
        bsz = uniqenc.size(0)
        K = self.tagset_size * self.Kmul
        # print(uniqenc.shape)
        # bsz x K*A_dim*2 -> bsz x K x A_dim or bsz x K x 2*A_dim
        cond_trans_mat = self.cond_trans_lin(uniqenc).view(bsz, K, -1)
        # print(cond_trans_mat.shape)
        # nufrom, nuto each bsz x K x A_dim
        A_dim = self.cond_A_dim
        nufrom, nuto = cond_trans_mat[:, :, :A_dim], cond_trans_mat[:, :, A_dim:]
        A_from, A_to = self.A_from, self.A_to
        if self.dropout.p > 0:
            A_from = self.dropout(A_from)
            nufrom = self.dropout(nufrom)
        tscores = torch.mm(A_from, A_to)
        if not self.yes_self_trans:
            tscores = tscores + self.selfmask
        trans_lps = tscores.unsqueeze(0).expand(bsz, K, K)
        trans_lps = trans_lps + torch.bmm(nufrom, nuto.transpose(1, 2))
        trans_lps = self.lsm(trans_lps.view(-1, K)).view(bsz, K, K)

        init_lps = self.lsm(self.init_lin(uniqenc))  # bsz x K
        trans_lps = trans_lps.view(1, bsz, K, K).expand(seqlen - 1, bsz, K, K)
        return init_lps, trans_lps

    def encode(self, src, avgmask, uniqfields):
        """
        args:
          src - bsz x nfields x nfeats
          avgmask - bsz x nfields, with 0s for pad and 1/tru_nfields for rest
          uniqfields - bsz x maxfields
        returns bsz x emb_size, bsz x nfields x emb_size
        """
        bsz, nfields, nfeats = src.size()
        emb_size = self.word_embeds.embedding_dim
        # do src stuff that depends on words
        embs = self.word_embeds(src.view(-1, nfeats))  # bsz*nfields x nfeats x emb_size
        if self.max_pool:
            embs = F.relu(embs.sum(1) + self.src_bias.expand(bsz * nfields, emb_size))
            if avgmask is not None:
                masked = (embs.view(bsz, nfields, emb_size)
                          * avgmask.unsqueeze(2).expand(bsz, nfields, emb_size))
            else:
                masked = embs.view(bsz, nfields, emb_size)
            srcenc = F.max_pool1d(masked.transpose(1, 2), nfields).squeeze(2)  # bsz x emb_size
        else:
            embs = F.tanh(embs.sum(1) + self.src_bias.expand(bsz * nfields, emb_size))
            # average it manually, bleh
            if avgmask is not None:
                srcenc = (embs.view(bsz, nfields, emb_size)
                          * avgmask.unsqueeze(2).expand(bsz, nfields, emb_size)).sum(1)
            else:
                srcenc = embs.view(bsz, nfields, emb_size).mean(1)  # bsz x emb_size

        srcfieldenc = embs.view(bsz, nfields, emb_size)

        # do stuff that depends only on uniq fields
        uniqenc = self.word_embeds(uniqfields).sum(1)  # bsz x nfields x emb_size -> bsz x emb_size

        # add a bias
        uniqenc = uniqenc + self.uniq_bias.expand_as(uniqenc)
        uniqenc = F.relu(uniqenc)

        return {"srcenc":srcenc, "srcfieldenc":srcfieldenc, "uniqenc":uniqenc}


    def get_score(self, z, dict_computed):
        '''
        compute the unnormalized score of p(y,z | x)

        :param x:
        :param y:
        :param z:
        :return:
        '''
        emission = dict_computed['emission']
        transition = dict_computed['transition'] # bsz * K * K
        length_prob = dict_computed['length']
        init = dict_computed['init'] # bsz * K
        sample_num = len(z[0])

        L, seqlen, bsz, K = emission.shape
        result_lst_all = []
        for b in range(bsz):
            result_lst = []
            for sample_idx in range(sample_num):
                start_lst, end_lst, state_lst = z[b][sample_idx]
                result = 0
                state_prev = -1
                for idx in range(len(start_lst)):
                    start_pos, end_pos, state_curr = start_lst[idx], end_lst[idx] - 1, state_lst[idx]
                    length = end_pos - start_pos
                    result += emission[length, start_pos, b , state_curr] + \
                              length_prob[min(L-1, seqlen-1-start_pos)][length, state_curr] # L * T * bsz * K
                    if state_prev  == -1:
                        result += init[b, state_curr]
                        state_prev = state_curr
                    else:
                        result += transition[start_pos-1, b, state_prev, state_curr]
                        state_prev = state_curr
                result_lst.append(result)
            result_lst_all.append(result_lst)
        return result_lst_all

    def recover_bps(self, delt, bps, bps_star):
        """
        delt, bps, bps_star - seqlen+1 x bsz x K
        returns:
           bsz-length list of lists with (start_idx, end_idx, label) entries
        """
        seqlenp1, bsz, K = delt.size()
        seqlen = seqlenp1 - 1
        seqs = []
        for b in range(bsz):
            seq = []
            _, last_lab = delt[seqlen][b].max(0)
            last_lab = last_lab.item()
            curr_idx = seqlen  # 1-indexed
            while True:
                last_len = bps[curr_idx][b][last_lab].item()
                seq.append((curr_idx - last_len, curr_idx, last_lab))  # start_idx, end_idx, label, 0-idxd
                curr_idx -= last_len
                if curr_idx == 0:
                    break
                last_lab = bps_star[curr_idx][b][last_lab].item()
            seqs.append(seq[::-1])
        return seqs

    def viterbi(self, dict_computed, constraints=None, ret_delt=False):
        """
        pi               - 1 x K
        bwd_obs_logprobs - L x T x bsz x K, obs probs ending at t
        trans_logprobs   - T-1 x bsz x K x K, trans_logprobs[t] = p(q_{t+1} | q_t).
        see https://hal.inria.fr/hal-01064672v2/document
        """
        pi, trans_logprobs, bwd_obs_logprobs, len_logprobs, = dict_computed['init'], dict_computed['transition'], \
                                              dict_computed['emission_alpha'], dict_computed['length']
        neginf = -1e38
        L, seqlen, bsz, K = bwd_obs_logprobs.size()
        delt = trans_logprobs.new(seqlen + 1, bsz, K).fill_(neginf)
        delt_star = trans_logprobs.new(seqlen + 1, bsz, K).fill_(neginf)
        delt_star[0].copy_(pi.expand(bsz, K))

        # currently len_logprobs contains tensors that are [1 step back; 2 steps back; ... L steps_back]
        # but we need to flip on the 0'th axis
        flipped_len_logprobs = []
        for l in range(len(len_logprobs)):
            llps = len_logprobs[l]
            flipped_len_logprobs.append(torch.stack([llps[-i - 1] for i in range(llps.size(0))]))

        bps = delt.long().fill_(L)
        bps_star = delt_star.long()
        bps_star[0].copy_(torch.arange(0, K).view(1, K).expand(bsz, K))

        mask = trans_logprobs.new(L, bsz, K)

        for t in range(1, seqlen + 1):
            steps_back = min(L, t)
            steps_fwd = min(L, seqlen - t + 1)

            if steps_back <= steps_fwd:
                # steps_fwd x K -> steps_back x K
                len_terms = flipped_len_logprobs[min(L - 1, steps_fwd - 1)][-steps_back:]
            else:  # we need to pick probs from different distributions...
                len_terms = torch.stack([len_logprobs[min(L, seqlen + 1 - t + jj) - 1][jj]
                                         for jj in range(L - 1, -1, -1)])

            if constraints is not None and constraints[t] is not None:
                tmask = mask.narrow(0, 0, steps_back).zero_()
                # steps_back x bsz x K -> steps_back*bsz x K
                tmask.view(-1, K).index_fill_(0, constraints[t], neginf)

            # delt_t(j) = log \sum_l p(x_{t-l+1:t}) delt*_{t-l} p(l_t)
            delt_terms = (delt_star[t - steps_back:t]  # steps_back x bsz x K
                          + bwd_obs_logprobs[-steps_back:, t - 1])  # steps_back x bsz x K (0-idx)
            # delt_terms.sub_(bwd_maxlens[t-steps_back:t].expand_as(delt_terms)) # steps_back x bsz x K
            delt_terms.add_(len_terms.unsqueeze(1).expand(steps_back, bsz, K))

            if constraints is not None and constraints[t] is not None:
                delt_terms.add_(tmask)

            maxes, argmaxes = torch.max(delt_terms, 0)  # 1 x bsz x K, 1 x bsz x K
            delt[t] = maxes.squeeze(0)  # bsz x K
            # bps[t] = argmaxes.squeeze(0) # bsz x K
            bps[t].sub_(argmaxes.squeeze(0))  # keep track of steps back taken: L - argmax
            if steps_back < L:
                bps[t].sub_(L - steps_back)
            if t < seqlen:
                # delt*_t(k) = log \sum_j delt_t(j) p(q_{t+1}=k | q_t = j)
                # get bsz x K x K trans logprobs, viz., p(q_{t+1}=j|i) w/ 0th dim i, 2nd dim j
                tps = trans_logprobs[t - 1]  # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-indexed
                delt_t = delt[t]  # bsz x K, viz, p(x, j)
                delt_star_terms = (tps.transpose(0, 1)  # K x bsz x K
                                   + delt_t.unsqueeze(2).expand(bsz, K, K).transpose(0, 1))
                maxes, argmaxes = torch.max(delt_star_terms, 0)  # 1 x bsz x K, 1 x bsz x K
                delt_star[t] = maxes.squeeze(0)
                bps_star[t] = argmaxes.squeeze(0)

        # return delt, delt_star, bps, bps_star, recover_bps(delt, bps, bps_star)
        if ret_delt:
            return self.recover_bps(delt, bps, bps_star), delt[-1]  # bsz x K total scores
        else:
            return self.recover_bps(delt, bps, bps_star)


    def just_fwd(self, pi, trans_logprobs, bwd_obs_logprobs, len_logprobs, constraints=None):
        # TEMP
        """
        pi               - bsz x K
        bwd_obs_logprobs - L x T x bsz x K, obs probs ending at t
        trans_logprobs   - T-1 x bsz x K x K, trans_logprobs[t] = p(q_{t+1} | q_t)
        """
        neginf = -1e38  # -float("inf")
        L, seqlen, bsz, K = bwd_obs_logprobs.size()
        # we'll be 1-indexed for alphas and betas
        alph = [None] * (seqlen + 1)
        alph_star = [None] * (seqlen + 1)
        alph_star[0] = pi
        mask = trans_logprobs.new(L, bsz, K)

        bwd_maxlens = trans_logprobs.new(seqlen).fill_(L)  # store max possible length generated from t
        bwd_maxlens[-L:].copy_(torch.arange(L, 0, -1))
        bwd_maxlens = bwd_maxlens.log_().view(seqlen, 1, 1)

        for t in range(1, seqlen + 1):
            steps_back = min(L, t)
            len_terms = len_logprobs[min(L - 1, steps_back - 1)]  # steps_fwd x K

            if constraints is not None and constraints[t] is not None:
                tmask = mask.narrow(0, 0, steps_back).zero_()
                # steps_back x bsz x K -> steps_back*bsz x K
                tmask.view(-1, K).index_fill_(0, constraints[t], neginf)

            # alph_t(j) = log \sum_l p(x_{t-l+1:t}) alph*_{t-l} p(l_t)
            # print('hello -here ')
            # print(bwd_obs_logprobs[-steps_back:, t - 1])
            alph_terms = (torch.stack(alph_star[t - steps_back:t])  # steps_back x bsz x K
                          + bwd_obs_logprobs[-steps_back:, t - 1]  # steps_back x bsz x K (0-idx)
                          # + len_terms.unsqueeze(1).expand(steps_back, bsz, K))
                          - bwd_maxlens[t - steps_back:t].expand(steps_back, bsz, K))
            if constraints is not None and constraints[t] is not None:
                alph_terms = alph_terms + tmask  # Variable(tmask)

            alph[t] = logsumexp0(alph_terms)  # bsz x K

            if t < seqlen:
                # alph*_t(k) = log \sum_j alph_t(j) p(q_{t+1}=k | q_t = j)
                # get bsz x K x K trans logprobs, viz., p(q_{t+1}=j|i) w/ 0th dim i, 2nd dim j
                tps = trans_logprobs[t - 1]  # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-indexed
                alph_t = alph[t]  # bsz x K, viz, p(x, j)

                alph_star_terms = (tps.transpose(0, 1)  # K x bsz x K
                                   + alph_t.unsqueeze(2).expand(bsz, K, K).transpose(0, 1))
                alph_star[t] = logsumexp0(alph_star_terms)

        print('result from forward pass of alphas is ', logsumexp1(alph[seqlen]))
        return alph, alph_star

    def just_bwd(self, trans_logprobs, fwd_obs_logprobs, len_logprobs, constraints=None):
        """
        fwd_obs_logprobs - L x T x bsz x K, obs probs starting at t
        trans_logprobs   - T-1 x bsz x K x K, trans_logprobs[t] = p(q_{t+1} | q_t)
        """
        neginf = -1e38  # -float("inf")
        # print(fwd_obs_logprobs.shape)
        L, seqlen, bsz, K = fwd_obs_logprobs.size()

        # we'll be 1-indexed for alphas and betas
        beta = [None] * (seqlen + 1)
        beta_star = [None] * (seqlen + 1)
        beta[seqlen] = Variable(trans_logprobs.data.new(bsz, K).zero_())
        mask = trans_logprobs.data.new(L, bsz, K)

        for t in range(1, seqlen + 1):
            steps_fwd = min(L, t)
            # print(len_logprobs)
            len_terms = len_logprobs[min(L - 1, steps_fwd - 1)]  # steps_fwd x K

            if constraints is not None and constraints[seqlen - t + 1] is not None:
                tmask = mask.narrow(0, 0, steps_fwd).zero_()
                # steps_fwd x bsz x K -> steps_fwd*bsz x K
                tmask.view(-1, K).index_fill_(0, constraints[seqlen - t + 1], neginf)

            # beta*_t(k) = log \sum_l beta_{t+l}(k) p(x_{t+1:t+l}) p(l_t)
            # print('funny:', fwd_obs_logprobs[:steps_fwd, seqlen - t])
            beta_star_terms = (torch.stack(beta[seqlen - t + 1:seqlen - t + 1 + steps_fwd])  # steps_fwd x bsz x K
                               + fwd_obs_logprobs[:steps_fwd, seqlen - t]  # steps_fwd x bsz x K
                               + len_terms.unsqueeze(1).expand(steps_fwd, bsz, K))

            if constraints is not None and constraints[seqlen - t + 1] is not None:
                beta_star_terms = beta_star_terms + Variable(tmask)

            beta_star[seqlen - t] = logsumexp0(beta_star_terms)
            if seqlen - t > 0:
                # beta_t(j) = log \sum_k beta*_t(k) p(q_{t+1} = k | q_t=j)
                betastar_nt = beta_star[seqlen - t]  # bsz x K
                # get bsz x K x K trans logprobs, viz., p(q_{t+1}=j|i) w/ 0th dim i, 2nd dim j
                tps = trans_logprobs[seqlen - t - 1]  # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-idxed
                beta_terms = betastar_nt.unsqueeze(1).expand(bsz, K, K) + tps  # bsz x K x K
                beta[seqlen - t] = logsumexp2(beta_terms)  # bsz x K
        # print(beta, beta_star)
        return beta, beta_star

    def get_Z(self, trans_logprobs, fwd_obs_logprobs, len_logprobs, init):
        '''
        compute the partition function for this HSMM of p(y | x).
        :param x:
        :param y:
        :return:
        '''
        beta, beta_star = self.just_bwd(trans_logprobs, fwd_obs_logprobs, len_logprobs)
        print(beta_star[0].shape, init.shape)
        Z = logsumexp1(beta_star[0] + init)
        # print('partition from Z is {}'.format(Z))
        return Z

    def get_sample(self, dict_computed, constraints=None, sample_num=1):
        '''
        sample from the HSMM model p(z | x, y). Return a list of list. The outer list has
        length = bsz, and the inner list has length = sample_num. Thus, the ith sample of the
        jth x is result[j][i] = start_lst, end_lst, state_lst.
        :param x:
        :param y:
        :return:
        '''

        result_lst_style_all = []
        result_vtb_style_all = []
        emission = dict_computed['emission']
        transition = dict_computed['transition']
        length_prob = dict_computed['length']
        beta_star = dict_computed['beta_star']
        init = dict_computed['init']
        beta = dict_computed['beta']
        neginf = -1e38  # -float("inf")
        L, seqlen, bsz, K = emission.size()

        ''' To sample, we can use the beta scores, '''

        for b in range(bsz):
            result_lst_style = []
            result_vtb_style = []
            for sample_idx in range(sample_num):
                # print('samplng from the member of batch {}'.format(b))
                length_lst = [0]
                prob = nn.Softmax(dim=0)(beta_star[length_lst[-1]][b] + init[b])
                try:
                    sample = torch.multinomial(prob, 1).item()
                    state_lst = [sample]
                except:
                    print(beta)
                    print(beta_star)
                    print(init)
                    print(beta_star[length_lst[-1]][b] + init[b])
                    print(prob)
                    print('dead')
                    10/0

                t = seqlen
                while(t >= 1):
                    # print('current t = {}, seqlen = {}'.format(seqlen - t, seqlen))
                    # t = seqlen + 1 - t
                    steps_fwd = min(L, t)
                    len_terms = length_prob[min(L - 1, steps_fwd - 1)]  # steps_fwd x K

                    if seqlen - t > 0:
                        # beta_t(j) = log \sum_k beta*_t(k) p(q_{t+1} = k | q_t=j)
                        betastar_nt = beta_star[seqlen - t][b]  # bsz x K
                        # get bsz x K x K trans logprobs, viz., p(q_{t+1}=j|i) w/ 0th dim i, 2nd dim j
                        tps = transition[seqlen - t - 1]  # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-idxed
                        # print(tps.shape)
                        prob_state = nn.Softmax(dim=0)(betastar_nt + tps[b,state_lst[-1]])  # bsz x K
                        sample_state = torch.multinomial(prob_state, 1).item()
                        state_lst.append(sample_state)
                        # print("sampled_state is {}".format(state_lst[-1]))


                    if constraints is not None and constraints[seqlen - t + 1] is not None:
                        tmask = mask.narrow(0, 0, steps_fwd).zero_()
                        # steps_fwd x bsz x K -> steps_fwd*bsz x K
                        tmask.view(-1, K).index_fill_(0, constraints[seqlen - t + 1], neginf)

                    # beta*_t(k) = log \sum_l beta_{t+l}(k) p(x_{t+1:t+l}) p(l_t)
                    prob_dist = nn.Softmax(dim=0)(torch.stack(beta[seqlen - t + 1:seqlen - t + 1 + steps_fwd])[:, b, state_lst[-1]]
                                                     + emission[:steps_fwd, seqlen - t, b, state_lst[-1]] +
                                                     len_terms.unsqueeze(1).expand(steps_fwd, bsz, K)[:, b, state_lst[-1]])
                    sample_dist = torch.multinomial(prob_dist, 1).item()
                    length_lst.append(sample_dist + 1)
                    # print('sampled_length={}'.format(length_lst[-1]))

                    t = t - length_lst[-1]
                    # beta_star_terms = (torch.stack(beta[seqlen - t + 1:seqlen - t + 1 + steps_fwd])  # steps_fwd x bsz x K
                    #                    + emission[:steps_fwd, seqlen - t]  # steps_fwd x bsz x K
                    #                    + len_terms.unsqueeze(1).expand(steps_fwd, bsz, K))

                # print('FINALLY, current t = {}, seqlen = {}'.format(seqlen - t, seqlen))
                # print(length_lst, state_lst)
                lst_style_sample, viterb_style_sample = self.process_sample(length_lst, state_lst)
                result_lst_style.append(lst_style_sample)
                result_vtb_style.append(viterb_style_sample)
            result_lst_style_all.append(result_lst_style)
            result_vtb_style_all.append(result_vtb_style)

        return result_lst_style_all, result_vtb_style_all


    def process_sample(self, length_lst, state_lst):
        end_lst = np.cumsum(length_lst[1:])
        start_lst = np.cumsum(length_lst[1:]) - length_lst[1:]

        # viterbi style
        viterb_style = []
        for idx, (start, end, state) in enumerate(zip(start_lst, end_lst, state_lst)):
            viterb_style.append( (start, end, state))
        return (start_lst, end_lst, state_lst), viterb_style


    def get_entr(self, dict_computed, constraints=None):
        '''
        compute the entropy of H(Z | X,Y) from the beta values

        :param x:
        :param y:
        :return:
        '''
        fwd_obs_logprobs = dict_computed['emission']
        trans_logprobs = dict_computed['transition']
        len_logprobs = dict_computed['length']
        init = dict_computed['init']

        neginf = -1e38  # -float("inf")
        L, seqlen, bsz, K = fwd_obs_logprobs.size()

        # we'll be 1-indexed for alphas and betas
        entr = [None] * (seqlen + 1)
        entr_star = [None] * (seqlen + 1)

        beta = [None] * (seqlen + 1)
        beta_star = [None] * (seqlen + 1)

        beta[seqlen] = Variable(trans_logprobs.data.new(bsz, K).zero_())
        entr[seqlen] = Variable(trans_logprobs.data.new(bsz, K).zero_())
        mask = trans_logprobs.data.new(L, bsz, K)

        for t in range(1, seqlen + 1):
            steps_fwd = min(L, t)
            # print(len_logprobs)
            len_terms = len_logprobs[min(L - 1, steps_fwd - 1)]  # steps_fwd x K

            if constraints is not None and constraints[seqlen - t + 1] is not None:
                tmask = mask.narrow(0, 0, steps_fwd).zero_()
                # steps_fwd x bsz x K -> steps_fwd*bsz x K
                tmask.view(-1, K).index_fill_(0, constraints[seqlen - t + 1], neginf)

            # beta*_t(k) = log \sum_l beta_{t+l}(k) p(x_{t+1:t+l}) p(l_t)
            # dim = steps_fwd * bsz * K


            beta_star_terms = (torch.stack(beta[seqlen - t + 1:seqlen - t + 1 + steps_fwd])  # steps_fwd x bsz x K
                              + fwd_obs_logprobs[:steps_fwd, seqlen - t]  # steps_fwd x bsz x K
                              + len_terms.unsqueeze(1).expand(steps_fwd, bsz, K))

            if constraints is not None and constraints[seqlen - t + 1] is not None:
                beta_star_terms = beta_star_terms + Variable(tmask)

            beta_star[seqlen - t] = logsumexp0(beta_star_terms) # bsz * K
            weight_logprob = beta_star_terms - beta_star[seqlen - t].unsqueeze(0).expand(steps_fwd, bsz, K)
            weight_prob = weight_logprob.exp()
            entr_star_terms = weight_prob * (torch.stack(entr[seqlen - t + 1:seqlen - t + 1 + steps_fwd])
                                             - weight_logprob)  # steps_fwd x bsz x K
            # print(entr_star_terms)
            entr_star[seqlen - t] = torch.sum(entr_star_terms, dim=0)
            # print(entr_star[seqlen - t])
            # print('99999' * 10)
            # print(entr_star[seqlen - t].shape)
            # print(beta_star[seqlen - t].shape)

            if seqlen - t > 0:
                # beta_t(j) = log \sum_k beta*_t(k) p(q_{t+1} = k | q_t=j)
                betastar_nt = beta_star[seqlen - t]  # bsz x K
                # get bsz x K x K trans logprobs, viz., p(q_{t+1}=j|i) w/ 0th dim i, 2nd dim j
                tps = trans_logprobs[seqlen - t - 1]  # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-idxed
                beta_terms = betastar_nt.unsqueeze(1).expand(bsz, K, K) + tps  # bsz x K x K
                beta[seqlen - t] = logsumexp2(beta_terms)  # bsz x K
                beta_logprob = beta_terms - beta[seqlen - t].unsqueeze(2).expand(bsz, K, K)
                beta_prob = beta_logprob.exp()
                # print('66666' * 10)
                # print(tps)
                # print(beta_logprob)
                # print(beta_prob)
                # print('77777' * 10)
                entr_terms = beta_prob * (entr_star[seqlen - t].unsqueeze(1).expand(bsz, K, K) - beta_logprob )
                entr[seqlen - t] = torch.sum(entr_terms, dim=2)

        Z_terms = beta_star[0] + init
        Z = logsumexp1(Z_terms)
        # print("compute partition is {}".format(Z))
        # print(entr_star)
        # print(entr)
        Z_logprob = Z_terms - Z.expand(bsz, K)
        Z_prob = Z_logprob.exp()
        entr_terms = Z_prob * (entr_star[0] - Z_logprob)
        entr_Z = torch.sum(entr_terms, dim=1)
        # print('the computed entropy is {}'.format(entr_Z))
        dict_computed['beta'] = beta
        dict_computed['beta_star'] = beta_star
        dict_computed['entr'] = entr
        dict_computed['entr_star'] = entr_star
        dict_computed['Z'] = Z
        dict_computed['entropy'] = entr_Z

        return Z, entr_Z


    def get_weights(self, sent, uniqenc):
        scores2 = self.get_span_scores(sent)
        scores_beta = self.get_emission_score(scores2)
        scores_alphas = bwd_from_fwd_obs_logprobs(scores_beta.data)
        init, trans = self.get_transition_score(uniqenc, sent.size()[1])
        len_lst, len_score = self.get_length_score()
        dict_computed = {}
        dict_computed['emission'] = scores_beta
        dict_computed['emission_alpha'] = scores_alphas
        dict_computed['transition'] = trans
        dict_computed['length'] = len_lst
        dict_computed['init'] = init
        return dict_computed

    def save_model(self, path = 'latest_model'):
        torch.save(self.state_dict(), path)
        # with open(path, 'bw') as f:
        # pickle.dump(self, f)

    def load_model(self, path = 'latest_model'):
        self.load_state_dict(torch.load(path))
        # self = torch.load(path)
        return self



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-data', type=str, default='', help='path to data dir')
    parser.add_argument('-epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('-bsz', type=int, default=16, help='batch size')
    parser.add_argument('-seed', type=int, default=1111, help='random seed')
    parser.add_argument('-cuda', action='store_true', help='use CUDA')
    parser.add_argument('-log_interval', type=int, default=200,
                        help='minibatches to wait before logging training status')
    parser.add_argument('-save', type=str, default='', help='path to save the final model')
    parser.add_argument('-load', type=str, default='', help='path to saved model')
    parser.add_argument('-test', action='store_true', help='use test data')
    parser.add_argument('-thresh', type=int, default=9, help='prune if occurs <= thresh')
    parser.add_argument('-max_mbs_per_epoch', type=int, default=35000, help='max minibatches per epoch')

    parser.add_argument('-emb_size', type=int, default=100, help='size of word embeddings')
    parser.add_argument('-hid_size', type=int, default=100, help='size of rnn hidden state')
    parser.add_argument('-layers', type=int, default=1, help='num rnn layers')
    parser.add_argument('-A_dim', type=int, default=64,
                        help='dim of factors if factoring transition matrix')
    parser.add_argument('-cond_A_dim', type=int, default=32,
                        help='dim of factors if factoring transition matrix')
    parser.add_argument('-smaller_cond_dim', type=int, default=64,
                        help='dim of thing we feed into linear to get transitions')
    parser.add_argument('-yes_self_trans', action='store_true', help='')
    parser.add_argument('-mlpinp', action='store_true', help='')
    parser.add_argument('-mlp_sz_mult', type=int, default=2, help='mlp hidsz is this x emb_size')
    parser.add_argument('-max_pool', action='store_true', help='for word-fields')

    parser.add_argument('-constr_tr_epochs', type=int, default=100, help='')
    parser.add_argument('-no_ar_epochs', type=int, default=100, help='')

    parser.add_argument('-word_ar', action='store_true', help='')
    parser.add_argument('-ar_after_decay', action='store_true', help='')
    parser.add_argument('-no_ar_for_vit', action='store_true', help='')
    parser.add_argument('-fine_tune', action='store_true', help='only train ar rnn')

    parser.add_argument('-dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('-emb_drop', action='store_true', help='dropout on embeddings')
    parser.add_argument('-lse_obj', action='store_true', help='')
    parser.add_argument('-sep_attn', action='store_true', help='')
    parser.add_argument('-max_seqlen', type=int, default=70, help='')

    parser.add_argument('-K', type=int, default=10, help='number of states')
    parser.add_argument('-Kmul', type=int, default=1, help='number of states multiplier')
    parser.add_argument('-L', type=int, default=10, help='max segment length')
    parser.add_argument('-unif_lenps', action='store_true', help='')
    parser.add_argument('-one_rnn', action='store_true', help='')

    parser.add_argument('-initrange', type=float, default=0.1, help='uniform init interval')
    parser.add_argument('-lr_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('-optim', type=str, default="sgd", help='optimization algorithm')
    parser.add_argument('-onmt_decay', action='store_true', help='')
    parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('-interactive', action='store_true', help='')
    parser.add_argument('-label_train', action='store_true', help='')
    parser.add_argument('-gen_from_fi', type=str, default='', help='')
    parser.add_argument('-verbose', action='store_true', help='')
    parser.add_argument('-prev_loss', type=float, default=None, help='')
    parser.add_argument('-best_loss', type=float, default=None, help='')

    parser.add_argument('-tagged_fi', type=str, default='', help='path to tagged fi')
    parser.add_argument('-ntemplates', type=int, default=200, help='num templates for gen')
    parser.add_argument('-beamsz', type=int, default=1, help='')
    parser.add_argument('-gen_wts', type=str, default='1,1', help='')
    parser.add_argument('-min_gen_tokes', type=int, default=0, help='')
    parser.add_argument('-min_gen_states', type=int, default=0, help='')
    parser.add_argument('-gen_on_valid', action='store_true', help='')
    parser.add_argument('-align', action='store_true', help='')
    parser.add_argument('-wid_workers', type=str, default='', help='')

    parser.add_argument('-embedding_dim', type=int, default=50, help='')
    parser.add_argument('-hidden_dim', type=int, default=50, help='')
    parser.add_argument('-vocab_size', type=int, default=1000, help='')
    parser.add_argument('-tagset_size', type=int, default=10, help='')
    parser.add_argument('-q_dim', type=int, default=50, help='')
    parser.add_argument('-q_lr', type=int, default=0.05, help='')
    parser.add_argument('-action_lr', type=int, default=0.05, help='')
    parser.add_argument('-lr', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('-conditional_dim', type=int, default=30, help='')
    parser.add_argument('-train_q_epochs', type=int, default=30, help='')
    parser.add_argument('--max_grad_norm', default=5, type=float, help='gradient clipping parameter')
    parser.add_argument('--q_max_grad_norm', default=1, type=float, help='gradient clipping parameter for q')

    # -------- TESTING the HSMM CRF -------------------------------------------
    args = parser.parse_args()

    torch.manual_seed(66)
    crf = HSMM(args)

    optimizer = torch.optim.Adam(crf.parameters(), lr=0.01)

    sent1 = torch.LongTensor([5, 0, 6,  7, 10, 120, 50]).view(1, -1)
    sent2 = torch.LongTensor([3, 4, 3, 4, 11, 151, 51]).view(1, -1)
    sent = torch.cat([sent1, sent2], dim=0)  # assume that batch_size=2


    bsz = 50
    sent_lst = []
    for i in range(bsz):
        sent = torch.LongTensor(1, 7).random_(20, 100).view(1, -1)
        sent_lst.append(sent)

    sent = torch.cat(sent_lst, dim=0)

    # print('-- debugging 1 --'*5)
    #
    # print(init.shape)
    # print(scores2.shape)
    # print(scores_beta.shape)
    # print(scores_alphas.shape)
    #
    # print(len_lst[0].shape, len_lst[1].shape, len(len_lst))

    # print(trans.shape)
    # # scores = torch.randn(5,8,2,4)
    # trans = torch.randn(6,2,10,10)
    # init = torch.randn(2,10)
    # len_lst = [torch.randn(i, 10) for i in range(1, 4)]
    #
    # scores_beta = torch.randn(3, 3, 2, 10) * 1
    # print(scores_beta.shape)
    #
    # scores = crf.get_emission_score(scores2)
    # scores_alphas = crf.get_emission_score_for_alpha(scores2)

    scores2 = crf.get_span_scores(sent)
    scores_beta = crf.get_emission_score(scores2)
    scores_alphas = crf.get_emission_score_for_alpha(scores2)
    uniqenc = torch.randn([bsz, 30])
    init, trans = crf.get_transition_score(uniqenc, sent.size()[1])
    len_lst, len_score = crf.get_length_score()

    dict_computed = {}
    dict_computed['emission'] = scores_beta
    dict_computed['transition'] = trans
    dict_computed['emission_alpha'] = scores_alphas
    dict_computed['length'] = len_lst
    dict_computed['init'] = init
    Z2, entr_Z = crf.get_entr(dict_computed)


    samples_lst, samples_vtb = crf.get_sample(dict_computed, sample_num=1)
    # for idx, elem in enumerate(samples_vtb):
    #     print(idx, elem)

    viterb = crf.viterbi(dict_computed)
    # for idx, elem in enumerate(viterb):
    #     # print(get_acc_seg(elem, samples_vtb[idx]))
    #     print(idx, elem)


    for idx in range(80):
        scores2 = crf.get_span_scores(sent)
        scores_beta = crf.get_emission_score(scores2)
        scores_alphas = crf.get_emission_score_for_alpha(scores2)
        # uniqenc = torch.randn([bsz, 30])
        init, trans = crf.get_transition_score(uniqenc, sent.size()[1])
        len_lst, len_score = crf.get_length_score()

        dict_computed = {}
        dict_computed['emission'] = scores_beta
        dict_computed['emission_alpha'] = scores_alphas
        dict_computed['transition'] = trans
        dict_computed['length'] = len_lst
        dict_computed['init'] = init

        # score2 = torch.randn(1,3,2,1)
        # save_L = crf.L
        # crf.L = 3
        # scores_beta = crf.get_emission_score(scores2)
        # scores_alphas = crf.get_emission_score_for_alpha(scores2)

        # torch.set_printoptions(precision=2)
        # print('hello')
        # print(scores_beta)
        # print(']'*50)
        # print(scores_alphas)

        Z2, entr_Z = crf.get_entr(dict_computed)
        sample_score = crf.get_score(samples_lst, dict_computed)
        # print(sample_score)
        # print(Z2)
        target = [torch.stack(samples) for samples in sample_score]
        target = torch.stack(target, dim=0)
        bsz, num_sample = target.shape
        target = (target - Z2.expand(bsz, num_sample)).sum()
        # target = (torch.stack(sample_score)-Z2.view(-1)).sum()
        (-target).backward()
        if idx % 10 == 0:
            pass

        optimizer.step()
        optimizer.zero_grad()

    crf.save_model()

    for _ in range(20):
        scores2 = crf.get_span_scores(sent)
        scores_beta = crf.get_emission_score(scores2)
        scores_alphas = crf.get_emission_score_for_alpha(scores2)
        # uniqenc = torch.randn([bsz, 30])
        init, trans = crf.get_transition_score(uniqenc, sent.size()[1])
        len_lst, len_score = crf.get_length_score()

        dict_computed = {}
        dict_computed['emission'] = scores_beta
        dict_computed['emission_alpha'] = scores_alphas
        dict_computed['transition'] = trans
        dict_computed['length'] = len_lst
        dict_computed['init'] = init
        Z2, entr_Z = crf.get_entr(dict_computed)
        samples_lst_2, samples_vtb_2 = crf.get_sample(dict_computed)
        sample_score = crf.get_score(samples_lst_2, dict_computed)

        # print('1105')
        # print(sample_score)
        target = [torch.stack(samples) for samples in sample_score]
        target = torch.stack(target, dim=0)
        bsz, num_sample = target.shape
        target = (target - Z2.expand(bsz, num_sample))
        # print(target.shape)
        # print(target.shape)
        # print()
        acc_avg = 0
        for idx, elem in  enumerate(samples_vtb_2):
            # only one sample per example. so just add a zero-index
            acc_avg += get_acc_seg(elem[0], samples_vtb[idx][0])
            # print(idx, elem, target[idx].exp().item())
            # print(idx, samples_vtb[idx])
            # print(get_acc_seg(elem, samples_vtb[idx]))
        print('avg_acc for sampling is {}'.format(acc_avg / len(viterb)))


    # crf.just_fwd(init, trans, scores_alphas, len_lst)
    # samples_lst, samples_vtb = crf.get_sample( dict_computed)
    # sample_score = crf.get_score( samples_lst, dict_computed )
    # crf.entr(alphas)
    # print('hi')
    # print(entr_Z, Z2)
    Z2, entr_Z = crf.get_entr2(dict_computed)
    # print(Z2)
    # print(entr_Z)
    # print(samples)
    # print(sample_score)
    viterb = crf.viterbi(dict_computed)
    acc_avg = 0
    for idx, elem in enumerate(viterb):
        temp = get_acc_seg(elem, samples_vtb[idx][0])
        acc_avg += temp
    print("average accuracy for viterbi is {}".format(acc_avg/len(viterb)))