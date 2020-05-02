
    def convert_state_(self, z):
        T = len(z)
        prev_state = -1
        idx = 0
        start = []
        end = []
        state = []
        while (idx < T):
            if prev_state != z[idx]:
                state.append(z[idx])
                if idx - 1 >= 0:
                    end.append(idx - 1)
                start.append(idx)
                prev_state = state[-1]
            else:
                idx += 1
        end.append(T - 1)
        return [(start, end, state)]


    def get_entr2(self, dict_computed, constraints=None):
        '''
        compute the entropy of H(Z | X,Y) from the beta values

        :param x:
        :param y:
        :return:
        '''

        # grads = {}
        #
        # def save_grad(name):
        #     def hook(grad):
        #         grads[name] = grad
        #
        #     return hook


        # In here, save_grad('y') returns a hook (a function) that keeps 'y' as name
        # y.register_hook(save_grad('y'))
        # z.register_hook(save_grad('z'))
        # z.backward()
        #
        # print(grads['y'])
        # print(grads['z'])



        fwd_obs_logprobs = dict_computed['emission']
        trans_logprobs = dict_computed['transition']
        len_logprobs = dict_computed['length']
        init = dict_computed['init']

        # trans_logprobs.register_hook(save_grad('trans'))
        self.trans = nn.Parameter(torch.zeros(trans_logprobs.shape))
        trans_logprobs = self.trans + trans_logprobs

        self.emission = nn.Parameter(torch.zeros(fwd_obs_logprobs.shape))
        fwd_obs_logprobs = fwd_obs_logprobs + self.emission

        len_logprobs_2 = []
        for idx, elem in enumerate(len_logprobs):
            len_logprobs_2.append(torch.zeros(elem.shape, requires_grad=True))
            len_logprobs[idx] += len_logprobs_2[-1]

        # len_logprobs_2 = []
        # for idx, elem in enumerate(len_logprobs):
        #     elem = Variable(elem)
        #     elem.register_hook(save_grad('length{}'.format(idx)))
        #     len_logprobs2.append(elem)

        # len_logprobs = len_logprobs_2

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

            if seqlen - t > 0:
                # beta_t(j) = log \sum_k beta*_t(k) p(q_{t+1} = k | q_t=j)
                betastar_nt = beta_star[seqlen - t]  # bsz x K
                # get bsz x K x K trans logprobs, viz., p(q_{t+1}=j|i) w/ 0th dim i, 2nd dim j
                tps = trans_logprobs[seqlen - t - 1]  # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-idxed
                beta_terms = betastar_nt.unsqueeze(1).expand(bsz, K, K) + tps  # bsz x K x K
                beta[seqlen - t] = logsumexp2(beta_terms)  # bsz x K
                beta_logprob = beta_terms - beta[seqlen - t].unsqueeze(2).expand(bsz, K, K)
                beta_prob = beta_logprob.exp()
                entr_terms = beta_prob * (entr_star[seqlen - t].unsqueeze(1).expand(bsz, K, K) - beta_logprob )
                entr[seqlen - t] = torch.sum(entr_terms, dim=2)

        self.init2 = nn.Parameter(torch.zeros(init.shape))
        Z_terms = beta_star[0] + init + self.init2
        Z = logsumexp1(Z_terms)
        Z_logprob = Z_terms - Z.expand(bsz, K)
        Z_prob = Z_logprob.exp()
        entr_terms = Z_prob * (entr_star[0] - Z_logprob)
        entr_Z = torch.sum(entr_terms, dim=1)
        dict_computed['beta'] = beta
        dict_computed['beta_star'] = beta_star
        dict_computed['entr'] = entr
        dict_computed['entr_star'] = entr_star
        dict_computed['Z'] = Z
        dict_computed['entropy'] = entr_Z

        unigram_grad = torch.autograd.grad(Z[1], [self.trans] + len_logprobs_2 + [self.init2, self.emission],
                                           create_graph=True, allow_unused=True)


        entr1 = 0
        for idx, elem in enumerate(len_logprobs):
            entr1 += (unigram_grad[idx + 1] * -elem).sum()

        entr1 += (unigram_grad[0] * -trans_logprobs).sum()

        entr1 += Z[1].sum()

        entr1 += (unigram_grad[-2] * -init).sum()
        entr1 += (unigram_grad[-1] * -fwd_obs_logprobs).sum()


        # opt2 = torch.autograd.grad(entr_Z[0], (self.trans),
        #                            create_graph=True, allow_unused=True)

        # opt1 = torch.autograd.grad(entr1.sum(), (self.trans, self.trans), create_graph=True, allow_unused=True)

        # print('lolllolol')
        # print(grads['trans'])
        return Z, entr_Z













        class RNNLM(nn.Module):
    def __init__(self, options):
        super(RNNLM, self).__init__()

        # vocab = 10000,
        # w_dim = 650,
        # h_dim = 650,
        # num_layers = 2,
        # state_size = 50,
        # dropout = 0.5

        self.h_dim = options.hidden_dim
        self.w_dim = options.embedding_dim
        self.tagset_size = options.tagset_size
        self.num_layers = 1
        self.vocab = options.vocab_size
        self.conditional_dim = options.conditional_dim

        self.word_vecs = nn.Embedding(self.vocab, self.w_dim)
        self.dropout = nn.Dropout(options.dropout)
        self.rnn = nn.LSTM(self.w_dim, self.h_dim, num_layers=self.num_layers,
                           dropout=options.dropout, batch_first=True)
        self.vocab_linear = nn.Linear(self.h_dim + self.conditional_dim, self.vocab)
        # self.vocab_linear.weight = self.word_vecs.weight  # weight sharing
        self.state_pick = nn.Linear(self.h_dim + self.conditional_dim, self.tagset_size)

    def forward(self, condi, sent):
        word_vecs = self.dropout(self.word_vecs(sent[:, :-1]))
        # print(word_vecs.shape)
        bsz, sent_len, _ = word_vecs.shape
        h, _ = self.rnn(word_vecs)
        h = torch.cat([condi.expand(1, sent_len ,self.conditional_dim), h], dim=-1)
        log_prob = F.log_softmax(self.vocab_linear(self.dropout(h)), 2)  # b x l x v
        ll = torch.gather(log_prob, 2, sent[:, 1:].unsqueeze(2)).squeeze(2)
        return ll.sum(1)

    def forward_with_state(self, condi, sent, state):
        ''' In this case, the sent has a BOS and a EOS symbol. '''
        # print(state.shape, sent.shape)
        word_vecs = self.dropout(self.word_vecs(sent[:, :-1])) # bsz * len
        bsz, sent_len, _ = word_vecs.shape
        h, _ = self.rnn(word_vecs)
        h = torch.cat([condi.expand(1, sent_len, self.conditional_dim), h], dim=-1)
        state_choice = nn.LogSoftmax(dim=-1)(self.state_pick(h))
        log_prob = F.log_softmax(self.vocab_linear(self.dropout(h)), 2)  # b x l x v
        # print(log_prob.shape)
        # print(sent.shape)
        # print(state_choice.shape)
        # print(state.shape)
        ll = torch.gather(log_prob, 2, sent[:, 1:].unsqueeze(2)).squeeze(2)
        ll_state = torch.gather(state_choice, 2, state[:, 1:].unsqueeze(2)).squeeze(2)
        return ll.sum(1), ll_state.sum(1)

    def viterbi_state(self, condi, sent):
        word_vecs = self.dropout(self.word_vecs(sent[:, :-1]))  # bsz * len
        bsz, sent_len, _ = word_vecs.shape
        h, _ = self.rnn(word_vecs)
        h = torch.cat([condi.expand(1, sent_len, self.conditional_dim), h], dim=-1)
        state_choice = nn.LogSoftmax(dim=-1)(self.state_pick(h))
        val, arg = torch.max(state_choice, dim=-1)
        return arg

    def generate(self, condi=0, bos=2, eos=3, max_len=45):
        x = []
        # bos = torch.LongTensor(1, 1).cuda().fill_(bos)
        bos = torch.LongTensor(1, 1).fill_(bos)
        emb = self.dropout(self.word_vecs(bos))   # why do we still need drop out at generation / test time?
        prev_h = None
        logp = 0
        for l in range(max_len):
            h, prev_h = self.rnn(emb, prev_h)
            h = torch.cat([condi, h], dim=-1)
            prob = F.softmax(self.vocab_linear(self.dropout(h.squeeze(1))), 1)
            sample = torch.multinomial(prob, 1)
            emb = self.dropout(self.word_vecs(sample))
            x.append(sample.item())
            if x[-1] == eos:
                x.pop()
                break
        return x


    def generate_with_state(self, condi=0, bos=2, eos=3, max_len=45):
        x = []
        state = []
        # bos = torch.LongTensor(1, 1).cuda().fill_(bos)
        bos = torch.LongTensor(1, 1).fill_(bos)
        # print(bos)
        emb = self.dropout(self.word_vecs(bos))   # why do we still need drop out at generation / test time?
        prev_h = None
        logp_w = 0
        logp_s = 0
        for l in range(max_len):
            h, prev_h = self.rnn(emb, prev_h)
            h = torch.cat([condi, h], dim=-1)
            prob = F.softmax(self.vocab_linear(self.dropout(h.squeeze(1))), 1)
            sample = torch.multinomial(prob, 1)
            # print(prob.shape)
            logp_w += torch.log(prob[:, sample])
            # print(prob.shape)
            prob_state = nn.Softmax(dim=-1)(self.state_pick(h).squeeze(1))
            # print(prob_state)
            sample_state = torch.multinomial(prob_state, 1)
            logp_s += torch.log(prob_state[:, sample_state])
            # print(prob_state.shape)
            state.append(sample_state.item())
            emb = self.dropout(self.word_vecs(sample))
            x.append(sample.item())
            if x[-1] == eos:
                x.pop()
                state.pop()
                break

        return x, state, logp_w, logp_s


class RNN_cond_Gen2(nn.Module):
    def __init__(self, opt):
        super(RNN_cond_Gen2, self).__init__()

        self.h_dim = opt.hidden_dim
        self.w_dim = opt.embedding_dim
        self.tagset_size = opt.tagset_size
        self.num_layers = 1
        self.word_vecs = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.dropout = nn.Dropout(opt.dropout)
        self.rnn = nn.LSTM(self.w_dim, self.h_dim, num_layers=self.num_layers,
                           dropout=opt.dropout, batch_first=True)
        self.conditional_dim = opt.conditional_dim
        self.vocab_linear = nn.Linear(self.h_dim + opt.conditional_dim, opt.vocab_size)
        # self.vocab_linear.weight = self.word_vecs.weight  # weight sharing
        self.state_pick = nn.Linear(self.h_dim + opt.conditional_dim, self.tagset_size)

        if opt.decoder == 'crf':
            self.hsmm_crf = HSMM(opt)
        elif opt.decoder == 'gen':
            self.hsmm_crf = HSMM_generative(opt)

    def forward(self, sent, condi_):
        word_vecs = self.dropout(self.word_vecs(sent[:, :-1]))
        h, _ = self.rnn(word_vecs)
        log_prob = F.log_softmax(self.vocab_linear(self.dropout(h)), 2)  # b x l x v
        ll = torch.gather(log_prob, 2, sent[:, 1:].unsqueeze(2)).squeeze(2)
        return ll.sum(1)


    def get_sample_lst_vtb(self, z_gold):
        samples_lst, samples_vtb = [], []
        for elem in z_gold:
            samples_vtb.append([elem])
            samples_lst.append([elem])
        return samples_lst, samples_vtb

    def forward_with_crf2(self, sent, uniqenc, sample_num=3, gold_z = None):
        dict_computed = self.hsmm_crf.get_weights(sent, uniqenc)
        Z, entropy = self.hsmm_crf.get_entr(dict_computed)
        if not gold_z:
            samples_lst, samples_vtb  = self.hsmm_crf.get_sample(dict_computed, sample_num=sample_num)
        else:
            samples_lst, samples_vtb = gold_z

        sample_score = self.hsmm_crf.get_score(samples_lst, dict_computed)
        target = [torch.stack(samples) for samples in sample_score]
        target = torch.stack(target, dim=0)
        bsz, num_sample = target.shape
        state_llq = (target - Z.expand(bsz, num_sample))

        # replace this part with a better code.
        result2 = self.process_state_translate(samples_lst[0])
        word_ll, state_llp = self.forward_with_state( uniqenc, sent, result2)

        # state_llq = torch.stack(self.hsmm_crf.get_score(samples_state, dict_computed))
        # # print(Z.shape, state_llq.shape)
        # state_llq =  state_llq - Z

        result = {"word_ll": word_ll, "state_llp": state_llp,
                  "state_llq": state_llq, "entropy": entropy,
                  "samples_state": samples_lst, 'samples_vtb_style': samples_vtb}

        print('word_ll.shape={}, state_llp.shape={}, state_llq.shape={},'
              ' entr.shape={}, len(samples_state)={}'.format(word_ll.shape, state_llp.shape, state_llq.shape,
                                      entropy.shape, len(samples_lst[0])))

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


    def forward_with_state(self, condi, sent, state):
        # print('LISALISA')
        # print(sent.shape)
        # print(state.shape)


        word_vecs = self.dropout(self.word_vecs(sent[:, :]))
        h, _ = self.rnn(word_vecs)
        bsz, T, _ = h.shape
        h = torch.cat([condi.expand(bsz, T, self.conditional_dim), h], dim=-1)
        state_choice = nn.LogSoftmax(dim=-1)(self.state_pick(h))
        log_prob = F.log_softmax(self.vocab_linear(self.dropout(h)), 2)  # b x l x v

        ll = torch.gather(log_prob, 2, sent[:, :].unsqueeze(2)).squeeze(2)
        # print('investigate: log prob shape is {}, ll shape is {}, sent shape is {}'.format(log_prob.shape, ll.shape,
        #                                                                                    sent.shape))
        # print('investigate: state choice shape is {}, ll_state shape is {}, sent shape is {}'.format(state_choice.shape,
        #                                                                                              'tobe',
        #                                                                                              state.shape))
        ll_state = torch.gather(state_choice, 2, state.long().view(1, -1).unsqueeze(2)).squeeze(2)
        # print(ll.shape)
        return ll.sum(1), ll_state.sum(1)

    def generate(self, condi_, bos=2, eos=3, max_len=45):
        x = []
        # bos = torch.LongTensor(1, 1).cuda().fill_(bos)
        bos = torch.LongTensor(1, 1).fill_(bos)
        emb = self.dropout(self.word_vecs(bos))   # why do we still need drop out at generation / test time?
        prev_h = None
        for l in range(max_len):
            h, prev_h = self.rnn(emb, prev_h)
            prob = F.softmax(self.vocab_linear(self.dropout(h.squeeze(1))), 1)
            sample = torch.multinomial(prob, 1)
            emb = self.dropout(self.word_vecs(sample))
            x.append(sample.item())
            if x[-1] == eos:
                x.pop()
                break
        return x


    def generate_with_state(self, condi_, bos=2, eos=3, max_len=45):
        x = []
        state = []
        # bos = torch.LongTensor(1, 1).cuda().fill_(bos)
        bos = torch.LongTensor(1, 1).fill_(bos)
        # print(bos)
        emb = self.dropout(self.word_vecs(bos))   # why do we still need drop out at generation / test time?
        prev_h = None
        for l in range(max_len):
            h, prev_h = self.rnn(emb, prev_h)
            prob = F.softmax(self.vocab_linear(self.dropout(h.squeeze(1))), 1)
            sample = torch.multinomial(prob, 1)
            # print(prob.shape)
            prob_state = nn.Softmax(dim=-1)(self.state_pick(h).squeeze(1))
            # print(prob_state)
            sample_state = torch.multinomial(prob_state, 1)
            # print(prob_state.shape)
            state.append(sample_state.item())
            emb = self.dropout(self.word_vecs(sample))
            x.append(sample.item())
            if x[-1] == eos:
                x.pop()
                break
        return x, state



    
class Linear_CRF(nn.Module):

    def __init__(self, embedding_dim=100, hidden_dim=100, vocab_size=1000,
                 tag_to_ix={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}, START_TAG=0, STOP_TAG=1):
        super(Linear_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.START_TAG = START_TAG
        self.STOP_TAG = STOP_TAG

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        collect_alpha_t = []
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
            collect_alpha_t.append(forward_var.data)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        Z = log_sum_exp(terminal_var)
        collect_alpha_t[-1] = terminal_var.data
        # print(collect_alpha_t)
        return Z, collect_alpha_t

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sent):
        feats = self._get_lstm_features(sent)
        forward_score = self._forward_alg(feats)
        return forward_score

    def sample(self, collect_alpha_t):
        sample_lst = []
        for tagset_t in reversed(collect_alpha_t):
            tagset_z = log_sum_exp(tagset_t)
            log_prob = tagset_t - tagset_z
            # print(log_prob.exp())
            sample = torch.multinomial(log_prob.exp(), 1)
            sample_lst.append(sample.item())
        # print(sample_lst)
        return sample_lst

    def entr(self, collect_alpha_t):
        tag_dim = 7 #collect_alpha_t[0].size()
        entr = [[0] * tag_dim ]  # n * tagset
        # iterate through sentence
        for tagset_t in reversed(collect_alpha_t):
            new_entr = [0] * tag_dim
            tagset_z = log_sum_exp(tagset_t)
            log_prob = tagset_t - tagset_z
            prob = log_prob.exp()
            # iterate through tags
            for idx, elem in enumerate(prob.view(-1)):
                # print(idx)
                # print(elem * (entr[-1][idx] - np.log(elem)))
                new_entr[idx] = (elem * (entr[-1][idx] - np.log(elem))).item()
            # print(new_entr)
            entr.append(new_entr)
        # print(sum(entr[-1][2:]))
        return sum(entr[-1][2:])

    def _compute_beta(self):
        pass

    def _marginals(self, collect_alpha_t, collect_beta_t):

        pass

      def forward1(self, condi, sent):
        word_vecs = self.dropout(self.word_vecs(sent[:, :-1]))
        h, _ = self.rnn(word_vecs)
        h = torch.cat([condi, h])
        log_prob = F.log_softmax(self.vocab_linear(self.dropout(h)), 2)  # b x l x v
        ll = torch.gather(log_prob, 2, sent[:, 1:].unsqueeze(2)).squeeze(2)
        return ll.sum(1)

    def viterbi_state2(self, condi, sent):
        word_vecs = self.dropout(self.word_vecs(sent[:, :-1]))  # bsz * len
        bsz, sent_len, _ = word_vecs.shape
        h, _ = self.rnn(word_vecs)
        h = torch.cat([condi.expand(1, sent_len, self.conditional_dim), h], dim=-1)
        state_choice = nn.LogSoftmax(dim=-1)(self.state_pick(h))
        val, arg = torch.max(state_choice, dim=-1)
        return arg
        pass