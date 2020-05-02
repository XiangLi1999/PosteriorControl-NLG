args.optim_algo == 'lagging':
                full_lagging_opt(model, {"full":optimizer_full, 'p':optimizer_p, 'q':optimizer_q}, args, data_lst)

            elif args.optim_algo == 'simple':
                ''' carry out training'''
                flag_supervised = True
                flag_supervised_training = True if args.supervised_training == 'yes' else False # only True in special case.
                args.kl_pen = 0.1
                # args.delta_kl = 1 / (len(data_lst) / 10)
                args.lamb = args.lamb_init
                for e in range(args.epoch):
                    summary = defaultdict(float)
                    whole_obj_lst = []
                    for idx, item in enumerate(data_lst):
                        obj = gather_stats( model, item, summary, args.kl_pen )
                        whole_obj_lst.append(obj)

                        if idx % 10 == 0:
                            target = torch.stack(whole_obj_lst).mean()
                            target.backward()

                            if args.group_param == 'complicated':
                                if args.max_grad_norm > 0:
                                    torch.nn.utils.clip_grad_norm_(model_params + action_params, args.max_grad_norm)
                                if args.q_max_grad_norm > 0:
                                    torch.nn.utils.clip_grad_norm_(q_params, args.q_max_grad_norm)


                                q_optimizer.step()
                                optimizer.step()
                                action_optimizer.step()

                                q_optimizer.zero_grad()
                                optimizer.zero_grad()
                                action_optimizer.zero_grad()
                            elif args.group_param == 'variational':
                                if args.max_grad_norm > 0:
                                    torch.nn.utils.clip_grad_norm_(model.rnn_lm.parameters(), args.max_grad_norm)
                                if args.q_max_grad_norm > 0:
                                    torch.nn.utils.clip_grad_norm_(model.hsmm_crf.parameters(), args.q_max_grad_norm)

                                optimizer_p.step()
                                optimizer_q.step()
                                optimizer_w.step()

                                optimizer_p.zero_grad()
                                optimizer_q.zero_grad()
                                optimizer_w.zero_grad()

                                args.delta_kl = 1 / (len(data_lst) / args.batch_size)

                            else:
                                optimizer.step()
                                optimizer.zero_grad()

                            whole_obj_lst = []
                    args.kl_pen = min(args.kl_pen * 2, 1)
                    args.lamb = min(args.lamb * 2, args.lamb_max)
                    print(print_dict(summary))
                    investigate_viterb(model, data_lst)






    def forward_mi(self, condi, sent, sample_lst):
        '''
        This function computes a forward path and the regularization term.
        :param condi:
        :param sent:
        :param sample_lst:
        :return:
        '''
        logp_word_all = []
        logp_z_all = []
        for bsz_idx in range(len(sample_lst)):
            logp_word_example = []
            logp_z_example = []
            for samp_idx in range(len(sample_lst[bsz_idx])):
                start_lst, end_lst, state_lst = sample_lst[bsz_idx][samp_idx]
                state_lst = torch.LongTensor(state_lst).unsqueeze(0)
                word_vecs = self.dropout(self.word_vecs(sent[:, :]))
                state_vecs = self.dropout(self.state_vecs(state_lst))
                bsz, _, cond_dim = condi.shape
                h_prev = torch.zeros(1, bsz, self.h_dim)
                h = torch.zeros(bsz, 1, self.h_dim + self.conditional_dim)
                c_prev = torch.zeros(1, bsz, self.h_dim)
                logp_word = 0
                logp_z = 0
                state_lst = state_lst.squeeze(0)
                mi_total_hz = 0
                mi_total_hx = 0

                for segment_idx, (start, end, state_name) in enumerate(zip(start_lst, end_lst, state_lst)):
                    # decide to switch to a new state
                    temp = h[:, -1:]
                    logp_z += F.logsigmoid(self.action_pick(temp)).sum()
                    # generate that particular state.
                    if segment_idx == 0:
                        temp_state_pick = torch.cat([temp, self.start_vecs], dim=-1)
                        reg_state_pick = self.start_vecs
                    else:
                        temp_state_pick = torch.cat([temp, state_vecs[:, segment_idx - 1].unsqueeze(0)], dim=-1)
                        ''' regularized state space selection. '''
                        reg_state_pick =   state_vecs[:, segment_idx - 1].unsqueeze(0)

                    reg_state_logp = nn.LogSoftmax(dim=-1)(self.state_pick_reg(reg_state_pick)) # p(z|z_-1)
                    state_logp = nn.LogSoftmax(dim=-1)(self.state_pick(temp_state_pick)) # p(z|z_-1, h_prev)
                    logp_z += state_logp[:, :, state_name].mean()
                    mi_hz = get_kl_log(state_logp, reg_state_logp)
                    mi_total_hz += mi_hz

                    # concatenate the current state to the input of the RNN.
                    inp_rnn = torch.cat([word_vecs[:, start:end, :],
                                         state_vecs[:, segment_idx].unsqueeze(1).expand(bsz, end - start, self.s_dim)],
                                        dim=-1)
                    h, (h_prev, c_prev) = self.rnn(inp_rnn, (h_prev, c_prev))

                    # the attention step, here we just completely focus on the h.
                    h = torch.cat([h, condi.expand(bsz, end - start, self.conditional_dim)], dim=-1)
                    if end - start > 1:
                        temp = h[:, :-1]
                        logp_z += torch.log(1 - F.sigmoid(self.action_pick(temp))).sum()
                    # print(h.shape, state_vecs[:, segment_idx].unsqueeze(1).shape)
                    temp_vocab_pick = torch.cat([h, state_vecs[:, segment_idx].unsqueeze(1).expand(bsz, end-start, self.w_dim)], dim=-1)
                    # reg_vocab_pick = state_vecs[:, segment_idx].unsqueeze(1).expand(bsz, end-start, self.w_dim)
                    # print(temp_vocab_pick.shape, reg_vocab_pick.shape)

                    log_prob = F.log_softmax(self.vocab_pick(temp_vocab_pick), 2)  # b x l x v
                    if not self.one_rnn:
                        # append some symbol to first generate the first token.
                        word_feats = torch.cat([self.bos_emb, word_vecs[:, start:end, :]], dim=1)
                        reg_vocab_pick, _ =  self.seg_rnns[state_name](word_feats, None)
                        # reg_vocab_pick, _ =  self.seg_rnns[state_name](word_vecs[:, start:end, :], None)
                        reg_log_prob = F.log_softmax(self.vocab_pick_reg(reg_vocab_pick), 2)
                    else:
                        reg_vocab_pick, _ =  self.seg_rnns[0](inp_rnn, None)
                        reg_log_prob = F.log_softmax(self.vocab_pick_reg(reg_vocab_pick), 2)
                    # print(log_prob.shape, reg_log_prob.shape)
                    # mi_hx = get_kl_log(log_prob, reg_log_prob)
                    mi_hx = 0
                    mi_total_hx += mi_hx
                    # logp_word += torch.gather(log_prob, 2, sent[:, start:end].unsqueeze(2)).squeeze(2).sum(1).mean()
                    # predict all but the EOS symbol.
                    single_start2end_wp = torch.gather(reg_log_prob[:,:-1], 2, sent[:, start:end].unsqueeze(2)).squeeze(2).sum(1).mean()
                    # predict the EOS symbol
                    single_start2end_wp += reg_log_prob[:,-1][:, self.eos].mean()

                    #
                    # print('*'*10)
                    # print(start, end)
                    # print(state_name)
                    # print(sent[:, start:end])
                    # print(reg_log_prob)

                    logp_word += single_start2end_wp

                logp_word_example.append(logp_word)
                logp_z_example.append(logp_z)
            logp_word_all.append(torch.stack(logp_word_example))
            logp_z_all.append(torch.stack(logp_z_example))
        # print('done')
        return torch.stack(logp_word_all, dim=0), torch.stack(logp_z_all, dim=0), (mi_total_hz, mi_total_hx)



    def forward_with_state2(self, condi, sent, sample_lst):
        '''
        Note that in training with logP, don't have to train on the  first BOS, but always need to train on the last
        EOS symbol.
        Assume that sent has BOS and EOS symbol.
        :param condi:
        :param sent:
        :param state: The state encoding is a pair (start, end, state_name), where start is inclusive, but
        the end is not inclusive.
        Or more precise, the state has (start, end, state_name)
        :return:
        '''
        logp_word_all = []
        logp_z_all = []
        for bsz_idx in range(len(sample_lst)):
            logp_word_example = []
            logp_z_example = []
            for samp_idx in range(len(sample_lst[bsz_idx])):
                start_lst, end_lst, state_lst = sample_lst[bsz_idx][samp_idx]
                state_lst = torch.LongTensor(state_lst).unsqueeze(0)
                word_vecs = self.dropout(self.word_vecs(sent[:, :]))
                state_vecs = self.dropout(self.state_vecs(state_lst))
                bsz, _, cond_dim = condi.shape
                h_prev = torch.zeros(1, bsz, self.h_dim)
                h = torch.zeros(bsz, 1, self.h_dim + self.conditional_dim)
                c_prev = torch.zeros(1, bsz, self.h_dim)
                logp_word = 0
                logp_z = 0
                state_lst = state_lst.squeeze(0)
                mi_total_hz = 0
                mi_total_hx = 0

                for segment_idx, (start, end, state_name) in enumerate(zip(start_lst, end_lst, state_lst)):
                    # decide to switch to a new state
                    temp = h[:, -1:]
                    logp_z += F.logsigmoid(self.action_pick(temp)).sum()
                    # generate that particular state.
                    if segment_idx == 0:
                        temp_state_pick = torch.cat([temp, self.start_vecs], dim=-1)
                        reg_state_pick = self.start_vecs
                    else:
                        temp_state_pick = torch.cat([temp, state_vecs[:, segment_idx - 1].unsqueeze(0)], dim=-1)
                        ''' regularized state space selection. '''
                        reg_state_pick =   state_vecs[:, segment_idx - 1].unsqueeze(0)
                    # print(reg_state_pick.shape)
                    # print(temp_state_pick.shape)

                    reg_state_logp = nn.LogSoftmax(dim=-1)(self.state_pick_reg(reg_state_pick)) # p(z|z_-1)
                    state_logp = nn.LogSoftmax(dim=-1)(self.state_pick(temp_state_pick)) # p(z|z_-1, h_prev)
                    logp_z += state_logp[:, :, state_name].mean()
                    mi_hz = get_kl_log(state_logp, reg_state_logp)
                    # print(mi)
                    mi_total_hz += mi_hz
                    # concatenate the current state to the input of the RNN.
                    inp_rnn = torch.cat([word_vecs[:, start:end, :],
                                         state_vecs[:, segment_idx].unsqueeze(1).expand(bsz, end - start, self.s_dim)],
                                        dim=-1)
                    h, (h_prev, c_prev) = self.rnn(inp_rnn, (h_prev, c_prev))
                    # the attention step, here we just completely focus on the h.
                    h = torch.cat([h, condi.expand(bsz, end - start, self.conditional_dim)], dim=-1)
                    # print(h.shape)
                    if end - start > 1:
                        temp = h[:, :-1]
                        logp_z += torch.log(1 - F.sigmoid(self.action_pick(temp))).sum()
                    # print(h.shape, state_vecs[:, segment_idx].unsqueeze(1).shape)
                    temp_vocab_pick = torch.cat([h, state_vecs[:, segment_idx].unsqueeze(1).expand(bsz, end-start, self.w_dim)], dim=-1)
                    reg_vocab_pick = state_vecs[:, segment_idx].unsqueeze(1).expand(bsz, end-start, self.w_dim)
                    # print(temp_vocab_pick.shape, reg_vocab_pick.shape)

                    log_prob = F.log_softmax(self.vocab_pick(temp_vocab_pick), 2)  # b x l x v
                    reg_log_prob =  F.log_softmax(self.vocab_pick_reg(reg_vocab_pick), 2)
                    # print(log_prob.shape, reg_log_prob.shape)
                    mi_hx = get_kl_log(log_prob, reg_log_prob)
                    # mi_hx = 0
                    mi_total_hx += mi_hx
                    logp_word += torch.gather(log_prob, 2, sent[:, start:end].unsqueeze(2)).squeeze(2).sum(1).mean()
                    # logp_word += torch.gather(reg_log_prob, 2, sent[:, start:end].unsqueeze(2)).squeeze(2).sum(1).mean()
                logp_word_example.append(logp_word)
                logp_z_example.append(logp_z)
            logp_word_all.append(torch.stack(logp_word_example))
            logp_z_all.append(torch.stack(logp_z_example))

        return torch.stack(logp_word_all, dim=0), torch.stack(logp_z_all, dim=0)



    def forward_with_state(self, condi, sent, state):
        word_vecs = self.dropout(self.word_vecs(sent[:, :-1]))
        h, _ = self.rnn(word_vecs)
        h = torch.cat([condi, h])
        state_choice = nn.LogSoftmax(dim=-1)(self.state_pick(h))
        log_prob = F.log_softmax(self.vocab_linear(self.dropout(h)), 2)  # b x l x v

        ll = torch.gather(log_prob, 2, sent[:, 1:].unsqueeze(2)).squeeze(2)
        ll_state = torch.gather(state_choice, 2, state[:, 1:].unsqueeze(2)).squeeze(2)
        return ll.sum(1) + ll_state.sum(1)

    def generate(self, condi=0, bos=2, eos=3, max_len=45):
        x = []
        # bos = torch.LongTensor(1, 1).cuda().fill_(bos)
        bos = torch.LongTensor(1, 1).fill_(bos)
        emb = self.dropout(self.word_vecs(bos))   # why do we still need drop out at generation / test time?
        prev_h = None
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


    def generate_with_state2(self, condi, max_len=45):
        x = []
        state = []
        start_lst = []
        end_lst = []
        bos = torch.LongTensor(1, 1).fill_(self.bos)
        bsz, cond_dim = condi.shape
        condi = condi.unsqueeze(1)
        h_prev = torch.zeros(1, bsz, self.h_dim)
        c_prev = torch.zeros(1, bsz, self.h_dim)
        logp = 0
        emb = self.word_vecs(bos)   # why do we still need drop out at generation / test time?
        x.append(self.bos)
        state.append(self.bos_s)
        for l in range(max_len):
            if l == 0:
                sample_action = 1
                prob_action = torch.tensor([1.])
                temp = torch.cat([h_prev.transpose(0, 1), condi], dim=-1)
            else:
                temp = torch.cat([h_prev.transpose(0, 1), condi], dim=-1)
                prob_action = F.sigmoid(self.action_pick(temp))
                sample_action = torch.bernoulli(prob_action).item()
            if sample_action == 1:
                # valid start of a new segment.
                if l != 0:
                    end_lst.append(l)
                start_lst.append(l)
                logp += torch.log(prob_action).item()
                prob_state = nn.Softmax(dim=-1)(self.state_pick(temp).squeeze(1))
                sample_state = torch.multinomial(prob_state, 1)
                state_vecs = self.state_vecs(sample_state)
                state.append(sample_state.item())
                logp += torch.log(prob_state[:,sample_state.item()])

                # generate a word token distribution, conditioned on the state and the prev token.
                inp_rnn = torch.cat([emb,
                                     state_vecs],
                                    dim=-1)
                h, (h_prev, c_prev) = self.rnn(inp_rnn, (h_prev, c_prev))

                # the attention step, here we just completely focus on the h.
                h = torch.cat([condi.expand(bsz, 1, self.conditional_dim), h], dim=-1)
                prob_token = F.softmax(self.vocab_pick(self.dropout(h)), 2).squeeze(1)  # b x l x v
                sample_token = torch.multinomial(prob_token, 1)
                emb = self.word_vecs(sample_token)
                x.append(sample_token.item())
                logp += torch.log(prob_token[:, sample_token.item()]).item()


            else:
                logp += torch.log( 1- prob_action).item()

                inp_rnn = torch.cat([emb,
                                     state_vecs],
                                    dim=-1)
                h, (h_prev, c_prev) = self.rnn(inp_rnn, (h_prev, c_prev))

                # the attention step, here we just completely focus on the h.
                h = torch.cat([condi.expand(bsz, 1, self.conditional_dim), h], dim=-1)
                prob_token = F.softmax(self.vocab_pick(self.dropout(h)), 2).squeeze(1)  # b x l x v
                sample_token = torch.multinomial(prob_token, 1)
                emb = self.word_vecs(sample_token)
                x.append(sample_token.item())
                logp += torch.log(prob_token[:, sample_token.item()]).item()
            if x[-1] == self.eos:
                break

        end_lst.append(len(x) - 1)
        print(x)
        print(state)
        print(start_lst)
        print(end_lst)
        print(len(start_lst), len(end_lst))
        return x, start_lst, end_lst, state


if __name__ == '__main__':

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

    parser.add_argument('-initrange', type=float, default=0.05, help='uniform init interval')
    parser.add_argument('-lr_decay', type=float, default=0.05, help='learning rate decay')
    parser.add_argument('-optim', type=str, default="sgd", help='optimization algorithm')
    parser.add_argument('-onmt_decay', action='store_true', help='')
    parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('-interactive', action='store_true', help='')
    parser.add_argument('-label_train', action='store_true', help='')
    parser.add_argument('-gen_from_fi', type=str, default='', help='')
    parser.add_argument('-verbose', action='store_true', help='')
    parser.add_argument('-prev_loss', type=float, default=None, help='')
    parser.add_argument('-best_loss', type=float, default=None, help='')


    parser.add_argument('-embedding_dim', type=int, default=50, help='')
    parser.add_argument('-hidden_dim', type=int, default=50, help='')
    parser.add_argument('-vocab_size', type=int, default=1000, help='')
    parser.add_argument('-tagset_size', type=int, default=10, help='')
    parser.add_argument('-q_dim', type=int, default=50, help='')
    parser.add_argument('-q_lr', type=float, default=0.02, help='')
    parser.add_argument('-action_lr', type=float, default=0.02, help='')
    parser.add_argument('-lr', type=float, default=0.02, help='initial learning rate')
    parser.add_argument('-conditional_dim', type=int, default=30, help='')
    parser.add_argument('-train_q_epochs', type=int, default=30, help='')
    parser.add_argument('--max_grad_norm', default=5, type=float, help='gradient clipping parameter')
    parser.add_argument('--q_max_grad_norm', default=1, type=float, help='gradient clipping parameter for q')

    args = parser.parse_args()
    model = RNNLM(args)


    condi = torch.randn(1, 30)
    sent = torch.LongTensor(1, 25).random_(20, 50)
    # sent = torch.randn(5, 25)
    start_lst = [0, 5, 10, 15, 22]
    end_lst = [5, 10, 15, 22, 25]
    state_lst = torch.LongTensor(1, 5).random_(0, 10)
    # state_lst = torch.LongTensor([1, 2, 3, 4, 5])

    # model.generate_with_state(condi)
    model.forward_with_state2(condi, sent, start_lst, end_lst, state_lst)
    x, start_lst, end_lst, state_lst = model.generate_with_state2(condi)

# have people tried to train the encoder jointly with the decoder on the objective of p(x)? Then assume that the signal
# from p(x) can kind of aid the latent space.

