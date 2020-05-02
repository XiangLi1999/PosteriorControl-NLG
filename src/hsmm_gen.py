import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import cuda
import sys
from utils import *
from allennlp.modules.elmo import Elmo, batch_to_ids
import os
neginf = -1e38


class HSMM_generative(nn.Module):
    def __init__(self, args):
        '''
        initialization step.
        '''
        # embedding_dim, hidden_dim, vocab_size, tag_set_size,  q_dim=20, dropout=0,
        super(HSMM_generative, self).__init__()

        # Dimension setting.
        self.device = args.device
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.vocab_size = args.vocab_size
        self.tagset_size = args.tagset_size
        self.table_dim = args.table_dim
        self.table_hidden_dim = args.table_hidden_dim
        self.K = self.tagset_size
        self.gen_vocab_size = args.gen_size
        self.L = args.L
        self.cond_A_dim = args.A_dim
        self.unif_lenps = True #args.unif_lenps # set True

        self.dropout = nn.Dropout(args.dropout)
        self.Kmul = 1
        self.smaller_cond_dim = self.tagset_size * self.Kmul * self.cond_A_dim

        # self.pad_idx = args.pad_idx
        # self.word_embeds = word_embeds
        self.non_field = args.non_field
        # self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=args.pad_idx)

        ###############   real dataset encoding  ##############
        self.max_pool = args.max_pool
        self.src_bias = nn.Parameter(torch.randn(1, self.embedding_dim))
        self.uniq_bias = nn.Parameter(torch.randn(1, self.embedding_dim))

        ###############   transition scores  ##############
        self.trans_unif = (args.trans_unif == 'yes')
        if not self.trans_unif:
            self.get_trans = self.get_transition_score
            self.init_lin = nn.Linear(self.table_hidden_dim*2, self.tagset_size * self.Kmul)
            self.A_from = nn.Parameter(torch.randn(self.tagset_size * self.Kmul, self.cond_A_dim))
            self.A_to = nn.Parameter(torch.randn(self.cond_A_dim, self.tagset_size * self.Kmul))
            self.cond_trans_lin = nn.Sequential(
                nn.Linear(self.table_hidden_dim*2, self.smaller_cond_dim),
                nn.ReLU(),
                nn.Linear(self.smaller_cond_dim, self.tagset_size * self.Kmul * self.cond_A_dim * 2))
        else:
            self.get_trans = self.get_trans_unif


        ###############   Length scores ##############
        if self.unif_lenps:
            self.len_scores = nn.Parameter(torch.ones(1, self.L))
            self.len_scores.requires_grad = False

        self.yes_self_trans = False
        if not self.yes_self_trans:
            selfmask = torch.Tensor(self.Kmul * self.tagset_size).fill_(neginf)
            self.register_buffer('selfmask', torch.diag(selfmask))
            # self.register_buffer('selfmask', Variable(torch.diag(selfmask), requires_grad=False))

        ##############  Emission Scores  ##############
        self.mlpinp = 2
        self.use_word_only = False
        if self.mlpinp == 1:
            inp_feats = 4
            mlpinp_sz = inp_feats * self.embedding_dim
            rnninsz = self.embedding_dim
            self.inpmlp = nn.Sequential(nn.Linear(mlpinp_sz, rnninsz),
                                        nn.ReLU())
        elif self.mlpinp == 2:
            if self.use_word_only:
                rnninsz = self.w_dim
            else:
                rnninsz = self.table_dim

        self.decoder_type = args.decoder
        self.layers = 1
        print(self.decoder_type)
        if args.decoder == 'gen':
            self.h0_lin = nn.Linear(self.embedding_dim, 2 * self.hidden_dim * self.layers)
            self.state_att_gates = nn.Parameter(torch.randn(self.tagset_size, 1, 1, self.hidden_dim))
            self.state_att_biases = nn.Parameter(torch.randn(self.tagset_size, 1, 1, self.hidden_dim))
            self.sep_attn = args.sep_attn
            if self.sep_attn:
                self.state_att2_gates = nn.Parameter(torch.randn(self.tagset_size, 1, 1, self.hidden_dim ))
                self.state_att2_biases = nn.Parameter(torch.randn(self.tagset_size, 1, 1,self.hidden_dim ))
            out_hid_sz = self.hidden_dim * 2
            self.state_out_gates = nn.Parameter(torch.randn(self.tagset_size, 1, 1, out_hid_sz))
            self.state_out_biases = nn.Parameter(torch.randn(self.tagset_size, 1, 1, out_hid_sz))
            self.decoder = nn.Linear(out_hid_sz, self.gen_vocab_size + 1) # the last element in this set is the end of phrase index.
            self.zeros = torch.Tensor(1, 1).fill_(neginf).to(args.device)
            self.eop_idx = self.gen_vocab_size

            self.one_rnn = args.one_rnn
            self.seg_rnns = nn.ModuleList()
            if args.one_rnn:
                self.seg_rnns.append(nn.LSTM(self.embedding_dim + self.embedding_dim, self.hidden_dim, self.layers, dropout=args.dropout))
                self.state_embs = nn.Parameter(torch.randn(self.tagset_size, 1, 1, self.embedding_dim))
            else:
                for _ in range(args.tagset_size):
                    self.seg_rnns.append(nn.LSTM(self.embedding_dim, self.hidden_dim, self.layers, dropout=args.dropout))
            self.emb_drop = True
            self.start_emb = nn.Parameter(torch.randn(1, 1, rnninsz))
            self.pad_emb = nn.Parameter(torch.zeros(1, 1, rnninsz))

        # only for the CRF version.
        elif args.decoder == 'crf':
            # self.attn_lin = nn.Linear(2*self.hidden_dim, self.hidden_dim)
            self.use_bert = (args.use_bert == 'yes')
            self.use_elmo = (args.use_elmo == 'yes')

            if self.use_bert:
                # use a bert model to score spans.
                self.init_bert()
                self.get_span_scores = self.get_bert_span_scores
            elif self.use_elmo:
                self.style = args.elmo_style
                print('use ELMo {}, style={}'.format(self.use_elmo, self.style))
                sys.stdout.flush()
                self.init_elmo()
                self.elmo_dim = 1024
                self.elmo_dim2 = 1024

                if self.style == 1:
                    mlpinp_sz = 3 * self.embedding_dim + self.elmo_dim
                    self.inpmlp = nn.Sequential(nn.Linear(mlpinp_sz, self.elmo_dim2),
                                                nn.ReLU())
                    bos_emb_size = self.embedding_dim * 3
                    self.bos_emb = nn.Parameter(torch.randn(1, 1, bos_emb_size))
                    self.eos_emb = nn.Parameter(torch.randn(1, 1, bos_emb_size))

                # self.h0_lin = nn.Linear(self.embedding_dim, 2 * self.hidden_dim * self.layers)
                # self.zeros = torch.Tensor(1, 1).fill_(neginf).to(args.device)
                # self.eop_idx = self.gen_vocab_size
                # self.one_attn_gates = True
                # self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                #                     num_layers=self.layers, bidirectional=True)

            else: # use a bi-lstm minus to score.
                self.h0_lin = nn.Linear(self.table_hidden_dim * 2, 2 * self.hidden_dim * self.layers)
                self.zeros = torch.Tensor(1, 1).fill_(neginf).to(args.device)
                self.eop_idx = self.gen_vocab_size
                self.bos_emb = nn.Parameter(torch.randn(1, 1, rnninsz))
                self.eos_emb = nn.Parameter(torch.randn(1, 1, rnninsz))
                self.lstm = nn.LSTM(rnninsz, self.hidden_dim,
                                    num_layers=self.layers, bidirectional=True)
                self.get_span_scores = self.get_lstm_span_scores
            # self.q_binary = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2), nn.ReLU(),
            #                               nn.Dropout(args.dropout), nn.Linear(self.hidden_dim * 2, self.tagset_size))
            self.additional_attn = (args.additional_attn == 'yes')
            if self.additional_attn:
                # style = 'Peter_Span'
                style = args.span_repr

                if self.use_elmo and style == 'peter':
                    print('using peter style of span representation.')
                    self.get_span_scores = self.get_elmo_span_scores1
                    self.attn_layer = self.attn_layer3
                    self.state_att_gates = nn.Linear(self.elmo_dim2, self.embedding_dim)
                    self.state_att_biases = nn.Linear(self.elmo_dim2, self.embedding_dim)

                    temp_ = self.elmo_dim2 + self.hidden_dim
                    self.state_out_gates = nn.Parameter(torch.randn(1, 1, temp_))
                    self.state_out_biases = nn.Parameter(torch.randn(1, 1, temp_))

                    temp_2 = temp_*4
                    self.q_binary = nn.Sequential(nn.Linear(temp_2, temp_),
                                                  nn.Tanh(), self.dropout,
                                                  nn.Linear(temp_ , self.tagset_size))
                elif self.use_elmo:
                    print('using BiLM-minus of span representation.')
                    self.get_span_scores = self.get_elmo_span_scores2
                    self.attn_layer = self.attn_layer3
                    self.state_att_gates = nn.Linear(self.elmo_dim2 // 2, self.embedding_dim)
                    self.state_att_biases = nn.Linear(self.elmo_dim2 // 2, self.embedding_dim)

                    temp_ = self.elmo_dim2 // 2 + self.hidden_dim
                    self.state_out_gates = nn.Parameter(torch.randn(1, 1, temp_))
                    self.state_out_biases = nn.Parameter(torch.randn(1, 1, temp_))

                    temp_2 = temp_ * 2
                    self.q_binary = nn.Sequential(nn.Linear(temp_2, temp_),
                                                  nn.Tanh(), self.dropout,
                                                  nn.Linear(temp_, self.tagset_size))
                else:
                    print('not using ELMo, still using the attention in the encoder.')
                    self.attn_layer = self.attn_layer2
                    self.state_att_gates = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
                    self.state_att_biases = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
                    out_hid_sz = self.hidden_dim * 2
                    self.state_out_gates = nn.Parameter(torch.randn(1, 1, out_hid_sz))
                    self.state_out_biases = nn.Parameter(torch.randn(1, 1, out_hid_sz))

                    self.q_binary = nn.Sequential(nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
                                                  nn.Tanh(), self.dropout, nn.Linear(self.hidden_dim * 2, self.tagset_size))

                    ''' novel writing of attention ... '''
                    # TODO: unsure about this writing of attention.
            else:
                if self.use_elmo:
                    print('use ELMo, but do not use additional attention in the encoder. ')
                    self.get_span_scores = self.get_elmo_span_scores3
                    temp_ = self.elmo_dim2
                    self.q_binary = nn.Sequential(nn.Linear(temp_, temp_ // 2),
                                                  nn.Tanh(), self.dropout,
                                                  nn.Linear(temp_ // 2, self.tagset_size))
                else:
                    print('NOT use ELMo, and not use additional attention in the encoder. ')
                    self.q_binary = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
                                                  nn.Tanh(), self.dropout, nn.Linear(self.hidden_dim * 2, self.tagset_size))

        ##############   finally  ##############
        self.ones_ = torch.ones(1,1).long().to(self.device)

        self.labeled_states = len(args.labeled_states)
        self.labeled_state_dict = args.field_idx2state_idx

        self.posterior_reg = args.posterior_reg
        self.hard_code = ( args.hard_code == 'yes')

        if args.posterior_reg == 1:
            if args.pr_reg_style == 'swap':
                self.posterior_reg_term = self.posterior_reg_term_swapStyled
            elif args.pr_reg_style == 'phrase':
                self.posterior_reg_term = self.posterior_reg_term_phraseStyled

            elif args.pr_reg_style == 'soft':
                self.posterior_reg_term = self.posterior_reg_term_soft
            elif args.pr_reg_style == 'wb:entr':
                self.posterior_reg_term = self.posterior_reg_term_wbEntr
            elif args.pr_reg_style == 'wb:soft':
                self.posterior_reg_term = self.posterior_reg_term_wbSoft
            elif args.pr_reg_style == 'wb:cluster':
                self.posterior_reg_term = self.posterior_reg_term_wbCluster
            elif args.pr_reg_style == 'wb:global':
                self.ff_lookup = nn.Linear(args.f_dim, self.tagset_size)
                self.posterior_reg_term = self.posterior_reg_term_global
            elif args.pr_reg_style == 'wb:hard':
                pass
            else:
                print('unknow posterior reg style')
                sys.stdout.flush()
        else:
            print('no posterior reg')
            sys.stdout.flush()
            pass



    def init_bert(self):
        self.bert_tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased',
                                   do_basic_tokenize=False)
        self.bert_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertModel', 'bert-base-cased')
        self.bert_model.eval()

    def init_elmo(self):
        system_config = '/home-4/xli150@jhu.edu/mutualinfo/others/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        system_weight = '/home-4/xli150@jhu.edu/mutualinfo/others/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
        config_isfile = os.path.isfile(system_config)
        weight_isfile = os.path.isfile(system_weight)
        options_file = system_config if config_isfile else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = system_weight if weight_isfile else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo_model = Elmo(options_file, weight_file, 1, dropout=0, requires_grad=False).to(self.device)


    def obs_logprobs(self, x, srcenc, srcfieldenc, fieldmask, combotargs):
        """
        args:
          x - seqlen x bsz x max_locs x nfeats
          srcenc - bsz x emb_size
          srcfieldenc - bsz x nfields x dim
          fieldmask - bsz x nfields mask with 0s and -infs where it's a dummy field
          combotargs - L x bsz*seqlen x max_locs
        returns:
          a L x seqlen x bsz x K tensor, where l'th row has prob of sequences of length l+1.
          specifically, obs_logprobs[:,t,i,k] gives p(x_t|k), p(x_{t:t+1}|k), ..., p(x_{t:t+l}|k).
          the infc code ignores the entries rows corresponding to x_{t:t+m} where t+m > T
        """
        seqlen, bsz, maxlocs, nfeats = x.size()
        embs = self.word_embeds(x.view(seqlen, -1)) # seqlen x bsz*maxlocs*nfeats x emb_size

        if self.mlpinp:
            inpembs = self.inpmlp(embs.view(seqlen, bsz, maxlocs, -1)).mean(2)
        else:
            inpembs = embs.view(seqlen, bsz, maxlocs, -1).mean(2) # seqlen x bsz x nfeats*emb_size
        #
        # if self.emb_drop:
        #     inpembs = self.drop(inpembs)

        # if self.ar:
        #     if self.word_ar:
        #         ar_embs = embs.view(seqlen, bsz, maxlocs, nfeats, -1)[:, :, 0, 0] # seqlen x bsz x embsz
        #     else: # ar on fields
        #         ar_embs = embs.view(seqlen, bsz, maxlocs, nfeats, -1)[:, :, :, 1].mean(2) # same
        #     if self.emb_drop:
        #         ar_embs = self.drop(ar_embs)
        #
        #     # add on initial <bos> thing; this is a HACK!
        #     embsz = ar_embs.size(2)
        #     ar_embs = torch.cat([self.lut.weight[2].view(1, 1, embsz).expand(1, bsz, embsz),
        #                             ar_embs], 0) # seqlen+1 x bsz x emb_size
        #     ar_states, _ = self.ar_rnn(ar_embs) # seqlen+1 x bsz x rnn_size

        # get L+1 x bsz*seqlen x emb_size segembs
        segembs = self.to_seg_embs(inpembs.transpose(0, 1))
        Lp1, bszsl, _ = segembs.size()
        # if self.ar:
        #     segars = self.to_seg_hist(ar_states.transpose(0, 1)) #L+1 x bsz*seqlen x rnn_size

        bsz, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.hidden_dim

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        inits = self.h0_lin(srcenc) # bsz x 2*dim
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:] # (bsz x dim, bsz x dim)
        h0 = torch.tanh(h0).unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
            -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
        c0 = c0.unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
            -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()

        # inits = self.h0_lin(srcenc)  # bsz x 2*dim
        # h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        # h_prev = F.tanh(h0).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
        # h_prev = torch.repeat_interleave(h_prev, self.sample_size, dim=1)  # layer * (bsz*sample_size) * dim
        # c_prev = c0.unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
        # c_prev = torch.repeat_interleave(c_prev, self.sample_size, dim=1)  # layer * (bsz*sample_size) * dim

        # easiest to just loop over K
        seg_lls = []
        for k in range(self.tagset_size):
            if self.one_rnn:
                state_emb_sz = self.state_embs.size(3)
                condembs = torch.cat([segembs, self.state_embs[k].expand(Lp1, bszsl, state_emb_sz)], 2)
                states, _ = self.seg_rnns[0](condembs, (h0, c0)) # L+1 x bsz*seqlen x rnn_size
            else:
                states, _ = self.seg_rnns[k](segembs, (h0, c0)) # L+1 x bsz*seqlen x rnn_size

            # if self.ar:
            #     states = states + segars # L+1 x bsz*seqlen x rnn_size
            #
            # if self.drop.p > 0:
            #     states = self.drop(states)
            attnin1 = (states * self.state_att_gates[k].expand_as(states)
                       + self.state_att_biases[k].expand_as(states)).view(
                           Lp1, bsz, seqlen, -1)
            # L+1 x bsz x seqlen x rnn_size -> bsz x (L+1)seqlen x rnn_size
            attnin1 = attnin1.transpose(0, 1).contiguous().view(bsz, Lp1*seqlen, -1)
            attnin1 = torch.tanh(attnin1)
            ascores = torch.bmm(attnin1, srcfieldenc.transpose(1, 2)) # bsz x (L+1)slen x nfield
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
            states_k = torch.tanh(cat_ctx * self.state_out_gates[k].expand_as(cat_ctx)
                              + self.state_out_biases[k].expand_as(cat_ctx)).view(
                                  Lp1, -1, out_hid_sz)

            if self.sep_attn:
                attnin2 = (states * self.state_att2_gates[k].expand_as(states)
                           + self.state_att2_biases[k].expand_as(states)).view(
                               Lp1, bsz, seqlen, -1)
                # L+1 x bsz x seqlen x rnn_size -> bsz x (L+1)seqlen x emb_size
                attnin2 = attnin2.transpose(0, 1).contiguous().view(bsz, Lp1*seqlen, -1)
                attnin2 = torch.tanh(attnin2)
                ascores = torch.bmm(attnin2, srcfieldenc.transpose(1, 2)) # bsz x (L+1)slen x nfield
                ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)

            normfn = F.log_softmax
            wlps_k = normfn(torch.cat([self.decoder(states_k.view(-1, out_hid_sz)), #L+1*bsz*sl x V
                                       ascores.view(bsz, Lp1, seqlen, nfields).transpose(
                                           0, 1).contiguous().view(-1, nfields)], 1), dim=1)
            # concatenate on dummy column for when only a single answer...
            wlps_k = torch.cat([wlps_k, self.zeros.expand(wlps_k.size(0), 1)], 1)
            # get scores for predicted next-words (but not for last words in each segment as usual)
            psk = wlps_k.narrow(0, 0, self.L*bszsl).gather(1, combotargs.view(self.L*bszsl, -1))
            lls_k = logsumexp1(psk)
            # if k == 0:
                # print('hi,', psk.shape)
                # print(combotargs.shape, combotargs[0].shape)
                # print('*'*30, '\n', combotargs[0].view(-1),'\n',)
                # print(wlps_k.narrow(0, 0, self.L*bszsl).shape, combotargs.view(self.L*bszsl, -1).shape)
                # print( '+'*30, '\n')

            # sum up log probs of words in each segment
            seglls_k = lls_k.view(self.L, -1).cumsum(0) # L x bsz*seqlen
            # need to add end-of-phrase prob too
            eop_lps = wlps_k.narrow(0, bszsl, self.L*bszsl)[:, self.eop_idx] # L*bsz*seqlen
            seglls_k = seglls_k + eop_lps.contiguous().view(self.L, -1)
            seg_lls.append(seglls_k)

        #  K x L x bsz x seqlen -> seqlen x L x bsz x K -> L x seqlen x bsz x K
        obslps = torch.stack(seg_lls).view(self.K, self.L, bsz, -1).transpose(
            0, 3).transpose(0, 1)
        if self.Kmul > 1:
            obslps = obslps.repeat(1, 1, 1, self.Kmul)
        return obslps

    def to_seg_embs(self, xemb):
        """
        xemb - bsz x seqlen x emb_size
        returns - L+1 x bsz*seqlen x emb_size,
           where [1 2 3 4]  becomes [<s> <s> <s> <s> <s> <s> <s> <s>]
                 [5 6 7 8]          [ 1   2   3   4   5   6   7   8 ]
                                    [ 2   3   4  <p>  6   7   8  <p>]
                                    [ 3   4  <p> <p>  7   8  <p> <p>]
        """
        bsz, seqlen, emb_size = xemb.size()
        newx = [self.start_emb.expand(bsz, seqlen, emb_size)]
        newx.append(xemb)
        for i in range(1, self.L):
            pad = self.pad_emb.expand(bsz, i, emb_size)
            rowi = torch.cat([xemb[:, i:], pad], 1)
            newx.append(rowi)
        # L+1 x bsz x seqlen x emb_size -> L+1 x bsz*seqlen x emb_size
        return torch.stack(newx).view(self.L+1, -1, emb_size)


    # def get_gen_beta_score(self, x, combotargs):
    #     # first form a matrix of size L * seqlen * bsz * embedding to denote the embeddings of L words.
    #     bsz, seqlen = x.size()
    #     embs = self.word_embeds(x).view(bsz, seqlen, -1)  # seqlen x bsz*maxlocs*nfeats x emb_size
    #     inpembs = self.dropout(embs)
    #     segembs = self.to_seg_embs(inpembs) # this has L + 1, because it automatically appends a BOS Symbol.
    #                                         # it also adds paddings to the end of the sequence.
    #                                         # L+1 x  bsz*seqlen x embedding_dim
    #     # Assume that we do not consider context at all. What I'll do it to initialize all the embeddings to 0 .
    #     seg_lls = []
    #     for k in range(self.tagset_size):
    #         states, _ = self.seg_rnns[k](segembs, None)  # L+1 x bsz*seqlen x hiddem_dim
    #         logp_Vocab = F.log_softmax(self.decoder(states.view(-1, self.hidden_dim)), dim=1)  # L+1*bsz*sl x V
    #         psk = logp_Vocab.narrow(0, 0, self.L * bsz*seqlen).gather(1, combotargs.view(self.L * bsz*seqlen, -1).long())
    #         lls_k = logsumexp1(psk) # why do we need this step?
    #
    #         seglls_k = lls_k.view(self.L, -1).cumsum(0)  # L x bsz*seqlen
    #         # need to add end-of-phrase prob too
    #         eop_lps = logp_Vocab.narrow(0, bsz*seqlen, self.L * bsz*seqlen)[:, self.eop_idx]  # L*bsz*seqlen
    #         seglls_k += eop_lps.view(self.L, -1)
    #         seg_lls.append(seglls_k)
    #
    #             #  K x L x bsz x seqlen -> seqlen x L x bsz x K -> L x seqlen x bsz x K
    #     obslps = torch.stack(seg_lls).view(self.tagset_size, self.L, bsz, -1).transpose(
    #         0, 3).transpose(0, 1)
    #     return obslps
    #
    # def get_gen_beta_score_condi(self, x, combotargs,  srcenc, srcfieldenc, fieldmask):
    #
    #     seqlen, bsz, maxlocs, nfeats = x.size()
    #     embs = self.word_embeds(x.view(seqlen, -1))  # seqlen x bsz*maxlocs*nfeats x emb_size
    #
    #     if self.mlpinp:
    #         inpembs = self.inpmlp(embs.view(seqlen, bsz, maxlocs, -1)).mean(2)
    #     else:
    #         inpembs = embs.view(seqlen, bsz, maxlocs, -1).mean(2)  # seqlen x bsz x nfeats*emb_size
    #     #
    #     # if self.emb_drop:
    #     #     inpembs = self.drop(inpembs)
    #
    #     ##############################3
    #     # first form a matrix of size L * seqlen * bsz * embedding to denote the embeddings of L words.
    #     # bsz, seqlen = x.size()
    #     # embs = self.word_embeds(x).view(bsz, seqlen, -1)  # seqlen x bsz*maxlocs*nfeats x emb_size
    #     # inpembs = self.dropout(embs)
    #     ##############################3
    #
    #     segembs = self.to_seg_embs(inpembs.transpose(0,1)) # this has L + 1, because it automatically appends a BOS Symbol.
    #                                         # it also adds paddings to the end of the sequence.
    #                                         # L+1 x  bsz*seqlen x embedding_dim
    #     Lp1, bszsl, _ = segembs.size()
    #
    #     # process context.
    #     bsz,nfields, encdim = srcfieldenc.size()
    #     # print(bsz, seqlen)
    #     layers, rnn_size = self.layers, self.hidden_dim
    #
    #     # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
    #     inits = self.h0_lin(srcenc)  # bsz x 2*dim
    #     h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
    #     h0 = torch.tanh(h0).unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
    #         -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
    #     c0 = c0.unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
    #         -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
    #     # Assume that we do not consider context at all. What I'll do it to initialize all the embeddings to 0 .
    #     seg_lls = []
    #     for k in range(self.tagset_size):
    #         states, _ = self.seg_rnns[k](segembs, (h0, c0))  # L+1 x bsz*seqlen x hiddem_dim
    #
    #         # include the conditional part via attention.
    #         attnin1 = (states * self.state_att_gates[k].expand_as(states)
    #                    + self.state_att_biases[k].expand_as(states)).view(
    #             Lp1, bsz, seqlen, -1)
    #         # L+1 x bsz x seqlen x rnn_size -> bsz x (L+1)seqlen x rnn_size
    #         attnin1 = attnin1.transpose(0, 1).contiguous().view(bsz, Lp1 * seqlen, -1)
    #         attnin1 = torch.tanh(attnin1)
    #         ascores = torch.bmm(attnin1, srcfieldenc.transpose(1, 2))  # bsz x (L+1)seqlen x nfield
    #         ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)
    #         aprobs = F.softmax(ascores, dim=2)
    #         # bsz x (L+1)seqlen x nfields * bsz x nfields x dim -> bsz x (L+1)seqlen x dim
    #         ctx = torch.bmm(aprobs, srcfieldenc)
    #         # concatenate states and ctx to get L+1 x bsz x seqlen x rnn_size + encdim
    #         cat_ctx = torch.cat([states.view(Lp1, bsz, seqlen, -1),
    #                              ctx.view(bsz, Lp1, seqlen, -1).transpose(0, 1)], 3)
    #         out_hid_sz = rnn_size + encdim
    #         cat_ctx = cat_ctx.view(Lp1, -1, out_hid_sz)
    #         # now linear to get L+1 x bsz*seqlen x rnn_size
    #         states_k = torch.tanh(cat_ctx * self.state_out_gates[k].expand_as(cat_ctx)
    #                           + self.state_out_biases[k].expand_as(cat_ctx)).view(Lp1, -1, out_hid_sz)
    #
    #         if self.sep_attn:
    #             attnin2 = (states * self.state_att2_gates[k].expand_as(states)
    #                        + self.state_att2_biases[k].expand_as(states)).view(Lp1, bsz, seqlen, -1)
    #             # L+1 x bsz x seqlen x rnn_size -> bsz x (L+1)seqlen x emb_size
    #             attnin2 = attnin2.transpose(0, 1).contiguous().view(bsz, Lp1 * seqlen, -1)
    #             attnin2 = torch.tanh(attnin2)
    #             ascores = torch.bmm(attnin2, srcfieldenc.transpose(1, 2))  # bsz x (L+1)slen x nfield
    #             ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)
    #
    #         wlps_k = F.log_softmax(torch.cat([self.decoder(states_k.view(-1, out_hid_sz)),  # L+1*bsz*sl x V
    #                                    ascores.view(bsz, Lp1, seqlen, nfields).transpose(0, 1).contiguous()
    #                                   .view(-1, nfields)], 1), dim=1)
    #         # concatenate on dummy column for when only a single answer...
    #         wlps_k = torch.cat([wlps_k, self.zeros.expand(wlps_k.size(0), 1)], 1)
    #
    #         # if False: #k == 0:
    #         #     print(wlps_k.shape)
    #         #     print(x.shape)
    #         #     print(combotargs.shape)
    #         #     print(self.vocab_size, self.gen_vocab_size, nfields)
    #         #     print(x[:, :, 0, 0])
    #         #     print(combotargs.squeeze(2).long())
    #
    #
    #         # concatenate on dummy column for when only a single answer...
    #         # wlps_k = torch.cat([wlps_k, Variable(self.zeros.expand(wlps_k.size(0), 1))], 1)
    #         # get scores for predicted next-words (but not for last words in each segment as usual)
    #         psk = wlps_k.narrow(0, 0, self.L * bszsl).gather(1, combotargs.view(self.L * bszsl, -1))
    #         # psk = wlps_k.narrow(0, 0, self.L * bszsl).gather(1, combotargs.view(self.L * bszsl, -1).long())
    #         lls_k = logsumexp1(psk)
    #
    #         # sum up log probs of words in each segment
    #         seglls_k = lls_k.view(self.L, -1).cumsum(0)  # L x bsz*seqlen
    #         # need to add end-of-phrase prob too
    #         eop_lps = wlps_k.narrow(0, bszsl, self.L * bszsl)[:, self.eop_idx]  # L*bsz*seqlen
    #         seglls_k += eop_lps.contiguous().view(self.L, -1)
    #         seg_lls.append(seglls_k)  #  K x L x bsz x seqlen -> seqlen x L x bsz x K -> L x seqlen x bsz x K
    #     obslps = torch.stack(seg_lls).view(self.tagset_size, self.L, bsz, -1).transpose(
    #         0, 3).transpose(0, 1)
    #     return obslps

    def attn_layer2(self, h, srcfieldenc, fieldmask):

        # state_att_gates = torch.index_select(self.state_att_gates, 0, k.view(-1)).view(bsz, seglen,
        #                                                                                -1)  # (bsz*sample, seglen, h_dim)
        # state_att_biases = torch.index_select(self.state_att_biases, 0, k.view(-1)).view(bsz, seglen,
        #                                                                                  -1)  # (bsz*sample, seglen, h_dim)
        h = h.transpose(0,1)
        attnin1 = h * self.state_att_gates + self.state_att_biases  # (bsz*sample, seglen, h_dim)
        attnin1 = F.tanh(attnin1)  # (bsz*sample, seglen, h_dim)
        state_out_gates = self.state_out_gates
        state_out_biases = self.state_out_biases
        ascores = torch.bmm(attnin1, srcfieldenc.transpose(1, 2))  # (bsz*sample_size, seglen,  src_field_len)
        ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)  # (bsz*sample_size, seglen,  src_field_len)
        aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)
        ctx = torch.bmm(aprobs, srcfieldenc)  # (bsz*sample_size, seglen, w_dim)
        cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)
        states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)  # (bsz*sample_size, seglen, w_dim+h_dim)

        # if self.sep_attn:
        #     attnin2 = h * self.state_att2_gates + self.state_att2_biases  # (bsz*sample, seglen, h_dim)
        #     attnin2 = F.tanh(attnin2)  # (bsz*sample, seglen, h_dim)
        #     ascores = torch.bmm(attnin2, srcfieldenc.transpose(1, 2))  # (bsz*sample_size, seglen,  src_field_len)
        #     ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)  # (bsz*sample_size, seglen,  src_field_len)
        return states_k #, ascores

    def attn_layer3(self, h, srcfieldenc, fieldmask):

        # state_att_gates = torch.index_select(self.state_att_gates, 0, k.view(-1)).view(bsz, seglen,
        #                                                                                -1)  # (bsz*sample, seglen, h_dim)
        # state_att_biases = torch.index_select(self.state_att_biases, 0, k.view(-1)).view(bsz, seglen,
        #                                                                                  -1)  # (bsz*sample, seglen, h_dim)
        h = h.transpose(0,1)
        attnin1 = self.state_att_gates(h) # (bsz*sample, seglen, h_dim)
        attnin1 = F.tanh(attnin1)  # (bsz*sample, seglen, h_dim)
        state_out_gates = self.state_out_gates
        state_out_biases = self.state_out_biases
        ascores = torch.bmm(attnin1, srcfieldenc.transpose(1, 2))  # (bsz*sample_size, seglen,  src_field_len)
        ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)  # (bsz*sample_size, seglen,  src_field_len)
        aprobs = F.softmax(ascores, dim=2)  # (bsz*sample_size, seglen,  src_field_len)
        ctx = torch.bmm(aprobs, srcfieldenc)  # (bsz*sample_size, seglen, w_dim)
        cat_ctx = torch.cat([h, ctx], 2)  # (bsz*sample_size, seglen, w_dim+h_dim)
        states_k = F.tanh(cat_ctx * state_out_gates + state_out_biases)  # (bsz*sample_size, seglen, w_dim+h_dim)

        # if self.sep_attn:
        #     attnin2 = h * self.state_att2_gates + self.state_att2_biases  # (bsz*sample, seglen, h_dim)
        #     attnin2 = F.tanh(attnin2)  # (bsz*sample, seglen, h_dim)
        #     ascores = torch.bmm(attnin2, srcfieldenc.transpose(1, 2))  # (bsz*sample_size, seglen,  src_field_len)
        #     ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)  # (bsz*sample_size, seglen,  src_field_len)
        return states_k #, ascores


    def get_pre_train_bert(self, tokens_tensor, segments_tensor):
        with torch.no_grad():
            encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensor)
        return encoded_layers[-1]

    def get_bert_span_scores(self, condi):
        x = condi['inps']
        seqlen, bsz_orig, maxlocs, nfeats = x.size()
        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        bsz_orig, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.hidden_dim

        word_embs =  self.get_pre_train_bert(x, fieldmask)

        x = condi['inps']
        seqlen, bsz_orig, maxlocs, nfeats = x.size()
        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        bsz_orig, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.hidden_dim

        # create input to RNN by [word, next state]
        word_embs = self.dropout(self.word_embeds(x))
        cumsum_encoded = torch.cumsum(encoded_layers[-1], dim=1)
        app_cat = torch.cat([torch.zeros(1, 1, 768), cumsum_encoded])
        result = app_cat[:, 1:].unsqueeze(1) - app_cat[:, :-1].unsqueeze(2)

        # avg out the length factor.
        a = torch.Tensor(list(range(seqlen + 1)))
        len_factor = a[1:].unsqueeze(0) - a[:-1].unsqueeze(1)
        concat = result / len_factor.unsqueeze(0).unsqueeze(3)

        scores = self.q_binary(concat).squeeze(3)
        batch_size, T, _, tag_dim = scores.shape
        # scores * mask.unsqueeze(0).unsqueeze(3).expand(batch_size, T, T, tag_dim)
        return scores


        # problem is how to incorporate the idea of the

        # pooling function 1 -> compute the cumsum and then compute the mean.

        # pooling function 2 -> compute the conv layer over the embeddings.


    def pre_save_elmo(self, en_targs):
        character_ids = batch_to_ids(en_targs).to(self.device)
        # embeddings = elmo._elmo_lstm._token_embedder(character_ids)
        bilm_output = self.elmo_model._elmo_lstm(character_ids)
        temp = bilm_output['activations']
        # print(temp[0].shape, temp[1].shape, temp[2].shape)
        second_layer = temp[1].transpose(0, 1)
        return second_layer

    def get_elmo_span_scores1(self, condi):
        # LISA new
        x = condi['inps']
        seqlen, bsz_orig, maxlocs, nfeats = x.size()
        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        en_targ = condi['en_targ']

        bsz_orig, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.hidden_dim
        character_ids = batch_to_ids(en_targ).to(self.device)
        # embeddings = elmo._elmo_lstm._token_embedder(character_ids)
        bilm_output = self.elmo_model._elmo_lstm(character_ids)
        temp = bilm_output['activations']
        second_layer = temp[1].transpose(0, 1)



        if self.style == 1:
            # MLP style
            field_x = x[:, :, 0, 1:]  # seqlen, bsz,  3
            field_embs = self.dropout(self.word_embeds(field_x)).view(seqlen, bsz_orig, -1)
            word_embs = torch.cat([second_layer,
                                   torch.cat([self.bos_emb.expand(-1, bsz_orig, -1),
                                              field_embs,
                                              self.eos_emb.expand(-1, bsz_orig, -1)])], dim=-1)
            q_h = self.inpmlp(word_embs)
        elif self.style == 2:
            # need to learn another LSTM to pass through.
            pass
        elif self.style == 3:
            q_h = second_layer

        # attention step: should look at xi; basically, this is before the span score.
        # the reason is that if we do it after the span, it would be extremely expensive to compute.
        if self.additional_attn:
            # apply attention here.
            q_h = self.attn_layer(q_h[1:-1], srcfieldenc, fieldmask)  # bsz x seqlen x dim

        # get the span representation : [xi, xj, xi*xj, xi-xj]
        diff = q_h
        diff_ = diff.unsqueeze(1) - diff.unsqueeze(2)
        prod_ = diff.unsqueeze(1) * diff.unsqueeze(2)
        concat_ = torch.cat([diff.unsqueeze(1).expand(-1, seqlen, seqlen, -1), diff.unsqueeze(2).expand(-1, seqlen, seqlen, -1)], dim=-1)
        full = torch.cat([diff_, prod_, concat_], -1)
        # as a result fwd_diff[i,j] = the difference ends at j and begins at i (inclusive).
        scores = self.q_binary(full).squeeze(3)
        batch_size, T, _, tag_dim = scores.shape
        return scores


    def get_elmo_span_scores2(self, condi):
        # LISA new
        x = condi['inps']
        seqlen, bsz_orig, maxlocs, nfeats = x.size()
        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        en_targ = condi['en_targ']

        bsz_orig, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.hidden_dim
        character_ids = batch_to_ids(en_targ).to(self.device)
        bilm_output = self.elmo_model._elmo_lstm(character_ids)
        temp = bilm_output['activations']
        second_layer = temp[1].transpose(0, 1)

        if self.style == 1:
            # MLP style
            field_x = x[:, :, 0, 1:]  # seqlen, bsz,  3
            field_embs = self.dropout(self.word_embeds(field_x)).view(seqlen, bsz_orig, -1)
            word_embs = torch.cat([second_layer,
                                   torch.cat([self.bos_emb.expand(-1, bsz_orig, -1),
                                              field_embs,
                                              self.eos_emb.expand(-1, bsz_orig, -1)])], dim=-1)
            q_h = self.inpmlp(word_embs)
        elif self.style == 2:
            # need to learn another LSTM to pass through.
            pass
        elif self.style == 3:
            q_h = second_layer

        half_size = q_h.size(-1) // 2
        temp_f = q_h[:, :, :half_size]
        temp_b = q_h[:, :, half_size:]

        if self.additional_attn:
            q_fwd = self.attn_layer(temp_f, srcfieldenc, fieldmask)  # bsz x seqlen x dim
            q_bwd = self.attn_layer(temp_b, srcfieldenc, fieldmask)  # bsz x seqlen x dim
        else:
            q_fwd = temp_f.transpose(0, 1)
            q_bwd = temp_b.transpose(0, 1)

        fwd = q_fwd[:, 1:, :]
        bwd = q_bwd[:, :-1, :]
        fwd_diff = fwd[:, 1:].unsqueeze(1) - fwd[:, :-1].unsqueeze(2)
        bwd_diff = bwd[:, :-1].unsqueeze(2) - bwd[:, 1:].unsqueeze(1)
        concat = torch.cat([fwd_diff, bwd_diff], 3)
        scores = self.q_binary(concat).squeeze(3)

        # as a result fwd_diff[i,j] = the difference ends at j and begins at i (inclusive).

        batch_size, T, _, tag_dim = scores.shape
        return scores


    def get_elmo_span_scores3(self, condi):
        # LISA new
        x = condi['inps']
        seqlen, bsz_orig, maxlocs, nfeats = x.size()
        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        en_targ = condi['en_targ']

        bsz_orig, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.hidden_dim
        # create input to RNN by [word, next state]
        character_ids = batch_to_ids(en_targ).to(self.device)
        # embeddings = elmo._elmo_lstm._token_embedder(character_ids)
        bilm_output = self.elmo_model._elmo_lstm(character_ids)
        temp = bilm_output['activations']
        # print(temp[0].shape, temp[1].shape, temp[2].shape)
        second_layer = temp[1].transpose(0,1)
        # print(field_x.shape)
        # style 1. concatenate the ELMo embedding with the representation of the type and field -> apply a MLP.
        # style 2. concatenate the ELMo embedding with the representation of the type and field -> apply another
        # LSTM layer.


        # get the span representation : [xi, xj, xi*xj, xi-xj]



        if self.style == 1:
            # MLP style
            field_x = x[:,:,0,1:] # seqlen, bsz,  3
            field_embs = self.dropout(self.word_embeds(field_x)).view(seqlen, bsz_orig, -1)
            word_embs = torch.cat([second_layer,
                                    torch.cat([self.bos_emb.expand(-1, bsz_orig, -1),
                                               field_embs,
                                               self.eos_emb.expand(-1, bsz_orig, -1)])], dim=-1)
            q_h = self.inpmlp(word_embs)
        elif self.style == 2:
            # need to learn another LSTM to pass through.
            pass
        elif self.style == 3:
            q_h = second_layer

        # attention step: should look at xi; basically, this is before the span score.
        # the reason is that if we do it after the span, it would be extremely expensive to compute.

        half_size = q_h.size(-1) // 2
        temp_f = q_h[:, :, :half_size]
        temp_b = q_h[:, :, half_size:]

        if self.additional_attn:
            # apply attention here.
            q_fwd = self.attn_layer(temp_f, srcfieldenc, fieldmask)  # bsz x seqlen x dim
            q_bwd = self.attn_layer(temp_b, srcfieldenc, fieldmask)  # bsz x seqlen x dim
        else:
            q_fwd = temp_f.transpose(0, 1)
            q_bwd = temp_b.transpose(0, 1)

        fwd = q_fwd[:, 1:, :]
        bwd = q_bwd[:, :-1, :]
        fwd_diff = fwd[:, 1:].unsqueeze(1) - fwd[:, :-1].unsqueeze(2)
        bwd_diff = bwd[:, :-1].unsqueeze(2) - bwd[:, 1:].unsqueeze(1)
        # as a result fwd_diff[i,j] = the difference ends at j and begins at i (inclusive).
        concat = torch.cat([fwd_diff, bwd_diff], 3)
        scores = self.q_binary(concat).squeeze(3)
        batch_size, T, _, tag_dim = scores.shape
        return scores


    def get_lstm_span_scores(self, condi):
        # produces the span scores s_ij
        # mask = torch.ones(x.size(1), x.size(1)).tril() * -1e20

        x = condi['inps']
        seqlen, bsz_orig, maxlocs, nfeats = x.size()
        srcenc = condi["srcenc"]
        srcfieldenc = condi["srcfieldenc"]
        fieldmask = condi["fmask"]
        bsz_orig, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.hidden_dim
        # create input to RNN by [word, next state]
        # word_embs = self.dropout(self.word_embeds(x))
        # TODO: to be replaced.
        # inpembs = self.inpmlp(word_embs.view(seqlen, bsz_orig, maxlocs, -1)).mean(2)
        inpembs = condi['tgt']
        x_vec = torch.cat([self.bos_emb.expand(-1, bsz_orig, -1), inpembs, self.eos_emb.expand(-1, bsz_orig, -1)],
                          dim=0)
        inits = self.h0_lin(srcenc)  # bsz x 2*dim
        h0, c0 = inits[:, :rnn_size * self.layers], inits[:, rnn_size * self.layers:]  # (bsz x dim, bsz x dim)
        h0 = torch.tanh(h0).unsqueeze(0).contiguous().expand(layers * 2, bsz_orig, rnn_size).contiguous()
        c0 = c0.unsqueeze(0).contiguous().expand(layers * 2, bsz_orig, rnn_size).contiguous()
        q_h, _ = self.lstm(x_vec, (h0, c0))
        q_h = self.dropout(q_h)

        temp_f = q_h[:, :, :self.hidden_dim]
        temp_b = q_h[:, :, self.hidden_dim:]

        if self.additional_attn:
            # apply attention here.
            q_fwd = self.attn_layer(temp_f, srcfieldenc, fieldmask)  # bsz x seqlen x dim
            q_bwd = self.attn_layer(temp_b, srcfieldenc, fieldmask)  # bsz x seqlen x dim
        else:
            q_fwd = temp_f.transpose(0, 1)
            q_bwd = temp_b.transpose(0, 1)

        fwd = q_fwd[:, 1:, :]
        bwd = q_bwd[:, :-1, :]
        fwd_diff = fwd[:, 1:].unsqueeze(1) - fwd[:, :-1].unsqueeze(2)
        bwd_diff = bwd[:, :-1].unsqueeze(2) - bwd[:, 1:].unsqueeze(1)
        # as a result fwd_diff[i,j] = the difference ends at j and begins at i (inclusive).
        concat = torch.cat([fwd_diff, bwd_diff], 3)
        scores = self.q_binary(concat).squeeze(3)
        batch_size, T, _, tag_dim = scores.shape
        return scores




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
        result = torch.ones(L, T, bsz, K).to(self.device) * neginf
        score_temp = score_temp.permute(1, 2, 0, 3)
        for start in range(T):
            result[:min(L, T - start), start, :, :] = score_temp[start, start:start + L, :, :]
        return result


    def get_emission_score_for_alpha(self, score_temp):
        # TEMP
        '''
        get phi(y_{t+1:t+l}, z_{t+1}).
        Since we have a CRF, we can score each span by LSTM-minu
        s.
        :param x: the input database (in Long Tensor)
        :param y: the output sentence (in Long Tensor)
        :return: L x T x bsz x K: length = l, end=t, batch_size = bsz, and state size = k
        '''
        L = self.L
        bsz, T, _, K = score_temp.size()
        result = torch.ones(L, T, bsz, K) * neginf
        score_temp = score_temp.permute(1, 2, 0, 3)
        # print(score_temp.shape)
        for end in range(0, T):
            result[-min(L + 1, end + 1):, end, :, :] = score_temp[max(end - L + 1, 0):end + 1, end, :, :]
        return result


    def get_length_score(self):
        """
        returns:
           [1xK tensor, 2 x K tensor, .., L-1 x K tensor, L x K tensor] of logprobs
        """
        K = self.tagset_size * self.Kmul

        if self.unif_lenps:
            len_scores = self.len_scores.expand(K, self.L)
        else:
            state_embs = torch.cat([self.A_from, self.A_to.t()], 1)  # K x 2*A_dim
            len_scores = self.len_decoder(state_embs)  # K x L
        lplist = [len_scores.data.new(1, K).zero_()]
        # lplist = [Variable(len_scores.data.new(1, K).zero_())]
        # print(lplist[-1])
        for l in range(2, self.L + 1):
            lplist.append(nn.LogSoftmax(dim=1)(len_scores.narrow(1, 0, l)).t())
        return lplist, len_scores




    def get_trans_unif(self, uniqenc, seqlen=10, bsz=10):
        K = self.tagset_size * self.Kmul

        transi_ = torch.ones(K,K).to(self.device)
        if not self.yes_self_trans:
            transi_ = transi_ + self.selfmask

        unif_trans = nn.LogSoftmax(dim=0)(torch.ones(K).to(self.device))
        unif_trans2 = nn.LogSoftmax(dim=1)(transi_)
        init_lps = unif_trans.unsqueeze(0).expand(bsz, K)  # bsz x K
        trans_lps = unif_trans2.unsqueeze(0).unsqueeze(0).expand(seqlen - 1, bsz, K, K)
        return init_lps, trans_lps


    def get_transition_score(self, uniqenc, seqlen=10, bsz=10):
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
        trans_lps = nn.LogSoftmax(dim=1)(trans_lps.view(-1, K)).view(bsz, K, K)

        init_lps = nn.LogSoftmax(dim=1)(self.init_lin(uniqenc))  # bsz x K
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

        ''' think about how to use ELMo to initialize this term. '''
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
            embs = torch.tanh(embs.sum(1) + self.src_bias.expand(bsz * nfields, emb_size))
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
        uniqenc = torch.tanh(uniqenc)
        return {"srcenc":srcenc, "srcfieldenc":srcfieldenc, "uniqenc":uniqenc, 'src_unique_field':uniqfields}



    def get_score(self, z, dict_computed):
        '''
        compute the unnormalized score of p(y,z | x)

        :param x:
        :param y:
        :param z:
        :return:
        '''

        # TODO: 1. make the get_score function not really indexing.
        # TODO: 2. correct the masking funciton to speed it up.
        # TODO: speed up the beamsearch.
        # TODO" solve the beamsearch problem .
        emission = dict_computed['emission']
        transition = dict_computed['transition']  # bsz * K * K
        length_prob = dict_computed['length']
        init = dict_computed['init']  # bsz * K
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
                    result += emission[length, start_pos, b, state_curr] + \
                              length_prob[min(L - 1, seqlen - 1 - start_pos)][length, state_curr]  # L * T * bsz * K
                    if state_prev == -1:
                        result += init[b, state_curr]
                        state_prev = state_curr
                    else:
                        result += transition[start_pos - 1, b, state_prev, state_curr]
                        state_prev = state_curr
                # LISA 2
                # if state_prev != self.tagset_size-1:
                #     result += neginf
                result_lst.append(result)
            result_lst_all.append(result_lst)
        # print(Z)
        # print(result)
        # assert Z > result

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
            # LISA2
            _, last_lab = delt[seqlen][b].max(0)
            last_lab = last_lab.item()

            # LISA 2.1
            # last_lab = self.tagset_size-1
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


    def viterbi(self, dict_computed, constraints=None, ret_delt=False, get_score=False):
        """
        pi               - 1 x K
        bwd_obs_logprobs - L x T x bsz x K, obs probs ending at t
        trans_logprobs   - T-1 x bsz x K x K, trans_logprobs[t] = p(q_{t+1} | q_t).
        see https://hal.inria.fr/hal-01064672v2/document
        """
        pi, trans_logprobs, bwd_obs_logprobs, len_logprobs, = dict_computed['init'], dict_computed['transition'], \
                                                              dict_computed['emission_alpha'], dict_computed['length']
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
                # print(steps_back, L, min(L - 1, steps_back-1)+1)
                len_terms = torch.stack([len_logprobs[min(L, seqlen + 1 - t + jj) - 1][jj]
                                         for jj in range(min(L - 1, steps_back-1), -1, -1)])

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
            else:
                # LISA
                pass
                # tps = torch.zeros(K, bsz).fill_(neginf).to(self.device)  # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-indexed
                # tps[-1,:] = 0
                # delt_t = delt[t]  # bsz x K, viz, p(x, j)
                # delt_star_terms = (tps.transpose(0, 1)  # K x bsz x K
                #                    + delt_t.unsqueeze(2).expand(bsz, K, K).transpose(0, 1))
                # maxes, argmaxes = torch.max(delt_star_terms, 0)  # 1 x bsz x K, 1 x bsz x K
                # delt_star[t] = maxes.squeeze(0)
                # bps_star[t] = argmaxes.squeeze(0)

        # return delt, delt_star, bps, bps_star, recover_bps(delt, bps, bps_star)
        if ret_delt:
            back_track = self.recover_bps(delt, bps, bps_star), delt[-1]  # bsz x K total scores
        else:
            back_track = self.recover_bps(delt, bps, bps_star)

        ''' try to convert this to  list form and compute the score. '''
        if get_score:
            vit_lst = [[vit2lst(x)] for x in back_track]
            vit_scores = self.get_score(vit_lst, dict_computed)
            target = [torch.stack(samples) for samples in vit_scores]
            target = torch.stack(target, dim=0)
            vit_scores = (target - dict_computed['Z'].expand(bsz, 1))
            return back_track, vit_scores
        else:
            return back_track




    def just_fwd(self, pi, trans_logprobs, bwd_obs_logprobs, len_logprobs, constraints=None):
        """
        pi               - bsz x K
        bwd_obs_logprobs - L x T x bsz x K, obs probs ending at t
        trans_logprobs   - T-1 x bsz x K x K, trans_logprobs[t] = p(q_{t+1} | q_t)
        """
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
        # print(fwd_obs_logprobs.shape)
        L, seqlen, bsz, K = fwd_obs_logprobs.size()

        # we'll be 1-indexed for alphas and betas
        beta = [None] * (seqlen + 1)
        beta_star = [None] * (seqlen + 1)
        #

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


    def get_Z(self, dict_computed):
        '''
        compute the partition function for this HSMM of p(y | x).
        :param x:
        :param y:
        :return:
        '''
        fwd_obs_logprobs = dict_computed['emission']
        trans_logprobs = dict_computed['transition']
        len_logprobs = dict_computed['length']
        init = dict_computed['init']

        beta, beta_star = self.just_bwd(trans_logprobs, fwd_obs_logprobs, len_logprobs)
        print(beta_star[0].shape, init.shape)
        Z = logsumexp1(beta_star[0] + init)
        # print('partition from Z is {}'.format(Z))
        return Z


    def get_sample_gpu(self, dict_computed, constraints=None, sample_num=1):
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
        L, seqlen, bsz, K = emission.size()


        ''' To sample, we can use the beta scores, '''

        for b in range(bsz):
            result_lst_style = []
            result_vtb_style = []
            for sample_idx in range(sample_num):
                length_lst = [0]
                prob = torch.softmax(beta_star[0][b] + init[b], dim=0)
                sample = torch.multinomial(prob, 1).item()
                # print('batch = ', b, prob, sample)
                state_lst = [sample]

                t = seqlen
                while (t >= 1):
                    steps_fwd = min(L, t)
                    len_terms = length_prob[min(L - 1, steps_fwd - 1)]  # steps_fwd x K

                    if seqlen - t > 0:
                        betastar_nt = beta_star[seqlen - t][b]  # bsz x K
                        tps = transition[seqlen - t - 1]  # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-idxed

                        # TODO: nn.Softmax -> torch.softmax. move to CPU, precompute the random number.
                        prob_state = torch.softmax(betastar_nt + tps[b, state_lst[-1]], dim=0)  # bsz x K
                        sample_state = torch.multinomial(prob_state, 1).item()
                        # print('batch == ', b, prob_state, sample_state)
                        state_lst.append(sample_state)

                    if constraints is not None and constraints[seqlen - t + 1] is not None:
                        tmask = mask.narrow(0, 0, steps_fwd).zero_()
                        tmask.view(-1, K).index_fill_(0, constraints[seqlen - t + 1], neginf)

                    prob_dist = nn.Softmax(dim=0)(
                        torch.stack(beta[seqlen - t + 1:seqlen - t + 1 + steps_fwd])[:, b, state_lst[-1]]
                        + emission[:steps_fwd, seqlen - t, b, state_lst[-1]] +
                        len_terms.unsqueeze(1).expand(steps_fwd, bsz, K)[:, b, state_lst[-1]])
                    sample_dist = torch.multinomial(prob_dist, 1).item()
                    # print('batch === ', b, prob_dist, sample_dist)
                    length_lst.append(sample_dist + 1)

                    t = t - length_lst[-1]

                lst_style_sample, viterb_style_sample = self.process_sample(length_lst, state_lst)
                result_lst_style.append(lst_style_sample)
                result_vtb_style.append(viterb_style_sample)
            result_lst_style_all.append(result_lst_style)
            result_vtb_style_all.append(result_vtb_style)

        return result_lst_style_all, result_vtb_style_all


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

        emission = dict_computed['emission'].detach().cpu()
        transition = dict_computed['transition'].detach().cpu()
        beta_star = dict_computed['beta_star']
        init = dict_computed['init'].detach().cpu()
        beta = dict_computed['beta']
        length_prob = [x.cpu() for x in dict_computed['length']]

        beta_star = torch.stack(beta_star[:-1], dim=0).detach().cpu()
        beta = torch.stack(beta[1:], dim=0).detach().cpu()

        L, seqlen, bsz, K = emission.size()

        beta = torch.cat([torch.ones(1, bsz, K), beta], dim=0)

        ''' To sample, we can use the beta scores, '''

        for b in range(bsz):
            result_lst_style = []
            result_vtb_style = []

            for sample_idx in range(sample_num):

                length_lst = [0]
                prob = torch.softmax(beta_star[0, b] + init[b], dim=0)
                sample = torch.multinomial(prob, 1).item()
                # print('batch = ', b, prob, sample)
                state_lst = [sample]

                t = seqlen
                while (t >= 1):
                    steps_fwd = min(L, t)
                    len_terms = length_prob[min(L - 1, steps_fwd - 1)]  # steps_fwd x K

                    if seqlen - t > 0:
                        betastar_nt = beta_star[seqlen - t, b]  # bsz x K
                        tps = transition[seqlen - t - 1]  # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-idxed
                        prob_state = torch.softmax(betastar_nt + tps[b, state_lst[-1]], dim=0)  # bsz x K
                        sample_state = torch.multinomial(prob_state, 1).item()
                        # print('batch == ', b, prob_state, sample_state)
                        state_lst.append(sample_state)

                    # if constraints is not None and constraints[seqlen - t + 1] is not None:
                    #     tmask = mask.narrow(0, 0, steps_fwd).zero_()
                    #     # steps_fwd x bsz x K -> steps_fwd*bsz x K
                    #     tmask.view(-1, K).index_fill_(0, constraints[seqlen - t + 1], neginf)

                    prob_dist = torch.softmax(beta[seqlen - t + 1:seqlen - t + 1 + steps_fwd, b, state_lst[-1]]
                        + emission[:steps_fwd, seqlen - t, b, state_lst[-1]] +
                        len_terms.unsqueeze(1).expand(steps_fwd, bsz, K)[:, b, state_lst[-1]], dim=0)
                    sample_dist = torch.multinomial(prob_dist, 1).item()
                    # print('batch === ', b, prob_dist, sample_dist)
                    length_lst.append(sample_dist + 1)
                    t = t - length_lst[-1]

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
            viterb_style.append((start, end, state))
        return (start_lst, end_lst, state_lst), viterb_style


    def get_pr_mask(self, ):
        pass


    def posterior_reg_term_fullStyled(self, sent, condi, expected_count):
        pass



    def posterior_reg_term_swapStyled(self, sent, condi, expected_count):
        '''
            gather the distinct field, and their related position in the sentence.
            condi['src']: bsz x  seqlen  x  3, where the first dim is field name, the second field is the index and
            the last term is  the word used.
            condi['inps']: widx, kidx, idxidx, nidx
            expected_count:  L x  seqlen  x  bsz x K
        '''
        # each segment should corresponds to exactly a table entry.
        # There should be full table entry in the segment -- rewarding those.

        L, seqlen, bsz, K = expected_count[0].shape
        mask = torch.zeros(L, seqlen, bsz).to(self.device)
        for l in range(L):
            mask[l,:seqlen-l] = condi['inps'][l:, :, 0, 1]
        mask[mask == self.non_field] = 0
        pr_mask = torch.zeros(expected_count[0].shape).to(self.device)
        for b in range(bsz):
            for l in range(L):
                for e in range(seqlen):
                    temp_unq = mask[:l+1,e,b].unique()
                    if temp_unq.size(0) >= 2:
                        pr_mask[l, e, b, :] = 1
                    elif temp_unq.size(0) == 1 and 0 not in temp_unq:
                        temp = [x.item() for x in temp_unq if x.item() != 0][0]
                        pr_mask[l, e, b, :] = 1
                        pr_mask[l, e, b, self.labeled_state_dict[temp]] = 0
                    elif temp_unq.size(0) == 1 and 0 in temp_unq:
                        pr_mask[l, e, b, :self.labeled_states] = 1

                    # not cover the complete segment from the left
                    if e > 0 and mask[0,e-1,b].item() == mask[0,e,b].item():
                        pr_mask[l, e, b, :] += 1
                    # not cover the complete segment from the right
                    if e+l < seqlen-1 and mask[0,e+l+1,b].item() == mask[0,e+l,b].item():
                        pr_mask[l, e, b, :] += 1

        ''' get the penalizing counts'''
        pen_term = (pr_mask * expected_count[0])

        pen_term = pen_term.sum()
        # hope that all the states are covered exactly once.
        for b in range(bsz):
            uniq_fields = condi['src_unique_field'][b].unique()
            for s in uniq_fields:
                if s.item() == 1:
                    continue
                curr = expected_count[0][:,:,b, self.labeled_state_dict[s.item()]].sum()
                pen_term  += 5*(curr-1)*(curr-1)

        return pen_term / bsz


    def posterior_reg_term_wbCluster(self, sent, condi, expected_count):

        pr_mask = condi['detailed_tgt_mask']

        bsz = pr_mask.size(2)
        ''' get the normal penalizing counts'''
        # .unsqueeze(-1).expand_as(expected_count[0])
        pen_term = expected_count[0][pr_mask == -1].sum()  # penalizing the bad constituents spans.

        # good_term2 = (expected_count[0] * pr_mask_lab.float()).sum() # the bigger the better
        #
        # # good_term3 = (expected_count[0][pr_mask_lab == 1]).sum()
        #
        # good_term3 = (expected_count[0][pr_mask > 0] * pr_mask_lab[pr_mask > 0].float()).sum()


        temp_ = pr_mask[pr_mask > 0] - 1

        good_term = (expected_count[0][pr_mask > 0].gather(1, temp_.unsqueeze(1) )).sum() # the bigger the better
        # print(good_term, good_term2, good_term3, good_term == good_term2)
        # remaining term should be mapped to other trash states.

        mini_term = expected_count[0][pr_mask == 0].narrow(1, 0, self.tagset_size-2).sum() # the smaller the better.



        result = (pen_term + mini_term - 30 * good_term)

        return result / bsz


    def posterior_reg_term_wbCluster2(self, sent, condi, expected_count):

        pr_mask = condi['detailed_tgt_mask'].permute(2, 1, 0)
        expected_count = expected_count.sum(-1)
        bsz = pr_mask.size(2)
        full_shape = expected_count.shape
        ''' get the normal penalizing counts'''
        # .unsqueeze(-1).expand_as(expected_count[0])
        pen_term = expected_count[pr_mask == -1].sum()  # penalizing the bad constituents spans.

        # good_term2 = (expected_count[0] * pr_mask_lab.float()).sum() # the bigger the better
        #
        # # good_term3 = (expected_count[0][pr_mask_lab == 1]).sum()
        #
        # good_term3 = (expected_count[0][pr_mask > 0] * pr_mask_lab[pr_mask > 0].float()).sum()


        temp_ = pr_mask[pr_mask > 0] - 1

        good_term = (expected_count[pr_mask > 0].gather(1, temp_.unsqueeze(1) )).sum() # the bigger the better
        # print(good_term, good_term2, good_term3, good_term == good_term2)
        # remaining term should be mapped to other trash states.

        mini_term = expected_count[pr_mask == 0].narrow(1, 0, self.tagset_size-2).sum() # the smaller the better.



        result = (pen_term + mini_term - 10 * good_term)

        return result / bsz

    def posterior_reg_term_wbSoft(self, sent, condi, expected_count):
        pr_mask = condi['detailed_tgt_mask']
        bsz = pr_mask.size(2)
        full_shape = expected_count[0].shape

        ''' get the normal penalizing counts'''
        # .unsqueeze(-1).expand_as(expected_count[0])
        pen_term = expected_count[0][pr_mask == -1].sum()  # penalizing the bad constituents spans.

        ''' get the entr term -> want to minimzie the entropy.'''
        entr_term = expected_count[0][pr_mask > 0] + 1e-15 # shape = -1 * tagset_size
        cond_p_field = entr_term / entr_term.sum(-1).unsqueeze(-1).expand_as(entr_term)
        entr_term_min_ = -(cond_p_field * cond_p_field.log()).sum()

        entr_term_min = 0
        for elem in pr_mask.unique():
            if elem == -1 or elem == 0:
                continue
            else:
                entr_term = expected_count[0][pr_mask == elem] + 1e-15  # shape = -1 * tagset_size
                entr_term = entr_term.sum(0)
                cond_p_field = entr_term / entr_term.sum(-1).unsqueeze(-1).expand_as(entr_term)
                entr_term_min -= (cond_p_field * cond_p_field.log()).sum()

        ''' another entr_term that concerns about the global usage of all states -> want to maximize '''
        temp = expected_count[0][pr_mask > 0].sum(0) # >= 0?
        temp = temp / temp.sum()
        entr_term_max = -torch.sum(temp * temp.log())

        result = (pen_term + entr_term_min + entr_term_min_ - 15 * entr_term_max)
        return result / bsz

    def posterior_reg_term_wbEntr(self, sent, condi, expected_count):
        '''

        :param sent:
        :param condi: (L, seqlen, bsz)
        :param expected_count:
        :return:
        '''
        pr_mask = condi['detailed_tgt_mask']
        bsz = pr_mask.size(2)
        full_shape = expected_count[0].shape

        ''' get the normal penalizing counts'''
        # .unsqueeze(-1).expand_as(expected_count[0])
        pen_term = expected_count[0][pr_mask == 10].sum() # penalizing the bad boundaries.

        ''' get the entr term -> want to minimzie the entropy.'''
        entr_term = expected_count[0][pr_mask == 3] + 1e-15 # shape = -1 * tagset_size
        cond_p_field = entr_term / entr_term.sum(-1).unsqueeze(-1).expand_as(entr_term)
        # print(cond_p_field.shape)
        # print(cond_p_field.sum(-1))
        entr_term_min = -(cond_p_field * cond_p_field.log()).sum()
        # print(entr_term_min.shape)

        ''' another entr_term that concerns about the global usage of all states -> want to maximize '''
        temp = expected_count[0][pr_mask != 10].sum(0)
        # print(temp.shape)
        # print(temp)
        # print(temp.sum())
        temp = temp / temp.sum()
        # print(temp.sum())
        # an entr here is.
        entr_term_max = -torch.sum(temp * temp.log())
        # print(entr_term_max)

        result = (pen_term + entr_term_min - 15*entr_term_max)
        return result / bsz

    def posterior_reg_term_global(self, sent, condi, expected_count):
        '''
            gather the distinct field, and their related position in the sentence.
            condi['src']: bsz x  seqlen  x  3, where the first dim is field name, the second field is the index and
            the last term is  the word used.
            condi['inps']: widx, kidx, idxidx, nidx
            expected_count:  L x  seqlen  x  bsz x K
        '''
        ''' 
            assume the condi['detailed_tgt_mask'] is the matrix of size L * seqlen * bsz * L, augmented with 
            (i,j, fieldname)
        '''
        # regularize expected count to be close to m -> using cross entropy..
        # yy = condi['targs']
        # yy_emb = condi['tgt'].transpose(0,1)
        # bsz, seqlen = yy.shape
        # yy_prob = torch.softmax(self.w_lookup(yy_emb), dim=-1) + 1e-15
        # yy_tag = yy_prob.log() # seqlen x bsz x K


        ff = condi['detailed_tgt_mask']
        ff_emb = condi['named_space_emb']
        # print(expected_count[0].shape)
        ff_prob = torch.softmax(self.ff_lookup(ff_emb), dim=-1) + 1e-15
        ff_tag = ff_prob.log()
        _, _, bsz = ff.shape
        # print(ff_emb.shape)
        # print(ff.shape)
        # print(ff_tag.shape)

        # count1 = ff.clone()
        # count1[ff <= 0] = 0
        # count1[ff > 0] = 1
        # print(count1)
        # print(ff)
        # print(ff == count1)

        mini = -torch.sum((expected_count[0][ff>0] + 1e-15).log() * ff_prob, dim=-1)
        state_count = expected_count[0][ff>0].sum(0) # bsz x K
        # print(state_count.shape)
        normalize = state_count.sum()
        state_prob = (state_count / (normalize + 1e-15)) + 1e-15

        # # normalize = state_count.sum(1).unsqueeze(1).expand(-1, self.tagset_size)
        # state_prob = (state_count / (normalize + 1e-15)) + 1e-15
        # # regularize the entropy of this, hope it's large.
        maxi = -torch.sum(state_prob * state_prob.log(), dim=0)
        maxi = maxi.sum()

        # regularize the w matrix to be sparse by regularizing the entropy.
        # print(ff_prob.shape, ff_tag.shape)
        w_entr = -torch.sum(ff_prob * ff_tag, dim=-1) # seqlen x bsz  hope to make this small.


        return (mini.sum() - maxi + w_entr.sum()) / bsz

    def posterior_reg_term_phraseStyled(self, sent, condi, expected_count):
        '''
            gather the distinct field, and their related position in the sentence.
            condi['src']: bsz x  seqlen  x  3, where the first dim is field name, the second field is the index and
            the last term is  the word used.
            condi['inps']: widx, kidx, idxidx, nidx
            expected_count:  L x  seqlen  x  bsz x K
        '''

        pr_mask = condi['detailed_tgt_mask']
        bsz = pr_mask.size(2)
        ''' get the normal penalizing counts'''
        pen_term = expected_count[0][pr_mask == -1].sum()  # penalizing the bad constituents spans.
        temp_ = pr_mask[pr_mask > 0] - 1
        good_term = (expected_count[0][pr_mask > 0].gather(1, temp_.unsqueeze(1))).sum()  # the bigger the better

        count_pen = 0

        # for b in range(bsz):
        #     uniq_fields = (pr_mask[:,:,b] - 1).unique()
        #
        #     uniq_fields = [x for x in uniq_fields if x >= 0 and x < 7]
        #     curr = expected_count[0][:, :, b, uniq_fields].sum(0).sum(1)  # length=unique
        #     count_pen += ((curr - 1) * (curr - 1)).sum()
        return (pen_term +  count_pen) / bsz - good_term

    def posterior_reg_term_soft(self, sent, condi, expected_count):
        '''
            gather the distinct field, and their related position in the sentence.
            condi['src']: bsz x  seqlen  x  3, where the first dim is field name, the second field is the index and
            the last term is  the word used.
            condi['inps']: widx, kidx, idxidx, nidx
            expected_count:  L x  seqlen  x  bsz x K
        '''

        marginals = condi['detailed_tgt_mask'].transpose(0,1) # float matrix of seqlen * bsz * K
        L, seqlen, bsz, K = expected_count[0].shape
        ''' get the normal penalizing counts'''
        # currently there is no good or bad span. The onlything is that we need to use the expectation matrix to
        # aggregate and regularize the marginals.

        pen_term = expected_count[0]
        temp_term = torch.cat([torch.zeros(L, L, bsz, K).to(self.device), pen_term], dim=1)
        agg_term = torch.zeros(seqlen, bsz, K).to(self.device)
        for i in range(L):
            agg_term += (temp_term[i:, (L-i):(L-i+seqlen)]).sum(0)

        agg_term = agg_term + 1e-15

        entr_min = (marginals * agg_term.log()).sum(-1) # seqlen * bsz

        return entr_min.mean()

    def posterior_reg_term_phraseStyled2(self, sent, condi, expected_count):
        '''
            gather the distinct field, and their related position in the sentence.
            condi['src']: bsz x  seqlen  x  3, where the first dim is field name, the second field is the index and
            the last term is  the word used.
            condi['inps']: widx, kidx, idxidx, nidx
            expected_count:  L x  seqlen  x  bsz x K
        '''
        # generate a mask that mark out the necessary segments that should be restricted to only contain some words.
        #   case 1. if we have a [A A A b B B] as a span, then state A should be penalized, state B should be penalized.
        #   case 2. if we have a [A A A b] as a span, then state A should not be penalized, and state B should.
        #   case 3. if we have a [b b b ] as a span, then state A  and state B should both be penalized.
        #   -> as a conclusion, we should only not penalize a state if the only one field is correctly related to the current thing.

        # also if we have A [A A A b b a ], where the complete sentence is not included, I will penalize this situation harshly as well.
        pr_mask = condi['detailed_tgt_mask'].float()
        bsz = pr_mask.size(2)

        print(pr_mask.shape)


        ''' get the normal penalizing counts'''
        pen_term = (pr_mask * expected_count[0])
        # print(pen_term.sum())


        reward_term = 0
        for k in range(self.labeled_states):
            temp = expected_count[0][:,:,:,k][pr_mask[:,:,:,k]<=0]
            if temp.size(0) == 0:
                continue
            reg = temp.sum() / (expected_count[0][:,:,:,k].sum() + 1e-10)
            reward_term += reg

        count_pen = 0
        for b in range(bsz):
            uniq_fields = pr_mask.unique().data
            uniq_fields = [self.labeled_state_dict[x.item()] for x in uniq_fields.data if x.item() != 1]
            for s in range(self.tagset_size):
                if s in uniq_fields:
                    curr = expected_count[0][:,:,b, s].sum()
                    count_pen +=  torch.abs(curr - 1)
                else:
                    count_pen +=  torch.abs(curr)

        return (pen_term.sum() + 10*count_pen) / bsz  - 10*reward_term


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

        count_helper = torch.zeros(fwd_obs_logprobs.shape, requires_grad=True).to(self.device)
        fwd_obs_logprobs = fwd_obs_logprobs + count_helper

        L, seqlen, bsz, K = fwd_obs_logprobs.size()

        # we'll be 1-indexed for alphas and betas
        entr = [None] * (seqlen + 1)
        entr_star = [None] * (seqlen + 1)

        beta = [None] * (seqlen + 1)
        beta_star = [None] * (seqlen + 1)

        # LISA2
        beta[seqlen] = trans_logprobs.data.new(bsz, K).zero_()
        entr[seqlen] = trans_logprobs.data.new(bsz, K).zero_()
        # print(beta[seqlen])
        # LISA2.1
        # beta[seqlen] =  torch.ones(bsz, K).fill_(neginf)
        # entr[seqlen] =  torch.zeros(bsz, K)
        # beta[seqlen][:,-1] = 0
        # entr[seqlen][:,-1] = 0
        # print(beta[seqlen])

        mask = trans_logprobs.data.new(L, bsz, K)

        for t in range(1, seqlen + 1):
            steps_fwd = min(L, t)
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
            # print(torch.stack(beta[seqlen - t + 1:seqlen - t + 1 + steps_fwd])[:, 0])
            # print( fwd_obs_logprobs[:steps_fwd, seqlen - t][:, 0])
            # print(len_terms.unsqueeze(1).expand(steps_fwd, bsz, K)[:, 0])
            #
            # print('qaq')
            # print(beta_star_terms[:, 0])
            # print()
            if constraints is not None and constraints[seqlen - t + 1] is not None:
                beta_star_terms = beta_star_terms + tmask

            beta_star[seqlen - t] = logsumexp0(beta_star_terms)  # bsz * K
            # print(beta_star[seqlen - t])
            weight_logprob = beta_star_terms - beta_star[seqlen - t].unsqueeze(0).expand(steps_fwd, bsz, K)
            weight_prob = weight_logprob.exp()

            entr_star_terms = weight_prob * (torch.stack(entr[seqlen - t + 1:seqlen - t + 1 + steps_fwd])
                                             - weight_logprob)  # steps_fwd x bsz x K
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
                entr_terms = beta_prob * (entr_star[seqlen - t].unsqueeze(1).expand(bsz, K, K) - beta_logprob)
                entr[seqlen - t] = torch.sum(entr_terms, dim=2)

        Z_terms = beta_star[0] + init
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

        if self.posterior_reg:
            count_expected = torch.autograd.grad(Z.sum(), count_helper, create_graph=True, allow_unused=True)
        else:
            count_expected = None
        return Z, entr_Z, count_expected


    def get_weights(self, sent, condi):
        init, trans = self.get_trans(condi['uniqenc'], sent.size()[1], sent.size()[0])
        # scores_beta = self.get_gen_beta_score(sent, combotargs)
        # srcenc, srcfieldenc, Variable(fmask, volatile=True),
        # if False:
        #     scores_beta = self.get_gen_beta_score_condi(condi['inps'], combotargs, condi['srcenc'],
        #                                                 condi['srcfieldenc'], condi['fieldmask'])
        if self.decoder_type == 'gen':
            combotargs = condi['combotargs']
            scores_beta = self.obs_logprobs(condi['inps'], condi['srcenc'],
                                            condi['srcfieldenc'], condi['fmask'], combotargs)  # L x T x bsz x K
        else:
            scores_temp = self.get_span_scores(condi)
            scores_beta = self.get_emission_score(scores_temp)
            # print(scores_beta)
            if self.hard_code:
                temp1 = condi['detailed_tgt_mask']
                mask1 = (temp1 > 0)
                scores_beta[mask1] = neginf

        len_lst, len_score = self.get_length_score()
        scores_alpha = bwd_from_fwd_obs_logprobs(scores_beta.data)


        dict_computed = {}
        dict_computed['emission'] = scores_beta
        dict_computed['emission_alpha'] = scores_alpha
        dict_computed['transition'] = trans
        dict_computed['length'] = len_lst
        dict_computed['init'] = init
        # # print(scores_beta.shape, scores_alpha.shape, trans.shape, init.shape)
        # # compute the aggregate log marginals:
        # # target shape = batch, N, L, K, K
        # log_marginals = scores_alpha.permute(2, 1, 0, 3).unsqueeze(-1).expand(-1, -1, -1, -1, self.tagset_size).clone()
        # log_marginals[:,0,:,:,:] += init.unsqueeze(1).unsqueeze(-1).expand(-1, self.L, self.tagset_size, self.tagset_size).clone()
        # temp = trans.permute(1, 0, 2, 3).unsqueeze(2).expand(-1, -1, self.L, -1, -1).clone()
        # log_marginals[:,1:, :, :,:] += temp
        # # print(log_marginals.shape)
        log_marginals = 0

        return dict_computed, log_marginals


    def save_model(self, path='latest_model'):
        torch.save(self.state_dict(), path)
        # with open(path, 'bw') as f:
        # pickle.dump(self, f)


    def load_model(self, path='latest_model'):
        self.load_state_dict(torch.load(path))
        # self = torch.load(path)
        return self


