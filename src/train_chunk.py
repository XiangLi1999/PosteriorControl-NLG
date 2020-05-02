from glo import get_logger, experiment, Option, Global, VarDict
import sys
import os
import math
import random
import argparse
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Chunking_Reader import Data_Loader_Chunk, Data_Loader_ptb
import labeled_data2 as labeled_data
from utils import *
# from data.utils import get_wikibio_poswrds, get_e2e_poswrds
from rnn import RNN_cond_Gen, Chunking_Model
from rnn_new import RNNLM2
from Chunking_Gen import Chunking_RNN
from hsmm_gen import HSMM_generative
import numpy as np
from toy_data import *
from beam_search import *
import time, datetime

logger = get_logger()

def _start():
    logger.info('Python Version:\n' + sys.version)
    logger.info('Numpy Version:%s' % np.__version__)
    logger.info('PyTorch Version:%s' % torch.__version__)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Option.add(vars(args))
    logger.info('Configuration:\n' + str(Option))
    logger.info('+'*77)


def _error_break():
    logger.info('<STOP> due to errors')

def _finish():
    # print(args.labeled_states)
    logger.info('<DONE>')


def visual_viterb(viterb, sentence):
    lst = []
    for (start, end, state) in viterb:
        lst.append('[{}'.format(state))
        lst += sentence[start:end]
        lst.append('{}]'.format(state))
    return lst

def segment_num_viterbi(viterbi):
    seg_num = sum([len(elem) for elem in viterbi])
    return seg_num

def beam_rnn(model_, example, timing_beam):
    (src, uniqfields, inps, fmask, amask, combotargs, en_src, en_targ,  src_wrd2fields), y, state_z = example
    y = y.transpose(1, 0)
    condi = model.encode_table(src, amask, uniqfields)
    condi['src'] = src
    condi['fmask'] = fmask
    condi['src_wrd2fields'] = src_wrd2fields
    bsz = y.size(0)
    decoded_batch_str = model_.rnn_lm.beam_forward(condi, bsz, timing_beam)
    return decoded_batch_str


def control_rnn(model_, example, template, beam_size, timing_beam):
    with torch.no_grad():
        (src, uniqfields, inps, fmask, amask, combotargs, en_src, en_targ,  src_wrd2fields), y, state_z = example
        y = y.transpose(1, 0)
        condi = model.encode_table(src, amask, uniqfields)
        condi['src'] = src
        condi['fmask'] = fmask
        condi['inps'] = inps
        condi['combotargs'] = combotargs
        condi['targs'] = combotargs[0]
        condi['src_wrd2fields'] = src_wrd2fields
        condi['template'] = [torch.LongTensor(x).to(args.device) for x in template]
        bsz = y.size(0)
        assert bsz == len(condi['template'])
        decoded_batch_str = model_.rnn_lm.beam_control(condi, bsz, beam_size)
    return decoded_batch_str

def beam_single(model_, example):
    with torch.no_grad():
        (src, uniqfields, inps, fmask, amask, combotargs, en_src, en_targ), y, state_z = example
        y = y.transpose(1, 0)
        condi = model.encode_table(src, amask, uniqfields)
        condi['fmask'] = fmask
        condi['inps'] = inps
        condi['combotargs'] = combotargs
        condi['targs'] = combotargs[0]
        bsz = y.size(0)
        seqlen = y.size(1)

        dict_computed = model.hsmm_crf.get_weights(y,condi)
        result_vit = model.hsmm_crf.viterbi(dict_computed)
        result = torch.zeros(bsz,  seqlen)
        for b_idx, samp in enumerate(result_vit):
            temp = sum([[state] * (t - s) for (s, t, state) in samp], [])
            result[b_idx] = torch.LongTensor(temp)
        latent_z = result.long()

        inits = model_.rnn_lm.h0_lin(condi['srcenc'])  # bsz x 2*dim
        h0, c0 = inits[:, :args.hidden_dim], inits[:, args.hidden_dim:]  # (bsz x dim, bsz x dim)
        h_prev = F.tanh(h0).unsqueeze(0).expand(1, -1, args.hidden_dim).contiguous()
        c_prev = c0.unsqueeze(0).expand(1, -1, args.hidden_dim).contiguous()


        decoded_batch = beam_decode(model_.rnn_lm, (h_prev, c_prev),condi, latent_z, bsz)

        result = []
        for examp_decode in decoded_batch:
            temp = torch.cat([torch.stack(x) for x in examp_decode], dim=1)
            result.append(temp.squeeze(-1).transpose(0,1))
    return result

def beam_corpus(model_, epoch_num='final', corpora_lab='valid', controlled=False, template=None):
    # NEW_LISA
    result = []
    correct = []
    src_ = []
    if corpora_lab == 'valid':
        corpora = corpus.valid
    elif corpora_lab == 'test' or corpora_lab == 'test_single':
        corpora = corpus.valid
    else:
        corpora = corpus.train

    timing_beam = defaultdict(float)
    temp_idx = 0
    full_length = len(corpora)
    logger.info('the number of batches is {}'.format(full_length))
    for i in range(len(corpora)):
        x, lab, src, locs, inps, en_src, en_targ, src_wrd2fields = corpora[i]
        if controlled:
            try:
                controlled_template = template[temp_idx]
            except:
                print(len(template), temp_idx)
        cidxs = None
        seqlen, bsz = x.size()
        nfields = src.size(1)

        if corpora_lab == 'valid' or corpora_lab == 'train' or corpora_lab == 'test':
            if seqlen < args.L or seqlen > args.max_seqlen:
                continue
        else:
            pass

        if not controlled:
            fmask = make_masks(src, args.pad_idx, max_pool=args.max_pool)
            # uniqfields = get_uniq_fields(src, args.pad_idx)
            combotargs = None

            if args.cuda:
                src = src.to(args.device)
                fmask = fmask.to(args.device)
                # uniqfields = uniqfields.to(args.device)

            # print(combotargs.device)
            result.append(beam_rnn(model_, ((src, None, inps, fmask, None, combotargs,
                                             en_src, en_targ,  src_wrd2fields), x, lab), timing_beam))
        else:
            combotargs = make_combo_targs(locs, x, 1, nfields, corpus.ngen_types)
            fmask = make_masks(src, args.pad_idx, max_pool=args.max_pool)
            uniqfields = get_uniq_fields(src, args.pad_idx)  # bsz x max_fields

            if args.cuda:
                combotargs = combotargs.to(args.device)
                src = src.to(args.device)
                x = x.to(args.device)
                inps = inps.to(args.device)
                fmask = fmask.to(args.device)
                # amask = amask.to(args.device)
                uniqfields = uniqfields.to(args.device)

            result.append(control_rnn(model_, ((src, uniqfields, inps, fmask, None, combotargs,
                                             en_src, en_targ, src_wrd2fields), x, lab), controlled_template,
                                      args.beam_size), timing_beam)

        sys.stdout.write('-')
        sys.stdout.flush()
        if timing_beam is not None:
            if i % (1000) == 0:
                str_ = ''
                for kk, elem  in timing_beam.items():
                    str_ +=  kk + ':' + str(datetime.timedelta(seconds=int(elem))) + '\t'
                print(str_)
                sys.stdout.flush()


        if not args.mem_eff_flag:
            correct += [' '.join(x) for x in en_targ]
            src_ += [' '.join(x) for x in en_src]
        else:
            correct += en_targ
            src_ += en_src
        temp_idx += 1

        # write to file.
    logger.info('writing the beam search result to {}'.format(args.save_out + '.beam'))
    with open(args.save_out + '.{}_{}_beam'.format(epoch_num, corpora_lab), 'w') as f:
        for elem in result:
            print(elem, file=f)
    logger.info('writing the correct answer to {}'.format(args.save_out + '.corr'))
    with open(args.save_out + '.{}_{}_corr'.format(epoch_num, corpora_lab), 'w') as f:
        for elem in correct:
            print(elem, file=f)
    logger.info('writing the source to {}'.format(args.save_out + '.src'))
    with open(args.save_out + '.{}_{}_src'.format(epoch_num, corpora_lab), 'w') as f:
        for elem in src_:
            print(elem, file=f)

    return result


def gather_rnn_stats(model_, example, summary):
    x, state_z, mask_pr, ner, en_targ = example

    condi = {}
    tgt_enc = model.word_vecs(x)

    condi['tgt'] = tgt_enc
    condi['tgt_word'] = tgt_enc
    condi['inps'] = x
    condi['targs'] = x
    condi['en_targ'] = en_targ


    bsz, seqlen = x.shape

    sample_lst_void = [ [([0],[seqlen],[0])] for _ in range(bsz)]

    result_dict = model_.forward(condi, sample_lst_void)
    word_ll, state_llp = result_dict['p(y)'], result_dict['p(z)']

    log_f = word_ll + state_llp
    obj = log_f.mean()


    summary['logp(y,z)'] += log_f.mean().item()
    summary['logp(word)'] += word_ll.mean().item()
    summary['logp(state)'] += state_llp.mean().item()
    summary['sent_num'] += 1
    summary['#Tokens'] += seqlen

    return -obj.mean()


def test_rnn(model, epoch_num, corpora_lab = 'valid'):
    summary = defaultdict(float)
    summary['viterbi'] = []
    if corpora_lab == 'valid':
        corpora = corpus.valid
        logger.info('evaluating on valid')
    elif corpora_lab == 'test':
        corpora = corpus.test
        logger.info('evaluating on test')
    else:
        corpora = corpus.train
        logger.info('evaluating on train')
    with torch.no_grad():
        for i in range(len(corpora)):
            x, pos, chk, ner, x_str = corpora[i]
            # cidxs = train_cidxs[trainperm[batch_idx]] if epoch <= args.constr_tr_epochs else None
            cidx = None
            bsz, seqlen = x.size()

            if seqlen < args.L or seqlen > args.max_seqlen:
                continue
            if args.cuda:
                x = x.to(args.device)
            obj = gather_rnn_stats(model, (x, pos, chk, ner, x_str), summary)

    logger.info('DONE EVAL epoch {}:{}'.format(epoch_num+1, print_dict(summary)))
    return summary, -summary['logp(y,z)']/summary['#Tokens']


def gather_stats(model_, example, summary, kl_pen, epo_num, file_handle, verbose=False, timing=None, more_info=False):
    x, state_z, mask_pr, ner, en_targ = example

    condi = {}
    tgt_enc = model.get_tgt_wembs(x)


    condi['tgt'] = tgt_enc
    condi['tgt_word'] = tgt_enc
    condi['inps'] = x
    condi['targs'] = x
    condi['en_targ'] = en_targ
    condi['z_emb'] = model.get_state_embs()
    condi['x_emb'] = model.get_word_embs()

    if mask_pr is not None:
        condi['detailed_tgt_mask'] = mask_pr.to(args.device)

    # print(en_targ)
    # print(condi['detailed_tgt_mask'][:,:,0])
    # print(en_targ)
    # print()

    # print('gather stats')
    # print(condi['en_targ'])
    # print(x)
    bsz, seqlen = x.shape
    # condi['detailed_tgt_mask'] = src_wrd2fields.to(args.device)

    # print(condi['targs'])
    result, dict_computed = model_.forward_with_crf(x, condi, args.sample_size, gold_z=None, timing=None, more_info=more_info)


    with torch.no_grad():
        result_vit, score_vit = model_.hsmm_crf.viterbi(dict_computed, get_score=True)

    # time_dd = time.time()
    log_f = result['word_ll'] + kl_pen * result['state_llp']
    iwae_ll = log_f.mean().detach() + kl_pen * result["entropy"].mean().detach()
    obj = log_f.mean()

    if args.multi_task == 'yes':
        obj += result['q(x)']

    if epo_num < args.train_q_epochs:
        obj += kl_pen * result["entropy"].mean()
        if args.sample_size > 1:
            baseline = torch.zeros_like(log_f)
            baseline_k = torch.zeros_like(log_f)
            for k in range(args.sample_size):
                baseline_k.copy_(log_f)
                baseline_k[:, k].fill_(0)
                baseline[:, k] = baseline_k.detach().sum(1) / (args.sample_size - 1)
            # print('see ', baseline.detach(), result['state_llq'].mean())
            obj += ((log_f.detach() - baseline.detach()) * result['state_llq']).mean()
        else:
            obj += (log_f.detach() * result['state_llq']).mean()

    # have posterior regularization on
    if args.posterior_reg == 1:
        obj -= args.pr_coef * result['pr']

    kl_total = (result['state_llq'] - result['state_llp']).mean().detach()
    ''' add a sanity check for the viterbi result against the non-viterbi ones. '''
    # temp = [[vit2lst(result_vit[0])]]

    ll_word_total = result['word_ll'].mean()
    entropy_total = result["entropy"].mean().item()

    if verbose:
        print('ELBO={0:.4g}, KL={1:.4g}, logp(X)={2:.4g}, logp(word)={3:.4g}, '
              'entr={4:.4g}'.format(iwae_ll.item(), kl_total, log_f.item(),
                                    ll_word_total, entropy_total))


    summary['ELBO'] += iwae_ll.item()
    summary['#Tokens'] += x.size(1)
    summary['KL'] += kl_total.item()
    summary['logp(y,z)'] += log_f.mean().item()
    summary['logp(word)'] += ll_word_total.item()
    summary['logp(state)'] += result['state_llp'].mean().item()
    summary['entr'] += entropy_total
    summary['sent_num'] += 1

    summary['total_joint'] += log_f.mean(1).sum(0).item()
    summary['total_word'] += result['word_ll'].mean(1).sum(0).item()
    ttemp = ((result['word_ll'] + result['state_llp']).detach() - result['state_llq'].detach()).mean(1).sum(0).item()
    # temp1 = (log_f.detach() - result['state_llq'].detach()).mean(1)
    temp2 = (logsumexp1((result['word_ll'] + result['state_llp']).detach() -
                        result['state_llq'].detach()) - math.log(args.sample_size))
    summary['total_ppl'] += ttemp
    summary['total_ppl_mc'] += temp2.sum(0).item()
    summary['total_sent'] += bsz
    summary['total_token'] += bsz * seqlen
    summary['avg_ppl'] += ttemp / seqlen # finally divide by bsz.
    summary['avg_ppl_mc'] += temp2.sum(0).item() / seqlen
    # print(summary['avg_ppl'])
    # print(summary['total_word'])
    # print(summary['total_ppl'])
    # # print(temp1, temp2)
    # print(temp2.sum(0).item(), ttemp)

    if state_z is not None and False:
        state_z_vit = vseq_2_vit_lst_batch(state_z)

        ''' should be divided by total_sent in the final step. '''
        summary['acc_unlabeled_DICE'] += get_segment_acc_batch(result_vit,
                                                               state_z_vit)  # get_v1_acc(state_z, result_vit)
        summary['edit_distance_unlabeled'] += get_segment_ed_batch(result_vit, state_z_vit)
        summary['f_score'] += get_f1_score_batch(state_z_vit, result_vit)



    if args.posterior_reg == 1:
        summary['pr'] += result['pr'].item()
    else:
        summary['pr'] += -1

    summary['klpen'] += kl_pen



    #############################################################################

    if not args.mem_eff_flag:
        seg_result = []
        # for b in range(y.size(0)):
        #     seg_result.append(' '.join(visual_viterb(result_vit[b], en_targ[b])))
        seg_result_sample = []
        full_score_s = result['state_llq'].detach().exp()
        for it, (t1, t2) in enumerate(zip(result_vit, result['samples_vtb_style'])):
            # seg_result.append(' '.join(visual_viterb(result_vit[it], en_targ[it])) + '|||{:.5f}'.
            #                   format((score_vit[it][0]).detach().exp().data))
            # print(result_vit[it])
            temp = []
            for _, _, itt in result_vit[it]:
                temp.append(str(itt))
            seg_result.append(' '.join(en_targ[it] ) + '|||{}'.format(' '.join(temp)) + '|||{:.5f}'.
                              format((score_vit[it][0]).detach().exp().data) + '|||{}'.format(str(full_score_s[it].data)))

            # print(full_score_s[it])
            # sample_segstr = ''
            # for it2 in range(args.sample_size):
            #     sample_segstr += ' '.join(visual_viterb(t2[it2], en_targ[it])) + '|||{:.5f}'.format(full_score_s[it][it2]) + '\n'
            # seg_result_sample.append(sample_segstr)

        # for elem1, elem2 in zip(seg_result, seg_result_sample):
        #     file_handle.write(str(elem1) + '\n')
        #     # file_handle.write(str(elem2) + '\n')

        for elem1 in seg_result:
            file_handle.write(str(elem1) + '\n')
            # file_handle.write(str(elem2) + '\n')

        # summary['viterbi'] += seg_result
        # summary['sample'] += seg_result_sample
    else:
        # summary['viterbi'] += [(x, y)for (x,y) in zip(result_vit, en_targ)] #score_vit[0].detach().exp().data
        # summary['sample'] += result['state_llq'].detach().exp().data
        for elem in zip(result_vit, en_targ):
            file_handle.write( str(elem) + '\n')
    #############################################################################

    summary['vit_len'] += (x.size(1)*x.size(0)/segment_num_viterbi(result_vit)) * bsz

    return -obj.mean()

def test(epoch_num, corpora_lab, vit_filehandle):
    summary = defaultdict(float)
    summary['viterbi'] = []
    summary['sample']=[]
    summary['template_vit'] = []
    summary['template_sample'] = []

    if corpora_lab == 'valid':
        corpora = corpus.valid
        logger.info('evaluating on valid')
    elif corpora_lab == 'test':
        corpora = corpus.test
        logger.info('evaluating on test')
    else:
        corpora = corpus.train
        logger.info('evaluating on train')
    with torch.no_grad():
        for i in range(len(corpora)):
            x, pos, chk, ner, x_str = corpora[i]

            bsz, seqlen = x.size()

            if seqlen < args.L or seqlen > args.max_seqlen:
                continue


            if args.cuda:
                x = x.to(args.device)

                # combotargs = combotargs.to(args.device)
                # # if cidxs is not None:
                # #     cidxs = [tens.cuda() if tens is not None else None for tens in cidxs]
                # src = src.to(args.device)
                # x = x.to(args.device)
                # inps = inps.to(args.device)
                # fmask = fmask.to(args.device)
                # # amask = amask.to(args.device)
                # uniqfields = uniqfields.to(args.device)
                # print(x.device)
            obj = gather_stats(model, (x, pos, chk, ner, x_str), summary, args.kl_pen, epoch_num,
                               vit_filehandle, more_info=True)

            # print(len(summary['template_vit']), i + 1)
            # assert len(summary['template_vit']) == i + 1
    logger.info('DONE EVAL epoch {}:{}'.format(epoch_num+1, print_dict2(summary)))
    return summary, -summary['ELBO']/summary['#Tokens']


def train(epoch_num, vit_filehandle):

    summary = defaultdict(float)
    summary['viterbi'] = []
    summary['sample'] = []
    summary['template_vit'] = []
    summary['template_sample'] = []

    timing = defaultdict(float)

    trainperm = torch.randperm(len(corpus.train))
    # info_lst = np.random.choice(list(range(nmini_batches)), 50, replace=False)
    print('total of {}'.format(nmini_batches))
    percent = 0
    # trainperm = list(range(len(corpus.train)))
    # trainperm = [435, 2374] + trainperm
    for batch_idx in range(nmini_batches):
        # the _ should have been the labels of segments (unsupervised)
        # print(batch_idx)
        x, pos, chk, ner, x_str = corpus.train[trainperm[batch_idx]]
        # cidxs = train_cidxs[trainperm[batch_idx]] if epoch <= args.constr_tr_epochs else None
        cidx = None
        bsz, seqlen = x.size()

        if seqlen < args.L or seqlen > args.max_seqlen:
            continue

        # combotargs = make_combo_targs(locs, x, 1, nfields, corpus.ngen_types)
        # get bsz x nfields, bsz x nfields masks
        # fmask = make_masks(src, args.pad_idx, max_pool=args.max_pool)
        # uniqfields = get_uniq_fields(src, args.pad_idx)  # bsz x max_fields

        if args.cuda:
            # combotargs = combotargs.to(args.device)
            # if cidxs is not None:
            #     cidxs = [tens.cuda() if tens is not None else None for tens in cidxs]
            # src = src.to(args.device)
            x = x.to(args.device)
            # inps = inps.to(args.device)
            # fmask = fmask.to(args.device)
            # # amask = amask.to(args.device)
            # uniqfields = uniqfields.to(args.device)
            # print(x.device)
        # logger.info(src.device)
        obj = gather_stats(model, (x, pos, chk, ner, x_str), summary, args.kl_pen, epoch_num, vit_filehandle,
                           timing=timing, more_info=True)

        obj.backward()

        if args.q_max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.q_max_grad_norm)

        optimizer_p.step()
        optimizer_q.step()

        optimizer_p.zero_grad()
        optimizer_q.zero_grad()

        args.kl_pen = min(args.kl_pen + args.delta_kl, 1)

        # if batch_idx % (nmini_batches//20) == 0:
        #     str_ = ''
        #     for kk, elem  in timing.items():
        #         str_ +=  kk + ':' + str(datetime.timedelta(seconds=int(elem))) + '\t'
        #     print(str_)
        #     sys.stdout.flush()

    # args.kl_pen = min(args.kl_pen * 2, 1)
    # args.lamb = min(args.lamb * 2, args.lamb_max)
    logger.info('finished epoch {}:{} \n'.format(epoch_num+1, print_dict2(summary)))
    return summary
    # investigate_viterb(model, data_lst)


def get_template(model, data):
    if data == 'valid':
        summary, stats = test(0, 'valid')
        templates0 = summary['template_vit']
        return templates0
        # templates1 = summary['sample']

def get_template_p(model, corpora_lab):
    result = []
    correct = []
    src_ = []

    if corpora_lab == 'valid':
        corpora = corpus.valid
        logger.info('evaluating on valid')
    else:
        corpora = corpus.train
        logger.info('evaluating on train')

    with torch.no_grad():
        for i in range(len(corpora)):
            x, lab, src, locs, inps, en_src, en_targ, src_wrd2fields = corpora[i]
            cidxs = None
            seqlen, bsz = x.size()
            nfields = src.size(1)
            if seqlen < args.L or seqlen > args.max_seqlen:
                continue

            if i % 20 == 0:
                print('hi')
                sys.stdout.flush()

            combotargs = make_combo_targs(locs, x, 1, nfields, corpus.ngen_types)
            # get bsz x nfields, bsz x nfields masks
            fmask, amask = make_masks(src, args.pad_idx, max_pool=args.max_pool)
            uniqfields = get_uniq_fields(src, args.pad_idx)  # bsz x max_fields

            if args.cuda:
                combotargs = combotargs.to(args.device)
                # if cidxs is not None:
                #     cidxs = [tens.cuda() if tens is not None else None for tens in cidxs]
                src = src.to(args.device)
                x = x.to(args.device)
                inps = inps.to(args.device)
                fmask, amask = fmask.to(args.device), amask.to(args.device)
                uniqfields = uniqfields.to(args.device)
                # print(x.device)

            condi = model.encode_table(src, amask, uniqfields)
            condi['fmask'] = fmask
            condi['inps'] = inps
            condi['combotargs'] = combotargs
            condi['targs'] = combotargs[0]
            condi['src'] = src
            condi['src_wrd2fields'] = src_wrd2fields

            # if args.decoder_constraint == 'yes':
            #     detailed_src_mask = gen_detailed_src_mask(src, args.field_idx2state_idx, args.tagset_size).to(
            #         args.device)
            #     condi['detailed_src_mask'] = detailed_src_mask
            #
            # if args.encoder_constraint == 'yes':
            #     detailed_tgt_mask = gen_detailed_tgt_mask(condi, args.field_idx2state_idx, args.L, seqlen, bsz,
            #                                               args.tagset_size, args.non_field).to(args.device)
            #     condi['detailed_tgt_mask'] = detailed_tgt_mask  # (L, seqlen, bsz, K)

            result.append(model.rnn_lm.beam_sample_template(condi, bsz))

            correct += [' '.join(x) for x in en_targ]
            src_ += [' '.join(x) for x in en_src]

    epoch_num = 0
    logger.info('writing the beam search result to {}'.format(args.save_out + '.beam'))
    with open(args.save_out + '.{}_{}_beam'.format(epoch_num, corpora_lab), 'w') as f:
        for elem in result:
            print(elem, file=f)
    logger.info('writing the correct answer to {}'.format(args.save_out + '.corr'))
    with open(args.save_out + '.{}_{}_corr'.format(epoch_num, corpora_lab), 'w') as f:
        for elem in correct:
            print(elem, file=f)
    logger.info('writing the source to {}'.format(args.save_out + '.src'))
    with open(args.save_out + '.{}_{}_src'.format(epoch_num, corpora_lab), 'w') as f:
        for elem in src_:
            print(elem, file=f)

    return result


parser = argparse.ArgumentParser(description='')
parser.add_argument('--save', type=str, default='', help='path to save the final model')
parser.add_argument('--save_out', type=str, default='save_out', help='path to save the output')
parser.add_argument('--min_epochs', default=6, type=int, help='do not decay learning rate for at least this many epochs')
parser.add_argument('--load', type=str, default='', help='path to saved model')
parser.add_argument('--test', action='store_true', help='use test data')
parser.add_argument('--thresh', type=int, default=5, help='prune if occurs <= thresh')
parser.add_argument('--max_mbs_per_epoch', type=int, default=35000, help='max minibatches per epoch')
parser.add_argument('--self_attn', type=str, default='no', help='the unconditional generation task if we will use attn')

parser.add_argument('--layers', type=int, default=1, help='num rnn layers')
parser.add_argument('--A_dim', type=int, default=64,
                    help='dim of factors if factoring transition matrix')
parser.add_argument('--cond_A_dim', type=int, default=32,
                    help='dim of factors if factoring transition matrix')
parser.add_argument('--smaller_cond_dim', type=int, default=64,
                    help='dim of thing we feed into linear to get transitions')
parser.add_argument('--yes_self_trans', action='store_true', help='')
parser.add_argument('--mlpinp', action='store_true', help='')
parser.add_argument('--mlp_sz_mult', type=int, default=2, help='mlp hidsz is this x emb_size')
parser.add_argument('--max_pool', action='store_true', help='for word-fields')

parser.add_argument('--constr_tr_epochs', type=int, default=100, help='')
parser.add_argument('--no_ar_epochs', type=int, default=100, help='')

parser.add_argument('--word_ar', action='store_true', help='')
parser.add_argument('--ar_after_decay', action='store_true', help='')
parser.add_argument('--no_ar_for_vit', action='store_true', help='')
parser.add_argument('--fine_tune', action='store_true', help='only train ar rnn')

parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--emb_drop', action='store_true', help='dropout on embeddings')
parser.add_argument('--lse_obj', action='store_true', help='')
parser.add_argument('--sep_attn', action='store_true', help='')
parser.add_argument('--max_seqlen', type=int, default=70, help='')

parser.add_argument('--K', type=int, default=10, help='number of states')
parser.add_argument('--Kmul', type=int, default=1, help='number of states multiplier')
parser.add_argument('--L', type=int, default=8, help='max segment length')
parser.add_argument('--unif_lenps', action='store_true', help='')
parser.add_argument('--one_rnn', action='store_true', help='')

parser.add_argument('--initrange', type=float, default=0.05, help='uniform init interval')
parser.add_argument('--optim', type=str, default="adam", help='optimization algorithm')
parser.add_argument('--onmt_decay', action='store_true', help='')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping') # prev = 5
parser.add_argument('--verbose', action='store_true', help='')

parser.add_argument('--ntemplates', type=int, default=200, help='num templates for gen')
parser.add_argument('--beamsz', type=int, default=1, help='')


parser.add_argument('--data_mode', type=str, default='real', help='1. toy, 2.real, 3. random')
parser.add_argument('--data', type=str, default='/Users/xiangli/Desktop/Sasha/FSA-RNN/data/e2e_aligned/', help='path to data dir')
# parser.add_argument('--data', type=str, default='/Users/xiangli/Desktop/Sasha/FSA-RNN/data/wb_aligned/', help='path to data dir')
parser.add_argument('--epochs', type=int, default=60, help='upper epoch limit')
parser.add_argument('--bsz', type=int, default=16, help='batch size')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='minibatches to wait before logging training status')

parser.add_argument('--embedding_dim', type=int, default=400, help='')
parser.add_argument('--f_dim', type=int, default=50, help='')
parser.add_argument('--i_dim', type=int, default=5, help='')
parser.add_argument('--s_dim', type=int, default=100, help='')
parser.add_argument('--table_hidden_dim', type=int, default=300, help='') # or 500
parser.add_argument('--hidden_dim', type=int, default=500, help='')
parser.add_argument('--tagset_size', type=int, default=20, help='')
parser.add_argument('--q_dim', type=int, default=50, help='')
parser.add_argument('--dual_attn', type=str, default='no', help='')

parser.add_argument('--lr_q', type=float, default=0.001, help='')
parser.add_argument('--lr_p', type=float, default=0.0005, help='') # 20 if for SGD.
parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay')
parser.add_argument('--lr_action', type=float, default=0.02, help='')
parser.add_argument('--lr', type=float, default=0.07, help='initial learning rate')

parser.add_argument('--conditional_dim', type=int, default=30, help='')
parser.add_argument('--train_q_epochs', type=int, default=5, help='')
parser.add_argument('--max_grad_norm', default=5, type=float, help='gradient clipping parameter')
parser.add_argument('--q_max_grad_norm', default=1, type=float, help='gradient clipping parameter for q')
parser.add_argument('--group_param', default='variational', type=str, help='how we group the optimization of the parameters')


parser.add_argument('--epoch', type=int, default=30, help='')
parser.add_argument('--full_independence', type=int, default=3, help='# -1 if fully independence, 1 if fully RNN, ')
parser.add_argument('--optim_algo', type=str, default="1", help='')
parser.add_argument('--epoch_inner', type=int, default=30, help='')
parser.add_argument('--sample_size', type=int, default=4, help='')
parser.add_argument('--mi_sample_size', type=int, default=5, help='')
parser.add_argument('--supervised_training', default='no', type=str, help='supervised training')
parser.add_argument('--train_data_path', default='temp_toy', type=str, help='supervised training')
parser.add_argument('--dev_data_path', default='temp_toy_dev', type=str, help='supervised training')
parser.add_argument('--mode', default='FULL', type=str, help='supervised training')
parser.add_argument('--option', default='beam', type=str, help='supervised training')
parser.add_argument('--lamb_init', type=float, default=0.3, help='lambda value')
parser.add_argument('--lamb_max', type=float, default=0.3, help='lambda value')
parser.add_argument('--batch_size', type=int, default=10, help='size of the batch')
parser.add_argument('--decoder', type=str, default='crf', help='gen or crf as the type of q distribution')
parser.add_argument('--decay', default=0.5, type=float, help='')
parser.add_argument('--prior', default='ar', type=str, help='')
parser.add_argument('--pr_coef', default=1., type=float, help='')
parser.add_argument('--ph', default='', type=str, help='')
parser.add_argument('--posterior_reg', default=1, type=int, help='1 means pr is turned on, 0 means pr is turned off.')
parser.add_argument('--weight_decay', default=0.0, type=float, help='')
parser.add_argument('--trans_unif', default='yes', type=str, help='whether we use a uniform transition '
                                                                  'distribution in the q model ')
parser.add_argument('--pr_reg_style', default='phrase', type=str, help='the style of pr.')
parser.add_argument('--use_bert', default='no', type=str, help='whether we will use bert as a pre-training, yes/no')
parser.add_argument('--use_elmo', default='no', type=str, help='whether we will use elmo as a pre-training, yes/no')
parser.add_argument('--elmo_style', default=1, type=int, help='elmo style choices 1: MLP. 2, or 3:ELMo-2')

parser.add_argument('--decoder_constraint', default='no', type=str, help='whether we precompute a decoder mask. '
                                                                          'to force the decoder copy attention '
                                                                          'to only related to state. ')
parser.add_argument('--encoder_constraint', default='yes', type=str, help='whether we precompute a  encoder mask. '
                                                                          'whether we force the encoder copy attention '
                                                                          'to only related to state. ')
parser.add_argument('--hard_code', default='no', type=str, help='whether we hard code rules in the encoder template.')
parser.add_argument('--beam_size', default=1, type=int, help='the K-best we use in beam search. ')
parser.add_argument('--additional_attn', default='no', type=str, help='yes or no. ')
parser.add_argument('--span_repr', default='lstmmin', type=str, help='lstm-minus or peter')
parser.add_argument('--multi_task', default='no', type=str, help='yes or no')




def get_optim(args, parameters, lr, weight_decay):
    if args.optim == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':

    try:
        args = parser.parse_args()
        # print(args)
        _start()

        args.device = None
        if torch.cuda.is_available():
            if not args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with -cuda")
                args.device = torch.device('cpu')
            else:
                torch.cuda.manual_seed(args.seed)
                args.device = torch.device('cuda')
        else:
            args.device = torch.device('cpu')


        if args.data_mode == 'random':
            ''' toy data generation using a full conditional RNN with random initialization '''
            data_lst = []
            data_generator = RNNLM(args)
            flag_supervised = True
            if flag_supervised:
                total_logp = 0
                total_sent = 0
                for idx in range(200):
                    conditional_ = torch.randn([1, 1, args.conditional_dim])
                    sent, state, logp_w, logp_s = data_generator.generate_with_state(conditional_)
                    example_logp = logp_w + logp_s
                    total_logp += example_logp
                    total_sent += 1
                    # print(conditional_, sent)
                    # print('*'*20 + str(idx) + '*'*(30))
                    data_lst.append((conditional_, sent, state))
                perplexity = (-total_logp / total_sent).exp()
                print(perplexity)
                logp = -total_logp / total_sent
                print('the generation logP={}'.format(logp.item()))
            else:
                for idx in range(3):
                    conditional_ = torch.randn([1, 1, args.conditional_dim])
                    sent = data_generator.generate(conditional_)
                    print(conditional_, sent)
                    print('*' * 20 + str(idx) + '*' * (30))
                    data_lst.append((conditional_, sent))

        elif args.data_mode == 'toy':

            corpus = Data_Loader_toy(args.data, args.bsz, args.L, args.tagset_size, test=args.test)

            args.vocab_size = len(corpus.vocab2idx)
            # args.tagset_size = len(corpus.state_vocab)
            args.pad_idx = corpus.vocab2idx["<pad>"]
            args.eos_idx = corpus.vocab2idx["<eos>"]
            args.bos_idx = corpus.vocab2idx["<bos>"]
        elif args.data_mode == 'real':
            # Load data
            # corpus = Data_Loader_Chunk(args.data, args.bsz, max_len=args.max_seqlen, thresh=args.thresh, test=args.test)
            corpus = Data_Loader_ptb(args.data, args.bsz, args.L, args.tagset_size, max_len=args.max_seqlen, thresh=args.thresh, test=args.test)
            args.pad_idx = corpus.vocab2idx["<pad>"]
            args.unk_idx = corpus.vocab2idx["<unk>"]
            args.eos_idx = corpus.vocab2idx["<eos>"]
            args.bos_idx = corpus.vocab2idx["<bos>"]
            args.vocab_size = len(corpus.vocab2idx)

            temp_field = [0, 0, 0, 0]
            args.temp_field = torch.LongTensor(temp_field).to(args.device)
            print(args.temp_field)
            # args.labeled_states = corpus.field_names2idx
            args.field_idx2state_idx = {}



        else:
            print('invalid data mode')

        if args.mode == 'RNN' and args.option == 'train':
            ''' comparing to the supervised version'''
            model = Chunking_RNN(args)
            model.to(args.device)
            optimizer_p = get_optim(args, model.parameters(), args.lr_p, args.weight_decay)
            # optimizer_p = torch.optim.Adam(model.parameters(), lr=args.lr_p, weight_decay=args.weight_decay)
            best_val_ppl = 1e10
            nmini_batches = min(len(corpus.train), args.max_mbs_per_epoch)
            for e in range(args.epoch):
                model.train()
                summary = defaultdict(float)
                trainperm = torch.randperm(len(corpus.train))
                percent = 0
                for batch_idx in range(nmini_batches):

                    x, pos, chk, ner, x_str = corpus.train[trainperm[batch_idx]]
                    # cidxs = train_cidxs[trainperm[batch_idx]] if epoch <= args.constr_tr_epochs else None
                    cidx = None
                    bsz, seqlen = x.size()

                    if seqlen < args.L or seqlen > args.max_seqlen:
                        continue
                    if args.cuda:
                        x = x.to(args.device)
                    obj = gather_rnn_stats(model, (x, pos, chk, ner, x_str), summary)
                    obj.backward()

                    #
                    # obj = gather_rnn_stats(model, (
                    # (src, uniqfields, inps, fmask, amask, combotargs, en_src, en_targ, src_wrd2fields), x, lab), summary)
                    # obj.backward()

                    if args.q_max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.q_max_grad_norm)
                    optimizer_p.step()
                    optimizer_p.zero_grad()
                    # if batch_idx % (nmini_batches//4) == 0:
                    #     logger.info('epoch {}: {}'.format(epoch_num+percent, print_dict(summary)))
                    #     percent += 1 / 5
                    # show the resulting viterbi segmentation.
                logger.info('finished epoch {}:{} \n'.format(e, print_dict(summary)))
                ''' beam search result '''
                if False and e > 0 and e % 5 == 0:
                    beam_corpus(model, e, 'valid')
                    beam_corpus(model, e, 'train')

                model.eval()
                summary, val_ppl = test_rnn(model, e, 'valid')

                print(val_ppl, best_val_ppl)
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    logger.info("saving checkpoint to {}".format(args.save + '_{}'.format(e)))
                    state = {"opt": args, "state_dict": model.cpu().state_dict(),
                             "lr": args.lr, "dict": corpus.vocab2idx}
                    torch.save(state, args.save + '_{}'.format(e))
                    model.to(args.device)
                    duration = 0
                else:
                    if e > args.min_epochs:
                        args.lr_p = args.decay * args.lr_p
                        for param_group in optimizer_p.param_groups:
                            param_group['lr'] = args.lr_p
                        logger.info('decay learning rate: lr_p={}'.format(args.lr_p))


                # if decay == 1:
                #     args.lr_p = args.decay * args.lr_p
                #     for param_group in optimizer_p.param_groups:
                #         param_group['lr'] = args.lr_p
                #     logger.info('decay learning rate: lr_p={}'.format(args.lr_p))
                #     decay=0
                #     duration = 0

                if args.lr_p < 0.0001:
                    logger.info("learning rate small -> stop")
                    break
            logger.info("saving checkpoint to {}".format(args.save))
            state = {"opt": args, "state_dict": model.cpu().state_dict(),
                 "lr": args.lr, "dict": corpus.vocab2idx}
            torch.save(state, args.save)
            model.to(args.device)
            model.eval()
            # beam_corpus(model, 'final', 'valid')
            # beam_corpus(model, 'final', 'train')

        if args.mode == 'RNN' and args.option == 'beam':
            if len(args.load) > 0:
                saved_stuff = torch.load(args.load)
                saved_args, saved_state = saved_stuff["opt"], saved_stuff["state_dict"]
                for k, v in args.__dict__.items():
                    if k not in saved_args.__dict__:
                        saved_args.__dict__[k] = v
                model = Chunking_RNN(saved_args)
                # for some reason selfmask breaks load_state
                model.load_state_dict(saved_state, strict=False)
                args.pad_idx = corpus.vocab2idx["<pad>"]
                if args.fine_tune:
                    for name, param in model.named_parameters():
                        if name in saved_state:
                            param.requires_grad = False
            model.to(args.device)
            model.eval()
            summary, val_ppl = test_rnn(model, 1, 'test')
            # beam_corpus(model, 'final', 'valid')
            # beam_corpus(model, 'final', 'train')


        ''' testing for the supervised version of the HSMM '''
        if args.mode == 'HSMM':
            flag_supervised = True
            model_base3 = HSMM(args)
            optimizer3 = torch.optim.Adam(model_base3.parameters(), lr=args.lr)
            for e in range(50):
                summary_base3 = defaultdict(float)
                whole_obj_lst = []
                for idx, item in enumerate(data_lst):
                    if flag_supervised:
                        x, y, state_z = item
                        y = torch.LongTensor(y).view(1, -1)
                        state_z_vit, state_z_lst = vseq_2_vit_lst(state_z)
                        state_z_vit = [state_z_vit]
                        state_z_lst = [[state_z_lst]]
                        # state_z = torch.LongTensor(state_z).view(1, -1)
                        # state_z = model_base3.convert_state_(state_z)
                    else:
                        x, y = item
                        y = torch.LongTensor(y).view(1, -1)

                    dict_computed = model_base3.get_weights(y, x)
                    Z, entropy = model_base3.get_entr(dict_computed)

                    sample_score = model_base3.get_score(state_z_lst, dict_computed)
                    target = [torch.stack(samples) for samples in sample_score]
                    target = torch.stack(target, dim=0)
                    bsz, num_sample = target.shape
                    state_llq = (target - Z.expand(bsz, num_sample))

                    result_vit = model_base3.viterbi(dict_computed)
                    # state_llq = torch.stack(model_base3.get_score(state_z_lst, dict_computed))
                    # state_llq = state_llq - Z

                    # print(state_llq)


                    whole_obj_lst.append(-(state_llq).mean())
                    # print(whole_obj_lst[-1])

                    summary_base3['logp(z|y,x)'] += -whole_obj_lst[-1].item()
                    summary_base3['entr'] += entropy.item()
                    summary_base3['sent_num'] += 1
                    summary_base3['acc_unlabeled_DICE'] += get_segment_acc(result_vit[0],
                                                                     state_z_vit[0])  # get_v1_acc(state_z, result_vit)
                    summary_base3['edit_distance_unlabeled'] += get_segment_ed(result_vit[0], state_z_vit[0])
                    summary_base3['acc_labeled'] += get_acc_seg(result_vit[0], state_z_vit[0])

                    if idx > 0 and idx % 5 == 0:
                        target = torch.stack(whole_obj_lst).mean()
                        target.backward()
                        optimizer3.step()
                        optimizer3.zero_grad()
                        whole_obj_lst = []
                print(print_dict(summary_base3))

                ''' testing by viterbi segmentation. '''
                avg_acc = 0
                avg_ed = 0
                lab_acc = 0
                for idx, item in enumerate(data_lst):
                    if flag_supervised:
                        x, y, state_z = item
                        y = torch.LongTensor(y).view(1, -1)
                        # state_z = torch.LongTensor(state_z).view(1, -1)
                        state_z_vit = vseq_2_vvit(state_z)
                        # state_z = model_base3.convert_state_(state_z)
                    else:
                        x, y = item
                        y = torch.LongTensor(y).view(1, -1)
                    dict_computed = model_base3.get_weights(y, x)
                    result_vit = model_base3.viterbi(dict_computed)
                    avg_acc += get_segment_acc(result_vit[0], state_z_vit)
                    avg_ed += get_segment_ed(result_vit[0], state_z_vit)
                    lab_acc += get_acc_seg(result_vit[0], state_z_vit)

                print('unlabeled dice = {}, unlabeled ed = {}, labeled accuracy '
                      'is {}'.format(avg_acc/len(data_lst), avg_ed/len(data_lst), lab_acc/ len(data_lst)))
                    # print('accuracy is {}'.format(get_acc_(result, state_z)))

        ''' testing for the UNsupervised version of the HSMM '''
        print(args.mode )
        if args.mode == 'HSMM_gen':
            flag_supervised = True
            model_base4 = HSMM_generative(args)
            optimizer4 = torch.optim.Adam(model_base4.parameters(), lr=args.lr)
            for e in range(50):
                summary_base4 = defaultdict(float)
                whole_obj_lst = []
                for idx, item in enumerate(data_lst):
                    if flag_supervised:
                        x, y, state_z = item
                        y = torch.LongTensor(y).view(1, -1)
                        state_z_vit, state_z_lst = vseq_2_vit_lst(state_z)
                        state_z_vit = [state_z_vit]
                        state_z_lst = [[state_z_lst]]
                        # state_z = torch.LongTensor(state_z).view(1, -1)
                        # state_z = model_base3.convert_state_(state_z)
                    else:
                        x, y = item
                        y = torch.LongTensor(y).view(1, -1)

                    dict_computed = model_base4.get_weights(y, x)
                    Z, entropy = model_base4.get_entr(dict_computed)
                    # print(Z.exp())
                    # sample_score = model_base4.get_score(state_z_lst, dict_computed)
                    # target = [torch.stack(samples) for samples in sample_score]
                    # target = torch.stack(target, dim=0)
                    # bsz, num_sample = target.shape
                    state_llq = Z
                    result_vit = model_base4.viterbi(dict_computed)
                    # state_llq = torch.stack(model_base3.get_score(state_z_lst, dict_computed))
                    # state_llq = state_llq - Z

                    # print(state_llq)

                    whole_obj_lst.append(-(state_llq).mean())
                    # print(whole_obj_lst[-1])

                    summary_base4['logp(y|x)'] += -whole_obj_lst[-1].item()
                    summary_base4['entr'] += entropy.item()
                    summary_base4['sent_num'] += 1
                    summary_base4['acc_unlabeled_DICE'] += get_segment_acc(result_vit[0],
                                                                           state_z_vit[
                                                                               0])  # get_v1_acc(state_z, result_vit)
                    summary_base4['edit_distance_unlabeled'] += get_segment_ed(result_vit[0], state_z_vit[0])
                    summary_base4['acc_labeled'] += get_acc_seg(result_vit[0], state_z_vit[0])

                    if idx % 5 == 0:
                        target = torch.stack(whole_obj_lst).mean()
                        target.backward()
                        optimizer4.step()
                        optimizer4.zero_grad()
                        whole_obj_lst = []
                print(print_dict(summary_base4))

                ''' testing by viterbi segmentation. '''

                investigate_viterb_hsmm(model_base4, data_lst)

        if args.mode == 'FULL':

            if len(args.load) > 0:
                saved_stuff = torch.load(args.load)
                saved_args, saved_state = saved_stuff["opt"], saved_stuff["state_dict"]
                saved_args.device = args.device
                saved_args.option = args.option
                for k, v in args.__dict__.items():
                    if k not in saved_args.__dict__:
                        saved_args.__dict__[k] = v
                model = Chunking_Model(saved_args).to(args.device)
                print('loaded model to the device', args.device)
                # for some reason selfmask breaks load_state
                model.load_state_dict(saved_state, strict=False)
                args.pad_idx = corpus.vocab2idx["<pad>"]
                if args.fine_tune:
                    for name, param in model.named_parameters():
                        if name in saved_state:
                            param.requires_grad = False
            else:
                model = Chunking_Model(args)
                model.to(args.device)
                args.pad_idx = corpus.vocab2idx["<pad>"]

                if args.group_param == 'variational':
                    # TODO: structural chocie, we moved the embeddings to the p model.
                    # optimizer_p = torch.optim.Adam(list(model.rnn_lm.parameters()) + list(model.word_vecs.parameters()), lr=args.lr_p, weight_decay=args.weight_decay)
                    optimizer_p = torch.optim.Adam(model.rnn_lm.parameters(), lr=args.lr_p, weight_decay=args.weight_decay)
                    optimizer_q = torch.optim.Adam(model.hsmm_crf.parameters(), lr=args.lr_q, weight_decay=args.weight_decay)

                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

            logger.info('start-training!')

            if args.option == 'train':
                args.mem_eff_flag = False
                args.kl_pen = 0.1
                duration = 0
                nmini_batches = min(len(corpus.train), args.max_mbs_per_epoch)
                args.delta_kl = 1 / nmini_batches
                best_val_ppl = 1e10
                decay = 0
                for e in range(args.epoch):
                    logger.info('epoch # {}'.format(e))
                    model.train()
                    if e > args.train_q_epochs:
                        logger.info('STOP training the inference network.')
                        args.q_lr = 0.
                        for param_group in optimizer_q.param_groups:
                            param_group['lr'] = args.q_lr

                    f_tr = open(args.save_out+'.{}_tr'.format(e), 'w+')
                    summary = train(e, f_tr)
                    f_tr.close()

                    # with open(args.save_out+'.{}_tr'.format(e), 'w') as f:
                    #     if not mem_eff_flag:
                    #         for elem1,elem2 in zip(summary['viterbi'], summary['sample']):
                    #             print(elem1, file=f)
                    #             print(elem2, file=f)
                    #     else:
                    #         for elem1 in summary['viterbi']:
                    #             print(elem1, file=f)

                    if e > 0 and e % 15 == 0 :
                        model.eval()
                        # beam_corpus(model, e, 'valid')


                    ''' eval on dev data '''
                    # LISA
                    if True:
                        model.eval()
                        f_dev = open(args.save_out+'.{}_dev'.format(e), 'w+')
                        summary, val_ppl = test(e, 'valid', f_dev)
                        f_dev.close()

                    else:
                        val_ppl = 1e10


                    print(best_val_ppl, val_ppl)
                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                        # LISA 233
                        logger.info("saving checkpoint to {}".format(args.save + '_{}'.format(e)))
                        state = {"opt": args, "state_dict": model.cpu().state_dict(),
                                 "lr": args.lr, "dict": corpus.vocab2idx}
                        torch.save(state, args.save + '_{}'.format(e))
                        model.to(args.device)
                        logger.info('finsihed saving the model ;)')

                        decay=0
                        duration = 0

                    else:
                        if e > args.min_epochs:
                            decay = 1
                            duration = duration + 1

                    if True:
                        # LISA 75
                        if decay == 1:
                            # args.lr = args.decay * args.lr
                            args.lr_q = args.decay * args.lr_q
                            args.lr_p = args.decay * args.lr_p
                            # args.action_lr = args.decay * args.action_lr
                            for param_group in optimizer_p.param_groups:
                                param_group['lr'] = args.lr_p
                            for param_group in optimizer_q.param_groups:
                                param_group['lr'] = args.lr_q
                            logger.info('decay learning rate: lr_q={}, lr_p={}'.format(args.lr_q, args.lr_p))
                            decay=0
                            duration = 0

                    if args.lr_p < 0.00001:
                        logger.info("learning rate small -> stop")
                        break

                logger.info("saving checkpoint to {}".format(args.save))
                state = {"opt": args, "state_dict": model.cpu().state_dict(),
                         "lr": args.lr, "dict": corpus.vocab2idx}
                torch.save(state, args.save + '_{}'.format(e))
                model.to(args.device)

                model.eval()
                # beam_corpus(model, 'final', 'valid')
                # beam_corpus(model, 'final', 'train')


            elif args.option == 'beam':
                args.mem_eff_flag = False

                with torch.no_grad():
                    args.kl_pen = 1
                    args.posterior_reg = 0
                    model.posterior_reg = 0
                    model.eval()
                    if not args.test:
                        f_dev = open(args.save_out + '.{}_dev'.format(1), 'w+')
                        summary, val_ppl = test(1, 'valid', f_dev)
                        f_dev.close()
                        # logger.info('EVAL the beam search result')
                        # beam_corpus(model, 'eval', 'valid')
                    else:
                        logger.info('TEST the beam search result')
                        f_dev = open(args.save_out + '.{}_dev'.format(1), 'w+')
                        summary, val_ppl = test(1, 'test', f_dev)
                        f_dev.close()

                        # beam_corpus(model, 'eval', 'test_single')
                        # beam_corpus(model, 'eval', 'test')


            elif args.option == 'controlled':
                args.kl_pen = 1
                args.posterior_reg = 0
                model.posterior_reg = 0
                model.eval()
                if not args.test:
                    # first design some resulting template by sampling.
                    templates = get_template(model, 'valid')
                    logger.info('EVAL the beam search result')
                    beam_corpus(model, 'eval', 'valid', controlled=True, template=templates)
                else:
                    logger.info('TEST the beam search result')
                    # beam_corpus(model, 'eval', 'test_single')
                    beam_corpus(model, 'eval', 'test')

            elif args.option == 'induce_template_p':
                args.kl_pen = 1
                args.posterior_reg = 0
                model.posterior_reg = 0
                model.eval()
                if not args.test:
                    # first design some resulting template by sampling.
                    templates = get_template_p(model, 'valid')
                    logger.info('EVAL the beam search result')


            elif args.option == 'elmo_pre':
                dataset_ = corpus.train
                print(len(dataset_))

                for idx, batch_idx in range(len(dataset_)):
                    # the _ should have been the labels of segments (unsupervised)
                    x, lab, src, locs, inps, en_src, en_targ, src_wrd2fields = dataset_[batch_idx]
                    # load ELMo batched.
                    second_layer = model.hsmm_crf.pre_save_elmo(en_targ)



        _finish()
    except Exception as err:
        _error_break()
        print(err)
        10/0
        sys.stdout.flush()
