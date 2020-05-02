#!/usr/bin/env python3
import numpy as np
import itertools
import random
import math
from collections import defaultdict, Counter
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import f1_score
import torch.nn as nn
# from allennlp.training.metrics.bleu import BLEU
neginf = -1e38

def gen_detailed_src_mask(src, dict_label, tagset_size):
    '''
        This helps with regularizing the decoder side.
        In this case, we are trying to enhance the element of the
    :param src: the source side. (bsz * nfields, nfeats)
    :param dict_label: the dict used for field labels to states.
    :return: binary (bsz * states * nfields)
    '''
    bsz, nfields, nfeats = src.size()
    result = torch.zeros(bsz, tagset_size, nfields)
    for b in range(bsz):
        for key, val in dict_label.items():
            result[b, val] = (src[b,:,0] == key) # 1 stands for good and 0 stands for not good.


    # mask our the 0s.
    result[result == 0] = neginf
    result[result == 1] = 0

    # for b in range(bsz):
    #     print(dict_label)
    #     print(result[b])
    #     print(src[b])
    #     print('lolol')

    return result

def gen_detailed_tgt_mask(condi, dict_label, L, seqlen, bsz, K, non_field, pr_reg_style):
    if pr_reg_style == 'swap':
        return gen_detailed_tgt_mask2(condi, dict_label, L, seqlen, bsz, K, non_field)
    elif pr_reg_style == 'phrase':
        return gen_detailed_tgt_mask2(condi, dict_label, L, seqlen, bsz, K, non_field)
    elif pr_reg_style == 'wb:entr':
        return gen_detailed_tgt_mask3(condi, dict_label, L, seqlen, bsz, K, non_field)
    elif pr_reg_style == 'wb:soft':
        return gen_detailed_tgt_mask1(condi, dict_label, L, seqlen, bsz, K, non_field)
    elif pr_reg_style == 'wb:hard':
        return
    else:
        print('unknow posterior reg style')
        return

def gen_detailed_tgt_mask3(condi, dict_label, L, seqlen, bsz, K, non_field):
    '''
        This helps with regulairzing the encoder side.
    :return: a penality mask of shape (L, seqlen, bsz, K)
    '''
    mask = torch.LongTensor(L, seqlen, bsz).fill_(0)
    for l in range(L):
        mask[l, :seqlen - l] = condi['inps'][l:, :, 0, 1]
    mask[mask == non_field] = 0
    pr_mask = torch.LongTensor(L, seqlen, bsz).fill_(0)

    for b in range(bsz):
        for l in range(L):
            for e in range(seqlen):
                temp_unq = mask[:l + 1, e, b].unique()
                ''' penalizing a boundary. '''
                if temp_unq.size(0) == 1 and 0 in temp_unq:
                    # all aux words -> penalize all labeled states.
                    pass
                elif (temp_unq.size(0) == 1 and 0 not in temp_unq) or (temp_unq.size(0) == 2 and 0 in temp_unq):
                    #  have some field words -> penalize all labeled states that does not relate to the field, and also
                    # all the unlabeled states.
                    pr_mask[l, e, b] = 3 # this means that the candidate is good and pure and could be tested against
                                         # the entropy term -> want low entropy here.
                else:
                    # penalizing all states from breaking the convension segment boundary.
                    pr_mask[l, e, b] = 10 # should be penalized for being a bad state.

                # not cover the complete segment from the left
                if e > 0 and mask[0, e - 1, b].item() == mask[0, e, b].item() and mask[0, e - 1, b].item() != 0:
                    pr_mask[l, e, b] = 10
                # not cover the complete segment from the right
                if e + l < seqlen - 1 and mask[0, e + l + 1, b].item() == mask[0, e + l, b].item() and mask[0, e + l + 1, b].item() != 0:
                    pr_mask[l, e, b] = 10

    return pr_mask

def gen_detailed_tgt_mask1(condi, dict_label, L, seqlen, bsz, K, non_field):
    '''
        This helps with regulairzing the encoder side.
    :return: a penality mask of shape (L, seqlen, bsz, K)
    '''
    mask = torch.LongTensor(L, seqlen, bsz).fill_(0)
    for l in range(L):
        mask[l, :seqlen - l] = condi['inps'][l:, :, 0, 1]
    mask[mask == non_field] = 0
    pr_mask = torch.LongTensor(L, seqlen, bsz).fill_(0)



    for b in range(bsz):
        for l in range(L):
            for e in range(seqlen):
                temp_unq = mask[:l + 1, e, b].unique()
                ''' penalizing a boundary. '''
                if temp_unq.size(0) == 1 and 0 in temp_unq:
                    # all aux words -> penalize all labeled states.
                    pass
                elif (temp_unq.size(0) == 1 and 0 not in temp_unq) or (temp_unq.size(0) == 2 and 0 in temp_unq):
                    #  have some field words -> penalize all labeled states that does not relate to the field, and also
                    # all the unlabeled states.
                    # find the temp other component.
                    temp = [x for x in temp_unq if x != 0][0]
                    pr_mask[l, e, b] = temp # this means that the candidate is good and pure and could be tested against
                                            # the entropy term -> want low entropy here.
                else:
                    # penalizing all states from breaking the convension segment boundary.
                    pr_mask[l, e, b] = -1 # should be penalized for being a bad state.

                # not cover the complete segment from the left
                if e > 0 and mask[0, e - 1, b].item() == mask[0, e, b].item() and mask[0, e - 1, b].item() != 0:
                    pr_mask[l, e, b] = -1
                # not cover the complete segment from the right
                if e + l < seqlen - 1 and mask[0, e + l + 1, b].item() == mask[0, e + l, b].item() and mask[0, e + l + 1, b].item() != 0:
                    pr_mask[l, e, b] = -1

    return pr_mask


def make_bwd_idxs_pre(L, T, constrs):
    """
    for use w/ bwd alg.
    constrs are a bsz-length list of lists of (start, end, label) 0-indexed tups
    """
    cidxs = [set() for t in range(T)]
    bsz = len(constrs)
    for b in range(bsz):
        for tup in constrs[b]:
            if len(tup) == 2:
                start, end = tup
            else:
                start, end = tup[0], tup[1]
            clen = end - start
            steps_fwd = min(L, T-start)
            # for first thing only allow segment length
            cidxs[start].update([l*bsz + b for l in range(steps_fwd) if l+1 != clen])

            # now disallow everything for everything else in the segment
            for i in range(start+1, end):
                steps_fwd = min(L, T-i)
                cidxs[i].update([l*bsz + b for l in range(steps_fwd)])

            # now disallow things w/in L of the start
            for i in range(max(start-L+1, 0), start):
                steps_fwd = min(L, T-i)
                cidxs[i].update([l*bsz + b for l in range(steps_fwd) if i+l >= start])

    oi_cidxs = [None]
    oi_cidxs.extend([torch.LongTensor(list(idxs)) if len(idxs) > 0 else None for idxs in cidxs])
    return oi_cidxs

def make_bwd_idxs_pre2(L, T, K, constrs):
    """
    for use w/ bwd alg.
    constrs are a bsz-length list of lists of (start, end, label) 0-indexed tups
    """

    bsz = len(constrs)
    pr_mask = torch.LongTensor(bsz, T, L).fill_(0)
    # pr_mask_lab = torch.LongTensor(L, T, bsz, K).fill_(0)
    cidxs = [set() for t in range(T)]
    for b in range(bsz):
        for tup in constrs[b]:
            if len(tup) == 2:
                start, end = tup
            else:
                start, end, lab = tup
            clen = end - start
            # print(lab, b, start, clen)
            if not clen > L:
                pr_mask[b, start, clen-1] = lab+1
                # assert lab < 68
                # pr_mask_lab[clen-1, start, b, lab] = 1
                # print(lab, K)
            else:
                pr_mask[b, start, L-1] = lab+1
                # assert lab < 68
                # pr_mask_lab[L - 1, start, b, lab] = 1
            steps_fwd = min(L, T-start)
            # for first thing only allow segment length
            cidxs[start].update([l*bsz + b for l in range(steps_fwd) if l+1 != clen])
            # now disallow everything for everything else in the segment
            for i in range(start+1, end):
                steps_fwd = min(L, T-i)
                cidxs[i].update([l*bsz + b for l in range(steps_fwd)])
                # print('lol', steps_fwd)
                # print(i, [l*bsz + b for l in range(steps_fwd)])

            # now disallow things w/in L of the start
            for i in range(max(start-L+1, 0), start):
                steps_fwd = min(L, T-i)
                cidxs[i].update([l*bsz + b for l in range(steps_fwd) if i+l >= start])
                # print('here', steps_fwd)
                # print(i, [l*bsz + b for l in range(steps_fwd) if i+l >= start])

    oi_cidxs = []
    oi_cidxs.extend([torch.LongTensor(list(idxs)) if len(idxs) > 0 else None for idxs in cidxs])

    pr_mask = pr_mask.permute([1, 2, 0]).contiguous().view(T, -1)
    for ii, idx in enumerate(oi_cidxs):
        if idx is not None:
            pr_mask[ii, idx] = -1


    return pr_mask.view(T, L, bsz).transpose(0,1)


def gen_detailed_tgt_mask_pre(inps, L, non_field, max_seqlen):
    '''
        This helps with regulairzing the encoder side.
    :return: a penality mask of shape (L, seqlen, bsz, K)
    '''
    seqlen, bsz, _, _ = inps.shape
    if seqlen < L or seqlen > max_seqlen:
        return None
    mask = torch.LongTensor(L, seqlen, bsz).fill_(0)
    for l in range(L):
        mask[l, :seqlen - l] = inps[l:, :, 0, 1]
    mask[mask == non_field] = 0
    pr_mask = torch.LongTensor(L, seqlen, bsz).fill_(0)

    for l in range(L):
        for e in range(seqlen):
            for b in range(bsz):
                temp_unq = mask[:l + 1, e, b].unique()
                if temp_unq.size(0) == 1 and 0 in temp_unq:
                    # all aux words -> penalize all labeled states.
                    pass
                elif (temp_unq.size(0) == 1 and 0 not in temp_unq) or (temp_unq.size(0) == 2 and 0 in temp_unq):
                    #  have some field words -> penalize all labeled states that does not relate to the field, and also
                    # all the unlabeled states.
                    # find the temp other component.
                    temp = [x for x in temp_unq if x != 0][0]
                    pr_mask[l, e, b] = temp # this means that the candidate is good and pure and could be tested against
                                            # the entropy term -> want low entropy here.
                else:
                    pr_mask[l, e, b] = -1 # should be penalized for being a bad state.

                # not cover the complete segment from the left
                if e > 0 and mask[0, e - 1, b].item() == mask[0, e, b].item() and mask[0, e - 1, b].item() != 0:
                    pr_mask[l, e, b] = -1
                # not cover the complete segment from the right
                if e + l < seqlen - 1 and mask[0, e + l + 1, b].item() == mask[0, e + l, b].item() and mask[0, e + l + 1, b].item() != 0:
                    pr_mask[l, e, b] = -1

    return pr_mask

def gen_detailed_tgt_mask2_(condi, dict_label, L, seqlen, bsz, K, non_field):
    '''
        This helps with regulairzing the encoder side.
    :return: a penality mask of shape (L, seqlen, bsz, K)
    '''
    mask = torch.LongTensor(L, seqlen, bsz).fill_(0)
    for l in range(L):
        mask[l, :seqlen - l] = condi['inps'][l:, :, 0, 1]
    mask[mask == non_field] = 0
    pr_mask = torch.LongTensor(L, seqlen, bsz, K).fill_(0)


    for b in range(bsz):
        for l in range(L):
            for e in range(seqlen):
                temp_unq = mask[:l + 1, e, b].unique()
                if temp_unq.size(0) == 1 and 0 in temp_unq:
                    # all aux words -> penalize all labeled states.
                    pr_mask[l, e, b, :len(dict_label)] = 3
                elif temp_unq.size(0) == 1 or (temp_unq.size(0) == 2 and 0 in temp_unq):
                    #  have some field words -> penalize all labeled states that does not relate to the field, and also
                    # all the unlabeled states.
                    temp = [x.item() for x in temp_unq if x.item() != 0][0]
                    pr_mask[l, e, b, :] = 3 # 3
                    pr_mask[l, e, b, dict_label[temp]] = -9
                else:
                    # penalizing all states from breaking the convension segment boundary.
                    pr_mask[l, e, b, :] = 10

                # not cover the complete segment from the left
                if e > 0 and mask[0, e - 1, b].item() == mask[0, e, b].item() and mask[0, e - 1, b].item() != 0:
                    pr_mask[l, e, b, :] += 10
                # not cover the complete segment from the right
                if e + l < seqlen - 1 and mask[0, e + l + 1, b].item() == mask[0, e + l, b].item() and mask[0, e + l + 1, b].item() != 0:
                    pr_mask[l, e, b, :] += 10
    return pr_mask

def gen_detailed_tgt_mask2(condi, dict_label, L, seqlen, bsz, K, non_field):
    '''
        This helps with regulairzing the encoder side.
    :return: a penality mask of shape (L, seqlen, bsz, K)
    '''
    mask = torch.LongTensor(L, seqlen, bsz).fill_(0)
    for l in range(L):
        mask[l, :seqlen - l] = condi['inps'][l:, :, 0, 1]
    mask[mask == non_field] = 0
    pr_mask = torch.LongTensor(L, seqlen, bsz, K).fill_(0)


    for b in range(bsz):
        for l in range(L):
            for e in range(seqlen):
                temp_unq = mask[:l + 1, e, b].unique()
                if temp_unq.size(0) == 1 and 0 in temp_unq:
                    # all aux words -> penalize all labeled states.
                    pr_mask[l, e, b, :len(dict_label)-1] = 1
                elif temp_unq.size(0) == 1 or (temp_unq.size(0) == 2 and 0 in temp_unq):
                    #  have some field words -> penalize all labeled states that does not relate to the field, and also
                    # all the unlabeled states.
                    temp = [x.item() for x in temp_unq if x.item() != 0][0]
                    pr_mask[l, e, b, :] = 1 # 3
                    pr_mask[l, e, b, dict_label[temp]] = -1
                else:
                    # penalizing all states from breaking the convension segment boundary.
                    pr_mask[l, e, b, :] = 1

                # not cover the complete segment from the left
                if e > 0 and mask[0, e - 1, b].item() == mask[0, e, b].item() and mask[0, e - 1, b].item() != 0:
                    pr_mask[l, e, b, :] = 1
                # not cover the complete segment from the right
                if e + l < seqlen - 1 and mask[0, e + l + 1, b].item() == mask[0, e + l, b].item() and mask[0, e + l + 1, b].item() != 0:
                    pr_mask[l, e, b, :] = 1
    return pr_mask



def idx2word(batch_decode, dictionary):
    sent_ = []
    for example in batch_decode:
        for word in example:
            sent_.append([dictionary[x] for x in word])
    return sent_ 
        
def form_word_embeds(embeds, temp, embeds_locs):
    # eg. <e> a a a <e> b b b b <e> c c
    #      0  1 1 1  0  2 2 2 2  0  3 3

    # temp[count][bound_idx] = 0 # mark boundary by <empty>
    # temp[count][not_bound_idx] = sent[idx_outer] # mark sentence
    temp = temp.view(-1, rnn_inp_dim)
    temp[torch.cat(embeds_locs), :] = embeds
    return temp

def make_combo_targs(locs, x, L, nfields, ngen_types):
    """
    combines word and copy targets into a single tensor.
    locs - seqlen x bsz x max_locs
    x - seqlen x bsz
    assumes we have word indices, then fields, then a dummy
    returns L x bsz*seqlen x max_locs tensor corresponding to xsegs[1:]
    """
    seqlen, bsz, max_locs = locs.size()
    # print('locs')
    # print(locs.view(seqlen, bsz).transpose(0,1))
    # first replace -1s in first loc with target words
    # print('-'*100)
    # print(locs.shape)
    # print(locs)
    # print(ngen_types)

    addloc = locs + (ngen_types+1) # seqlen x bsz x max_locs
    firstloc = addloc[:, :, 0] # seqlen x bsz
    targmask = (firstloc == ngen_types) # -1 will now have value ngentypes
    firstloc[targmask] = x[targmask]
    # now replace remaining -1s w/ zero location
    addloc[addloc == ngen_types] = ngen_types+1+nfields # last index
    # finally put in same format as x_segs
    newlocs = torch.LongTensor(L, seqlen, bsz, max_locs).fill_(ngen_types+1+nfields)
    for i in range(L):
        newlocs[i][:seqlen-i].copy_(addloc[i:])
    # print(addloc)
    # print('+' * 100)
    return newlocs.transpose(1, 2).contiguous().view(L, bsz*seqlen, max_locs)


def get_kl_log(p1, p2):
    '''this computes the KL divergence in the log space. '''
    kl = torch.sum(p1.exp() * (p1 - p2))
    return kl

def get_kl_p(p1, p2):
    kl = torch.sum(p1 * (torch.log(p1) - torch.log(p2)))
    return kl



def get_uniq_fields(src, pad_idx, keycol=1):
    """
    src - bsz x nfields x nfeats
    """
    bsz = src.size(0)
    # get unique keys for each example
    keys = [torch.LongTensor(sorted(list(set(src[b, :, keycol])))) for b in range(bsz)]
    maxkeys = max(keyset.size(0) for keyset in keys)
    fields = torch.LongTensor(bsz, maxkeys).fill_(pad_idx)
    for b, keyset in enumerate(keys):
        fields[b][:len(keyset)].copy_(keyset)
    return fields


def make_masks(src, pad_idx, max_pool=False):
    """
    src - bsz x nfields x nfeats
    """
    bsz, nfields, nfeats = src.size()
    fieldmask = (src.eq(pad_idx).sum(2) == nfeats) # binary bsz x nfields tensor where 1 means bad. and 0 means good.

    return fieldmask

    # TODO: old verssion has returned avg mask as well.
    # neginf = -1e38
    # avgmask = (1 - fieldmask).float() # 1s where not padding
    # if not max_pool:
    #     avgmask.div_(avgmask.sum(1, True).expand(bsz, nfields))
    # fieldmask = fieldmask.float() * neginf # 0 where not all pad and -1e38 elsewhere
    # return fieldmask, avgmask


def logsumexp0(X):
    """
    X - L x B x K
    returns:
        B x K
    """
    if X.dim() == 2:
        X = X.unsqueeze(2)
    axis = 0
    X2d = X.view(X.size(0), -1)
    maxes, _ = torch.max(X2d, axis, True)
    lse = maxes + torch.log(torch.sum(torch.exp(X2d - maxes.expand_as(X2d)), axis, True))
    lse = lse.view(X.size(1), -1)
    return lse

def logsumexp2(X):
    """
    X - L x B x K
    returns:
        L x B
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)
    X2d = X.view(-1, X.size(2))
    maxes, _ = torch.max(X2d, 1, True)
    lse = maxes + torch.log(torch.sum(torch.exp(X2d - maxes.expand_as(X2d)), 1, True))
    lse = lse.view(X.size(0), -1)
    return lse

def logsumexp1(X):
    """
    X - B x K
    returns:
        B x 1
    """
    maxes, _ = torch.max(X, 1, True)
    lse = maxes + torch.log(torch.sum(torch.exp(X - maxes.expand_as(X)), 1, True))
    return lse





# ------------- unsure ----------------------------


def vvit_2_vseq(v1):
    result = []
    for idx, (start, end, state) in enumerate(v1):
        result += (end-start) *  [state]
    return result

def get_acc_seg(v1, v2):
    v_2 = np.array(vvit_2_vseq(v2))
    v_1 = np.array(vvit_2_vseq(v1))
    acc = (v_1 == v_2).sum() / len(v_1)
    return acc
# -----------------------------------------

def bwd_from_fwd_obs_logprobs(fwd_obs_logprobs):
    """
    fwd_obs_logprobs - L x T x bsz x K,
       where fwd_obs_logprobs[:,t,:,:] gives p(x_t), p(x_{t:t+1}), ..., p(x_{t:t+l})
    returns:
      bwd_obs_logprobs - L x T x bsz x K,
        where bwd_obs_logprobs[:,t,:,:] gives p(x_{t-L+1:t}), ..., p(x_{t})
    iow, fwd_obs_logprobs gives probs of segments starting at t, and bwd_obs_logprobs
    gives probs of segments ending at t
    """
    L = fwd_obs_logprobs.size(0)
    bwd_obs_logprobs = fwd_obs_logprobs.new().resize_as_(fwd_obs_logprobs).fill_(-float("inf"))
    bwd_obs_logprobs[L-1].copy_(fwd_obs_logprobs[0])
    for l in range(1, L):
        bwd_obs_logprobs[L-l-1, l:].copy_(fwd_obs_logprobs[l, :-l])
    return bwd_obs_logprobs
#
# def make_masks(src, pad_idx, max_pool=False):
#     """
#     src - bsz x nfields x nfeats
#     """
#     neginf = -1e38
#     bsz, nfields, nfeats = src.size()
#     fieldmask = (src.eq(pad_idx).sum(2) == nfeats) # binary bsz x nfields tensor
#     avgmask = (1 - fieldmask).float() # 1s where not padding
#     if not max_pool:
#         avgmask.div_(avgmask.sum(1, True).expand(bsz, nfields))
#     fieldmask = fieldmask.float() * neginf # 0 where not all pad and -1e38 elsewhere
#     return fieldmask, avgmask

def print_segment(label, sent):
    str = ''
    result = [None] * (len(sent) + len(label)*2)
    i = 0
    prev_e = 0
    for (s,e,l) in label:
        if s != prev_e:
            result[i:(s-prev_e)+i] = sent[prev_e:s]
            i += (s-prev_e)
        result[i] = '['
        i += 1
        result[i:i+e-s] = sent[s:e]
        i += (e-s)
        result[i] = ']'
        i += 1
    return result

def calc_mi(model, data):
    '''MI = E_{x,z ~ q(x,z)} log q(x,z) / p(x) q(z) = E [log q(z|x) - log(q(z))]
    The first term can be computed by the Entropy and the Second term can be computed by sampling, we first
    sample x from the dataset, and then we sample z from q(z|x) It would be a biased estimate right...'''
    entr_lst = []
    log_density_lst = []
    for idx, item in enumerate(data):
        x, y, state_z = item
        y = torch.LongTensor(y).view(1, -1)
        state_z_vit = vseq_2_vvit(state_z)

        dict_computed = model.hsmm_crf.get_weights(y, x)
        Z, Z_entr = model.hsmm_crf.get_entr(dict_computed)
        neg_entropy = -Z_entr.mean()
        entr_lst.append(neg_entropy)
        samples_lst, samples_vtb = model.hsmm_crf.get_sample(dict_computed, sample_num=args.mi_sample_size)
        sample_score = model.hsmm_crf.get_score(samples_lst, dict_computed)
        target = [torch.stack(samples) for samples in sample_score]
        target = torch.stack(target, dim=0)
        bsz, num_sample = target.shape
        log_density = (target - Z.expand(bsz, num_sample))

        # log p(z) shape = (z_batch, x_batch)
        log_density_lst.append(log_density)

    # log q(z): aggregate posterior
    # [z_batch]
    neg_entropy = torch.stack(entr_lst).mean()
    log_density = torch.stack(log_density_lst, dim=1)
    x_bsz = len(data)
    log_qz = (logsumexp1(log_density) - math.log(x_bsz)).squeeze(0).squeeze(0)
    # print(neg_entropy.shape)
    # print(log_density.shape)
    # print(x_bsz)
    # print(log_qz.shape)
    # print(neg_entropy - log_qz.mean(-1))
    return (neg_entropy - log_qz.mean(-1)).item()


def vseq_2_vit_lst_batch(z_full):
    bsz = len(z_full)
    result = []
    for tt in range(bsz):
        z = z_full[tt] + [-1]
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
                    end.append(idx)
                start.append(idx)
                prev_state = state[-1]
            else:
                idx += 1
        end.append(T)
        result.append([(a, b, c) for (a, b, c) in zip(start, end, state)])
    return result


def vseq_2_vit_lst(z):
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
                end.append(idx)
            start.append(idx)
            prev_state = state[-1]
        else:
            idx += 1
    end.append(T)
    return [(a,b,c) for (a,b,c) in zip(start, end, state)], (start, end, state)

def vit2lst(z):
    start_lst, end_lst, state_lst = [], [], []
    for idx, (a,b,c) in enumerate(z):
        start_lst.append(a)
        end_lst.append(b)
        state_lst.append(c)

    return start_lst, end_lst, state_lst

def get_acc_(viterbi, gold):
    vtb, gld = np.array(viterbi), np.array(gold)
    mask = (vtb == gld).sum()
    return mask/len(gold)

def print_dict(dict_, avg=True):
    if avg:
        denom = dict_['sent_num']
    else:
        denom = 1
    result_str = ''
    for key, val in dict_.items():
        try:
            result_str += '{}={},\t'.format(key, val/denom)
        except:
            pass
    if 'logp(word)' in dict_ and '#Tokens' in dict_:
        result_str += '{}={}'.format('PPL', np.exp(-dict_['logp(word)'] / dict_['#Tokens']))

    return result_str

def print_dict2(dict_, avg=True):
    if avg:
        denom = dict_['total_sent']
        # denom = dict_['sent_num']
    else:
        denom = 1
    result_str = ''
    for key, val in dict_.items():
        try:
            result_str += '{}={},\t'.format(key, val/denom)
        except:
            pass
    if 'logp(word)' in dict_ and '#Tokens' in dict_:
        result_str += '{}={}\t'.format('PPLtotal', np.exp(-dict_['total_ppl'] / dict_['total_token']))
        result_str += '{}={}\t'.format('PPLmctotal', np.exp(-dict_['total_ppl_mc'] / dict_['total_token']))
        result_str += '{}={}\t'.format('PPLavg', np.exp(-dict_['avg_ppl'] / dict_['total_sent']))
        result_str += '{}={}\t'.format('PPLmcavg', np.exp(-dict_['avg_ppl_mc'] / dict_['total_sent']))
        result_str += '{}={}\t'.format('PPLrecon', np.exp(-dict_['total_word'] / dict_['total_token']))

    return result_str

def print_dict3(dict_, avg=True):

    result_str = ''

    result_str += '{}={}\t'.format('PPLtotal', np.exp(-dict_['total_ppl_'] / dict_['total_token_']))
    result_str += '{}={}\t'.format('PPLavg', np.exp(-dict_['avg_ppl_'] / dict_['total_sent_']))
    result_str += '{}={}\t'.format('PPLtotalMC', np.exp(-dict_['total_ppl_mc_'] / dict_['total_token_']))
    result_str += '{}={}\t'.format('PPLavgMC', np.exp(-dict_['avg_ppl_mc_'] / dict_['total_sent_']))
    result_str += '{}={}\t'.format('PPL_recon', np.exp(-dict_['report_recon_ll_'] / dict_['total_sent_']))
    result_str += '{}={}\t'.format('Entr', dict_['report_entropy_'] / dict_['total_sent_'])
    result_str += '{}={}\t'.format('KL_div', dict_['report_KL_'] / dict_['total_sent_'])

    return result_str



def v1_2_v2(v1):
    pass
def v2_2_v1(v2):
    start_lst, end_lst, state_lst = v2
    result = []
    for idx, (start, end, state) in enumerate(zip(start_lst, end_lst, state_lst)):
        result.append((start, end+1, state))
    return result

def v1_2_v3(v1):
    result = []
    for idx, (start, end, state) in enumerate(v1):
        result += (end-start) *  [state]
    return result

def get_v1_acc(v3, v1):
    # print(v1)
    # print(v3)
    v_1 = np.array(v1_2_v3(v1[0]))
    acc = (v_1 == v3).sum() / len(v_1)
    return acc

def get_acc_seg(v1, v2):
    v2 = v2_2_v1(v2[0])
    v_2 = np.array(v1_2_v3(v2))
    v_1 = np.array(v1_2_v3(v1[0]))
    acc = (v_1 == v_2).sum() / len(v_1)
    return acc

def get_v3_acc(v_1, v_2):
    acc = (v_1 == v_2).sum() / len(v_1)
    return acc

def vvit_2_vseq(v1):
    result = []
    for idx, (start, end, state) in enumerate(v1):
        result += (end-start) *  [state]
    return result


def vseq_2_vvit(z):
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
                end.append(idx)
            start.append(idx)
            prev_state = state[-1]
        else:
            idx += 1
    end.append(T)
    return [(a,b,c) for (a,b,c) in zip(start, end, state)]


def get_acc_seg(v1, v2):
    v_2 = np.array(vvit_2_vseq(v2))
    v_1 = np.array(vvit_2_vseq(v1))
    acc = (v_1 == v_2).sum() / len(v_1)
    return acc


def get_segment_acc(v1, v2):
    '''
    compute the dice score of two viterbi style. (start, end, state) where
    start is inclusive, and end is not inclusive.
    :param v1:
    :param v2:
    :return:
    '''
    ignore_state_v1 = [(a, b) for (a, b, c) in v1]
    ignore_state_v2 = [(a, b) for (a, b, c) in v2]
    lst3 = [value for value in ignore_state_v1 if value in ignore_state_v2]
    return len(lst3)*2 / (len(ignore_state_v1) + len(ignore_state_v2))


def get_segment_acc_batch(v1, v2):
    '''
    compute the dice score of two viterbi style. (start, end, state) where
    start is inclusive, and end is not inclusive.
    :param v1:
    :param v2:
    :return:
    '''

    bsz = len(v1)
    result = 0
    assert  bsz == len(v2)
    for ii in range(bsz):
        result += get_segment_acc(v1[ii], v2[ii])
    return result


def get_linear_boundary(v1):
    _, length, _ = v1[-1]
    result = [0] * (length + 1)
    for idx, (start, end, state) in enumerate(v1):
        if idx == 0:
            continue
        result[start] = 1
    return result

def editDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def get_segment_ed_batch(v1, v2):
    bsz = len(v1)
    assert len(v1) == len(v2)
    result = 0
    for ii in range(bsz):
        ignore_state_v1 = get_linear_boundary(v1[ii])
        ignore_state_v2 = get_linear_boundary(v2[ii])
        temp = editDistance(ignore_state_v1, ignore_state_v2)
        result +=  temp / len(ignore_state_v1)
    return result

def get_segment_ed(v1, v2):
    ignore_state_v1 = get_linear_boundary(v1)
    ignore_state_v2 = get_linear_boundary(v2)
    # print(ignore_state_v1)
    # print(ignore_state_v2)
    # print(len(ignore_state_v1), len(ignore_state_v2))
    temp = editDistance(ignore_state_v1, ignore_state_v2)
    # print(temp)
    return temp / len(ignore_state_v1)

def get_f1_score(v1, v2):
    '''

    :param v1:  the true sequence
    :param v2:  the predicted viterbi sequence.
    :return:
    '''
    ignore_state_v1 = get_linear_boundary(v1)
    ignore_state_v2 = get_linear_boundary(v2)
    # print(ignore_state_v1)
    # print(ignore_state_v2)
    temp = f1_score(ignore_state_v1, ignore_state_v2)
    # print(temp)
    return temp


def get_f1_score_batch(vv1, vv2):
    '''

    :param v1:  the true sequence
    :param v2:  the predicted viterbi sequence.
    :return:
    '''
    bsz = len(vv1)
    result = 0
    for i in range(bsz):
        v1 = vv1[i]
        v2 = vv2[i]
        ignore_state_v1 = get_linear_boundary(v1)
        ignore_state_v2 = get_linear_boundary(v2)
        # print(ignore_state_v1)
        # print(ignore_state_v2)
        result += f1_score(ignore_state_v1, ignore_state_v2)
    # print(temp)
    return result












# def argmax(vec):
#     # return the argmax as a python int
#     _, idx = torch.max(vec, 1)
#     return idx.item()
#
#
# def log_sum_exp(vec):
#     max_score = vec[0, argmax(vec)]
#     max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
#     return max_score + \
#            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
#
# def get_actions(tree, SHIFT=0, REDUCE=1, OPEN='(', CLOSE=')'):
#     # input tree in bracket form: ((A B) (C D))
#     # output action sequence: 0 0 1 0 0 1 1, where 0 is SHIFT and 1 is REDUCE
#     actions = []
#     tree = tree.strip()
#     i = 0
#     num_shift = 0
#     num_reduce = 0
#     left = 0
#     right = 0
#     while i < len(tree):
#         if tree[i] != ' ' and tree[i] != OPEN and tree[i] != CLOSE:  # terminal
#             if tree[i - 1] == OPEN or tree[i - 1] == ' ':
#                 actions.append(SHIFT)
#                 num_shift += 1
#         elif tree[i] == CLOSE:
#             actions.append(REDUCE)
#             num_reduce += 1
#             right += 1
#         elif tree[i] == OPEN:
#             left += 1
#         i += 1
#     assert (num_shift == num_reduce + 1)
#     return actions
#
#
# def get_tree(actions, sent=None, SHIFT=0, REDUCE=1):
#     # input action and sent (lists), e.g. S S R S S R R, A B C D
#     # output tree ((A B) (C D))
#     stack = []
#     pointer = 0
#     if sent is None:
#         sent = list(map(str, range((len(actions) + 1) // 2)))
#     for action in actions:
#         if action == SHIFT:
#             word = sent[pointer]
#             stack.append(word)
#             pointer += 1
#         elif action == REDUCE:
#             right = stack.pop()
#             left = stack.pop()
#             stack.append('(' + left + ' ' + right + ')')
#     assert (len(stack) == 1)
#     return stack[-1]
#
#
# def get_spans(actions, SHIFT=0, REDUCE=1):
#     sent = list(range((len(actions) + 1) // 2))
#     spans = []
#     pointer = 0
#     stack = []
#     for action in actions:
#         if action == SHIFT:
#             word = sent[pointer]
#             stack.append(word)
#             pointer += 1
#         elif action == REDUCE:
#             right = stack.pop()
#             left = stack.pop()
#             if isinstance(left, int):
#                 left = (left, None)
#             if isinstance(right, int):
#                 right = (None, right)
#             new_span = (left[0], right[1])
#             spans.append(new_span)
#             stack.append(new_span)
#     return spans
#
#
# def get_stats(span1, span2):
#     tp = 0
#     fp = 0
#     fn = 0
#     for span in span1:
#         if span in span2:
#             tp += 1
#         else:
#             fp += 1
#     for span in span2:
#         if span not in span1:
#             fn += 1
#     return tp, fp, fn
#
#
# def update_stats(pred_span, gold_spans, stats):
#     for gold_span, stat in zip(gold_spans, stats):
#         tp, fp, fn = get_stats(pred_span, gold_span)
#         stat[0] += tp
#         stat[1] += fp
#         stat[2] += fn
#
#
# def get_f1(stats):
#     f1s = []
#     for stat in stats:
#         prec = stat[0] / (stat[0] + stat[1])
#         recall = stat[0] / (stat[0] + stat[2])
#         f1 = 2 * prec * recall / (prec + recall) * 100 if prec + recall > 0 else 0.
#         f1s.append(f1)
#     return f1s
#
#
# def span_str(start=None, end=None):
#     assert (start is not None or end is not None)
#     if start is None:
#         return ' ' + str(end) + ')'
#     elif end is None:
#         return '(' + str(start) + ' '
#     else:
#         return ' (' + str(start) + ' ' + str(end) + ') '
#
#
# def get_tree_from_binary_matrix(matrix, length):
#     sent = list(map(str, range(length)))
#     n = len(sent)
#     tree = {}
#     for i in range(n):
#         tree[i] = sent[i]
#     for k in np.arange(1, n):
#         for s in np.arange(n):
#             t = s + k
#             if t > n - 1:
#                 break
#             if matrix[s][t].item() == 1:
#                 span = '(' + tree[s] + ' ' + tree[t] + ')'
#                 tree[s] = span
#                 tree[t] = span
#     return tree[0]
#
#
# def get_nonbinary_spans(actions, SHIFT=0, REDUCE=1):
#     spans = []
#     stack = []
#     pointer = 0
#     binary_actions = []
#     nonbinary_actions = []
#     num_shift = 0
#     num_reduce = 0
#     for action in actions:
#         # print(action, stack)
#         if action == "SHIFT":
#             nonbinary_actions.append(SHIFT)
#             stack.append((pointer, pointer))
#             pointer += 1
#             binary_actions.append(SHIFT)
#             num_shift += 1
#         elif action[:3] == 'NT(':
#             stack.append('(')
#         elif action == "REDUCE":
#             nonbinary_actions.append(REDUCE)
#             right = stack.pop()
#             left = right
#             n = 1
#             while stack[-1] is not '(':
#                 left = stack.pop()
#                 n += 1
#             span = (left[0], right[1])
#             if left[0] != right[1]:
#                 spans.append(span)
#             stack.pop()
#             stack.append(span)
#             while n > 1:
#                 n -= 1
#                 binary_actions.append(REDUCE)
#                 num_reduce += 1
#         else:
#             assert False
#     assert (len(stack) == 1)
#     assert (num_shift == num_reduce + 1)
#     return spans, binary_actions, nonbinary_actions


