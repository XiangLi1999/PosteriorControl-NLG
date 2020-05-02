#!/usr/bin/python
# --------------------------------------- 
# File Name : experiment.py
# Creation Date : 21-11-2017
# Last Modified : Tue Nov 21 11:04:35 2017
# Created By : wdd 
# ---------------------------------------
from shepherd import shepherd, init, post, USR, ALL, SYS, SPD, grid_search, basic_func, load_job_info, _args
import shutil

import os, re

try:
    import cPickle as pickle
except:
    import pickle

udv14_train = ['cs', 'ru_syntagrus', 'es_ancora', 'ca', 'es', 'fr', 'hi', 'la_ittb', 'it', 'de', 'zh', 'ar']
udv14_dev = ['en_cesl', 'en_esl', 'en_lines', 'en']
udv14 = ['ar', 'zh', 'en', 'hi', 'es_ancora']


def _itr_file_list(input, pattern):
    print('Search Patterm:', pattern)
    ptn = re.compile(pattern)
    for root, dir, files in os.walk(input):
        for fn in files:
            abs_fn = os.path.normpath(os.path.join(root, fn))
            m = ptn.match(abs_fn)
            if m:
                lang = m.groups()
                yield lang, abs_fn


def _get_data(config):
    ret = []
    for l in config.split('+'):
        if l == 'train':
            ret += udv14_train
        elif l == 'dev':
            ret += udv14_dev
    ret = config.split('|') if ret == [] else ret
    return ret




@shepherd(before=[init], after=[post])
def con_train22():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset', 'data/e2e_aligned/')
    # USR.set('bsz', '20')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    USR.set('trans_unif', 'yes')
    USR.set('h_dim', '512')
    USR.set('layers', '2')
    USR.set('min_epochs', '6')
    USR.set('lr_p', '0.001')
    USR.set('lr_q', '0.001')
    USR.set('weight_decay', '0.0')
    #
    # posterior reg setting
    #

    # USR.set('pr_reg_style', 'phrase')
    USR.set('posterior_reg', '1')
    # USR.set('pr_coef', '5')
    # USR.set('hard_code', 'no')  # for comparing hard coding v.s. not hard-coding.
    # USR.set('encoder_constraint', 'yes') # ALWAYS on. a prerequisite for both hard_code and posterior_reg
    # USR.set('decoder_constraint', 'no') # this could only be turned on if we use hard code.

    # USR.set('use_elmo', 'no')
    # USR.set('elmo_style', '1') # or 3.


    USR.set('train_q_epoch', '3')
    # USR.set('tagset_size', '10')
    # USR.set('seed', '1')
    # USR.set('thresh', '5')


    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 30' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --sample_size 4' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --option train' \
              % ALL()

    command += ' --posterior_reg 1 --trans_unif yes --hidden_dim 512 --embedding_dim 512 --layers 2 ' \
               '--train_q_epoch 3 --lr_p 0.001 --lr_q 0.001 --weight_decay 0.0'

    search_list = [
        ('pr_reg_style', 'wb:soft'), #wb:entr|
        ('pr_coef', '5'),
        ('hard_code', 'no'),
        ('decoder_constraint', 'no'),
        ('encoder_constraint', 'yes'),
        ('bsz', '7'), # |10
        ('tagset_size', '128'),
        ('prior', 'markov'), #ar
        ('max_mbs_per_epoch', '20000'),  # 20000|35000
        ('use_elmo', 'no'), # yes if we use elmo; no if we don't use ELMo.
        ('elmo_style', '1'),  # 1 if we use MLP and 3 if we only use ELMo.
        ('full_independence', '3'), # 1 means fullrnn; 0 means factorize by the prior. 2. is the RNNG style. 3 is the simplisitic style
        ('seed', '0'),
        ('thresh', '3')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def beam_search_wb():
    USR.set('dataset', 'data/wb_aligned/')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    USR.set('trans_unif', 'yes')
    # USR.set('h_dim', '512')
    USR.set('layers', '2')
    USR.set('min_epochs', '6')
    USR.set('weight_decay', '0.0')
    USR.set('posterior_reg', '1')
    USR.set('load', '')
    USR.set('test', 'no')

    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --load %(U_load)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 30' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --option beam' \
              % ALL()

    if USR.test == 'yes':
        command += ' --test'

    command += ' --posterior_reg 1 --trans_unif yes --layers 2 ' \
               '--train_q_epoch 3 --weight_decay 0.0 --full_independence 3'

    search_list = [
        ('pr_reg_style', 'wb:soft'),
        ('bsz', '10|8'),
        ('pr_coef', '15'),
        ('hard_code', 'no'),
        ('decoder_constraint', 'no'),
        ('encoder_constraint', 'yes'),
        ('tagset_size', '70'),
        ('max_mbs_per_epoch', '15000'),  # 20000|35000
        ('use_elmo', 'no'),  # yes if we use elmo; no if we don't use ELMo.
        ('elmo_style', '1'),  # 1 if we use MLP and 3 if we only use ELMo.
        ('seed', '0'),
        ('thresh', '1000'),
        ('hidden_dim', '500'),
        ('embedding_dim', '400'),
        ('lr_p', '0.0005'),
        ('lr_q', '0.001'),
        ('sample_size', '2'),
        ('dual_attn', 'yes'),
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def con_train_wb():
    USR.set('dataset', 'data/wb_aligned/')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    USR.set('trans_unif', 'yes')
    # USR.set('h_dim', '512')
    USR.set('layers', '2')
    USR.set('min_epochs', '8')
    USR.set('weight_decay', '0.0')
    USR.set('posterior_reg', '1')


    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 55' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --option train' \
              % ALL()

    command += ' --posterior_reg 1 --trans_unif yes --layers 2 ' \
               '--train_q_epoch 4 --weight_decay 0.0 --full_independence 3'

    search_list = [
        ('pr_reg_style', 'wb:soft'),
        ('bsz', '10|8'),
        ('pr_coef', '15'),
        ('hard_code', 'no'),
        ('decoder_constraint', 'no'),
        ('encoder_constraint', 'yes'),
        ('tagset_size', '64|128'),
        ('max_mbs_per_epoch', '25000'),  # 20000|35000
        ('use_elmo', 'no'), # yes if we use elmo; no if we don't use ELMo.
        ('elmo_style', '1'),  # 1 if we use MLP and 3 if we only use ELMo.
        ('seed', '0'),
        ('thresh', '1000'),
        ('hidden_dim', '500'),
        ('embedding_dim', '400'),
        ('lr_p', '0.0005'),
        ('lr_q', '0.001'),
        ('sample_size', '2'),
        ('dual_attn', 'yes'),
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def con_train_e2e_rnn():
    USR.set('dataset', 'data/e2e_aligned/')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    # USR.set('h_dim', '512')
    USR.set('layers', '2')
    USR.set('min_epochs', '8')
    USR.set('posterior_reg', '1')


    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 55' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --option train' \
              ' --mode RNN' \
              % ALL()

    command += ' --posterior_reg 1  --layers 2 ' \
               '--train_q_epoch 5  --full_independence 3'

    search_list = [
        ('pr_reg_style', 'phrase'),
        ('bsz', '20'),
        ('pr_coef', '5'), # prev it's 15
        ('hard_code', 'no'),
        ('decoder_constraint', 'no'),
        ('encoder_constraint', 'yes'),
        ('tagset_size', '11'), #150
        ('max_mbs_per_epoch', '25000'),  # 20000|35000
        ('use_elmo', 'no'), # yes if we use elmo; no if we don't use ELMo.
        ('seed', '0'),
        ('thresh', '1000'),
        ('hidden_dim', '500'),
        ('embedding_dim', '400'),
        ('lr_p', '0.0005'),
        ('lr_q', '0.001'),
        ('sample_size', '1'),
        ('dual_attn', 'no'),
        ('trans_unif', 'yes'),
        ('weight_decay', '0.0|0.0001'),
        ('task', 'bert|lex_rnn')
    ]

    """
     USR.set('trans_unif', 'yes')
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def con_train_e2e():
    USR.set('dataset', 'data/e2e_aligned/')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    # USR.set('h_dim', '512')
    USR.set('layers', '2')
    USR.set('min_epochs', '8')
    USR.set('posterior_reg', '1')


    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 55' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --option train' \
              % ALL()

    command += ' --posterior_reg 1  --layers 2 ' \
               '--train_q_epoch 5 --full_independence 3 --weight_decay 0  '

    search_list = [
        ('pr_reg_style', 'phrase'),
        ('bsz', '20'),
        ('pr_coef', '25|15|5|0'), # prev it's 15
        ('hard_code', 'no'),
        ('decoder_constraint', 'no'),
        ('encoder_constraint', 'yes'),
        ('tagset_size', '11'), #150
        ('max_mbs_per_epoch', '25000'),  # 20000|35000
        ('use_elmo', 'no'), # yes if we use elmo; no if we don't use ELMo.
        ('seed', '0'),
        ('thresh', '1000'),
        ('hidden_dim', '500'),
        ('embedding_dim', '500'),
        ('lr_p', '0.001'),
        ('lr_q', '0.001'),
        ('sample_size', '4'),
        ('dual_attn', 'yes'),
        ('trans_unif', 'yes'),
        ('task', 'lex'),
    ]

    """
     USR.set('trans_unif', 'yes')
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def con_train_e2e_test():
    USR.set('dataset', 'data/e2e_aligned/')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    # USR.set('h_dim', '512')
    USR.set('layers', '2')
    USR.set('min_epochs', '8')
    USR.set('posterior_reg', '1')


    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --load /n/holylfs/LABS/rush_lab/users/lisa/FSA-RNN/jobs/con_train_e2e-bert1/model/{config}' \
              ' --save_out %(S_output)s/{config}-test' \
              ' --epoch 55' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --option beam' \
              % ALL()

    command += ' --posterior_reg 1  --layers 2 ' \
               '--train_q_epoch 5 --full_independence 3 --weight_decay 0  --test --begin_r 0 --end_r 1000'

    search_list = [
        ('pr_reg_style', 'phrase'),
        ('bsz', '20'),
        ('pr_coef', '25|15|5|0'), # prev it's 15
        ('hard_code', 'no'),
        ('decoder_constraint', 'no'),
        ('encoder_constraint', 'yes'),
        ('tagset_size', '11'), #150
        ('max_mbs_per_epoch', '25000'),  # 20000|35000
        ('use_elmo', 'no'), # yes if we use elmo; no if we don't use ELMo.
        ('seed', '0'),
        ('thresh', '1000'),
        ('hidden_dim', '500'),
        ('embedding_dim', '500'),
        ('lr_p', '0.001'),
        ('lr_q', '0.001'),
        ('sample_size', '4'),
        ('dual_attn', 'yes'),
        ('trans_unif', 'yes'),
        ('task', 'bert1|bert2'),
    ]

    """
     USR.set('trans_unif', 'yes')
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def con_train_e2e_soft():
    USR.set('dataset', 'data/e2e_aligned/')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    # USR.set('h_dim', '512')
    USR.set('layers', '2')
    USR.set('min_epochs', '8')
    USR.set('weight_decay', '0.0')
    USR.set('posterior_reg', '1')


    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 55' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --option train' \
              % ALL()

    command += ' --posterior_reg 1  --layers 2 ' \
               '--train_q_epoch 5 --weight_decay 0.0 --full_independence 3'

    search_list = [
        ('pr_reg_style', 'soft'),
        ('bsz', '20'),
        ('pr_coef', '25|15|5'), # prev it's 15
        ('hard_code', 'no'),
        ('decoder_constraint', 'no'),
        ('encoder_constraint', 'yes'),
        ('tagset_size', '9'), #150
        ('max_mbs_per_epoch', '25000'),  # 20000|35000
        ('use_elmo', 'no'), # yes if we use elmo; no if we don't use ELMo.
        ('seed', '0'),
        ('thresh', '1000'),
        ('hidden_dim', '500'),
        ('embedding_dim', '400'),
        ('lr_p', '0.001'),
        ('lr_q', '0.001'),
        ('sample_size', '4'),
        ('dual_attn', 'no'),
        ('trans_unif', 'yes'),
        ('task', 'bert_distrib'),
    ]

    """
     USR.set('trans_unif', 'yes')
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def con_train_wbglobal():
    USR.set('dataset', 'data/wb_aligned/')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    # USR.set('h_dim', '512')
    USR.set('layers', '2')
    USR.set('min_epochs', '8')
    USR.set('weight_decay', '0.0')
    USR.set('posterior_reg', '1')


    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 55' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --option train' \
              % ALL()

    command += ' --posterior_reg 1  --layers 2 ' \
               '--train_q_epoch 5 --weight_decay 0.0 --full_independence 3'

    search_list = [
        ('pr_reg_style', 'wb:global'),
        ('bsz', '10'),
        ('pr_coef', '25|15|5'), # prev it's 15
        ('hard_code', 'no'),
        ('decoder_constraint', 'no'),
        ('encoder_constraint', 'yes'),
        ('tagset_size', '70'), #150
        ('max_mbs_per_epoch', '25000'),  # 20000|35000
        ('use_elmo', 'no'), # yes if we use elmo; no if we don't use ELMo.
        ('elmo_style', '1'),  # 1 if we use MLP and 3 if we only use ELMo.
        ('seed', '0'),
        ('thresh', '1000'),
        ('hidden_dim', '500'),
        ('embedding_dim', '400'),
        ('lr_p', '0.0005'),
        ('lr_q', '0.001'),
        ('sample_size', '3'),
        ('dual_attn', 'yes'),
        ('trans_unif', 'yes'),
    ]

    """
     USR.set('trans_unif', 'yes')
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def con_train_wbcluster():
    USR.set('dataset', 'data/wb_aligned/')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    # USR.set('h_dim', '512')
    USR.set('layers', '2')
    USR.set('min_epochs', '8')
    USR.set('weight_decay', '0.0')
    USR.set('posterior_reg', '1')


    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 55' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --option train' \
              % ALL()

    command += ' --posterior_reg 1  --layers 2 ' \
               '--train_q_epoch 5 --weight_decay 0.0 --full_independence 3'

    search_list = [
        ('pr_reg_style', 'wb:cluster'),
        ('bsz', '10'),
        ('pr_coef', '25|15|5'), # prev it's 15
        ('hard_code', 'no'),
        ('decoder_constraint', 'no'),
        ('encoder_constraint', 'yes'),
        ('tagset_size', '70'), #150
        ('max_mbs_per_epoch', '25000|10000'),  # 20000|35000
        ('use_elmo', 'no'), # yes if we use elmo; no if we don't use ELMo.
        ('elmo_style', '1'),  # 1 if we use MLP and 3 if we only use ELMo.
        ('seed', '0'),
        ('thresh', '1000'),
        ('hidden_dim', '500'),
        ('embedding_dim', '400'),
        ('lr_p', '0.0005'),
        ('lr_q', '0.001'),
        ('sample_size', '2'),
        ('dual_attn', 'yes'),
        ('trans_unif', 'yes'),
    ]

    """
     USR.set('trans_unif', 'yes')
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def train_permute():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('feature', 'TRA')
    USR.set('batch_size', '100')
    USR.set('mx_dep_train', '6')
    pattern = '.*/UD_.*/({lang})-ud-train.conllu$' % USR
    command = 'python %(S_python_dir)s/permutation/main.py' \
              ' --task train' % ALL() + \
              f' --feature {USR.feature}' \
              f' --batch_size {USR.batch_size}' \
              f' --mx_dep_train {USR.mx_dep_train}'
    USR.set('feature_vocab', 'lat-build_perm_feature-ud14')
    lgs = udv14
    for (src,), src_fn in _itr_file_list('%(S_data)s/%(U_tb)s/' % ALL(), pattern.format(lang='|'.join(lgs))):
        config, vocab = src, src
        src_out = os.path.join(SYS.tmp, config + '.conllu')
        src_spec = f' --feature_vocab {SYS.jobs}/{USR.feature_vocab}/output/{vocab}.pkl' \
                   f' --model {SYS.model}/{config}' \
                   f' --src {src_fn}' \
                   f' --src_out {src_out}'
        SPD().submit([command + src_spec], config)





@shepherd(before=[init], after=[post])
def con_train():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset', '/Users/xiangli/Desktop/Sasha/FSA-RNN/data/e2e_aligned/')
    USR.set('bsz', '20')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    USR.set('trans_unif', 'yes')
    USR.set('h_dim', '512')
    USR.set('layers', '2')
    USR.set('min_epochs', '6')

    USR.set('lr_p', '0.001')
    USR.set('lr_q', '0.001')
    USR.set('weight_decay', '0.0')
    #
    # posterior reg setting
    #
    USR.set('pr_reg_style', 'phrase')
    USR.set('posterior_reg', '1')
    USR.set('pr_coef', '5')
    USR.set('hard_code', 'no')  # for comparing hard coding v.s. not hard-coding.
    USR.set('encoder_constraint', 'yes') # ALWAYS on. a prerequisite for both hard_code and posterior_reg
    USR.set('decoder_constraint', 'no') # this could only be turned on if we use hard code.

    USR.set('use_elmo', 'no')
    USR.set('elmo_style', '1') # or 3.


    USR.set('train_q_epoch', '3')
    USR.set('tagset_size', '10')
    USR.set('seed', '1')
    USR.set('thresh', '5')


    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 30' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --sample_size 4' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --option train' \
              % ALL()

    # --pr_reg_style wb:ent

    # search_list = [
    #     ('posterior_reg', USR.posterior_reg),
    #     ('pr_reg_style', USR.pr_reg_style),
    #     ('pr_coef', USR.pr_coef),
    #     ('hard_code', USR.hard_code),
    #     ('decoder_constraint', USR.decoder_constraint),
    #     ('encoder_constraint', USR.encoder_constraint),
    #     ('trans_unif', USR.trans_unif),
    #     ('hidden_dim', USR.h_dim),
    #     ('bsz', USR.bsz),
    #     ('layers', USR.layers),
    #     ('embedding_dim', USR.h_dim),
    #     ('train_q_epoch', USR.train_q_epoch),
    #     ('tagset_size', USR.tagset_size),
    #     ('lr_p', USR.lr_p),
    #     ('lr_q', USR.lr_q),
    #     ('weight_decay', USR.weight_decay),
    #     ('prior', 'markov'), #ar
    #     ('max_mbs_per_epoch', '20000'),  # 20000|35000
    #     ('use_elmo', USR.use_elmo), # yes if we use elmo
    #     ('elmo_style', USR.elmo_style),  # 1 if we use MLP and 3 if we only use ELMo.
    #     ('full_independence', '3'), # 1 means fullrnn; 0 means factorize by the prior. 2. is the RNNG style.
    #                                 # 3 is the simplisitic style
    #     ('seed', USR.seed),
    #     ('thresh', USR.thresh),
    # ]

    command += ' --posterior_reg 1 --trans_unif yes --hidden_dim 512 --embedding_dim 512 --layers 2 ' \
               '--train_q_epoch 3 --lr_p 0.001 --lr_q 0.001 --weight_decay 0.0'
    search_list = [
        ('pr_reg_style', 'wb:entr|wb:soft'),
        ('pr_coef', '5'),
        ('hard_code', 'no'),
        ('decoder_constraint', 'no'),
        ('encoder_constraint', 'yes'),
        ('bsz', '7|10'),
        ('tagset_size', '128'),
        ('prior', 'markov'), #ar
        ('max_mbs_per_epoch', '20000'),  # 20000|35000
        ('use_elmo', 'yes'), # yes if we use elmo
        ('elmo_style', '1|3'),  # 1 if we use MLP and 3 if we only use ELMo.
        ('full_independence', '3'), # 1 means fullrnn; 0 means factorize by the prior. 2. is the RNNG style. 3 is the simplisitic style
        ('seed', '0'),
        ('thresh', '10|5')
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def beam_rnn():
    USR.set('tb', 'ud-treebanks-v1.4')
    USR.set('dataset', '/Users/xiangli/Desktop/Sasha/FSA-RNN/data/e2e_aligned/')
    USR.set('bsz', '10')
    USR.set('load', '')
    USR.set('decoder', 'crf')
    USR.set('full_dep', '1')
    USR.set('L', '8')
    USR.set('test', 'no')
    USR.set('lr_p', '0.01')
    USR.set('h_dim', '128')
    USR.set('weight_decay', '0.001')
    USR.set('seed', '0')
    USR.set('min_epochs', '20')
    USR.set('layers', '1')
    USR.set('optim', 'adam')

    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --load %(U_load)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 30' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --sample_size 1' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --bsz %(U_bsz)s' \
              ' --option beam' \
              ' --mode RNN' \
              ' --mlpinp' \
              % ALL()

    if USR.test == 'yes':
        command += ' --test'

    search_list = [
        ('ph', USR.ph),
        ('optim', USR.optim),
        ('min_epochs', USR.min_epochs),
        ('seed', USR.seed),
        ('weight_decay', USR.weight_decay),
        ('hidden_dim', USR.h_dim),
        ('embedding_dim', USR.h_dim),
        ('tagset_size', '10'),
        ('lr_p', USR.lr_p),
        ('layers', USR.layers),
        ('prior', 'markov'),  # ar
        ('full_independence', '3'),  # 1 means fullrnn; 0 means factorize by the prior. 2. is the RNNG style.
        # 3 is the simplisitic style
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def beam_control():
    USR.set('dataset', '/Users/xiangli/Desktop/Sasha/FSA-RNN/data/e2e_aligned/')
    USR.set('bsz', '10')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    USR.set('lr_p', '0.001')
    USR.set('lr_q', '0.001')
    USR.set('h_dim', '512')
    USR.set('train_q_epoch', '3')
    USR.set('layers', '2')
    USR.set('tagset_size', '10')
    USR.set('load', '')
    USR.set('ph', '1')
    USR.set('test', 'no')
    USR.set('beam_size', '1')

    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --load %(U_load)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 30' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --sample_size 4' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --bsz %(U_bsz)s' \
              ' --option controlled' \
              % ALL()

    if USR.test == 'yes':
        command += ' --test'


    search_list = [
        ('hidden_dim', USR.h_dim),
        ('bsz', USR.bsz),
        ('layers', USR.layers),
        ('embedding_dim', USR.h_dim),
        ('train_q_epoch', USR.train_q_epoch),
        ('tagset_size', USR.tagset_size),
        ('lr_p', USR.lr_p),
        ('lr_q', USR.lr_q),
        ('prior', 'markov'),  # ar
        ('full_independence', '3'),  # 1 means fullrnn; 0 means factorize by the prior. 2. is the RNNG style.
        # 3 is the simplisitic style
        ('ph', USR.ph),
        ('beam_size', USR.beam_size),
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return


@shepherd(before=[init], after=[post])
def get_p_template():
    USR.set('dataset', '/Users/xiangli/Desktop/Sasha/FSA-RNN/data/e2e_aligned/')
    USR.set('bsz', '10')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    USR.set('lr_p', '0.001')
    USR.set('lr_q', '0.001')
    USR.set('h_dim', '512')
    USR.set('train_q_epoch', '3')
    USR.set('layers', '2')
    USR.set('tagset_size', '10')
    USR.set('load', '')
    USR.set('ph', '1')
    USR.set('test', 'no')
    USR.set('beam_size', '1')

    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --load %(U_load)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 30' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --sample_size 4' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --bsz %(U_bsz)s' \
              ' --option induce_template_p' \
              % ALL()

    if USR.test == 'yes':
        command += ' --test'


    search_list = [
        ('hidden_dim', USR.h_dim),
        ('bsz', USR.bsz),
        ('layers', USR.layers),
        ('embedding_dim', USR.h_dim),
        ('train_q_epoch', USR.train_q_epoch),
        ('tagset_size', USR.tagset_size),
        ('lr_p', USR.lr_p),
        ('lr_q', USR.lr_q),
        ('prior', 'markov'),  # ar
        ('full_independence', '3'),  # 1 means fullrnn; 0 means factorize by the prior. 2. is the RNNG style.
        # 3 is the simplisitic style
        ('ph', USR.ph),
        ('beam_size', USR.beam_size),
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return

@shepherd(before=[init], after=[post])
def beam_search():
    USR.set('dataset', '/Users/xiangli/Desktop/Sasha/FSA-RNN/data/e2e_aligned/')
    USR.set('bsz', '10')
    USR.set('decoder', 'crf')
    USR.set('L', '8')
    USR.set('lr_p', '0.001')
    USR.set('lr_q', '0.001')
    USR.set('h_dim', '128')
    USR.set('train_q_epoch', '3')
    USR.set('layers', '2')
    USR.set('tagset_size', '10')
    USR.set('load', '')
    USR.set('ph', '1')
    USR.set('test', 'no')

    command = '%(S_python_itrptr)s %(S_python_dir)s/train.py' \
              ' --data %(U_dataset)s' \
              ' --load %(U_load)s' \
              ' --save %(S_model)s/{config}' \
              ' --save_out %(S_output)s/{config}' \
              ' --epoch 30' \
              ' --data_mode real' \
              ' --optim_algo 1' \
              ' --L %(U_L)s' \
              ' --decoder %(U_decoder)s' \
              ' --sample_size 4' \
              ' --cuda' \
              ' --one_rnn' \
              ' --sep_attn' \
              ' --bsz %(U_bsz)s' \
              ' --option beam' \
              % ALL()

    if USR.test == 'yes':
        command += ' --test'


    search_list = [
        ('hidden_dim', USR.h_dim),
        ('bsz', USR.bsz),
        ('layers', USR.layers),
        ('embedding_dim', USR.h_dim),
        ('train_q_epoch', USR.train_q_epoch),
        ('tagset_size', USR.tagset_size),
        ('lr_p', USR.lr_p),
        ('lr_q', USR.lr_q),
        ('prior', 'markov'),  # ar
        ('full_independence', '3'),  # 1 means fullrnn; 0 means factorize by the prior. 2. is the RNNG style.
        # 3 is the simplisitic style
        ('ph', USR.ph),
    ]

    """
    Submit jobs
    """
    grid_search(lambda map: basic_func(command, map), search_list, seed=1)
    return





@shepherd(before=[init], after=[post])
def collect_tb_info():
    USR.set('tb', 'ud-treebanks-v1.4')
    pattern = '.*/(.*?)-ud-train.conllu$' % USR
    command = 'python %(S_python_dir)s/permutation/main.py --task collect_info' % ALL()
    for (src,), src_fn in _itr_file_list('%(S_data)s/%(U_tb)s/' % ALL(), pattern):
        config = src
        src_spec = f' --test {src_fn}'
        SPD().submit([command + src_spec], config)



def _itr_file(input, pattern):
    print('Search Patterm:', pattern)
    ptn = re.compile(pattern)
    for root, dir, files in os.walk(input):
        for fn in files:
            abs_fn = os.path.normpath(os.path.join(root, fn))
            m = ptn.match(abs_fn)
            if m:
                lang = m.groups()
                yield lang, abs_fn

