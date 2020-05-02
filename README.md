# FSA-RNN

Posterior Control of Blackbox Generation

-- How to train? run the following commands to generate the scripts, and run the scripts. 

python shepherd.py -d con_train_wbcluster-1

python shepherd.py -d con_train_wbglobal-1

python shepherd.py -d con_train_e2e-1

-- Download and play with the trained model: 

https://drive.google.com/drive/folders/1e-chqMmCx-NmVenrcqspNyLm8gwCqCOP?usp=sharing is the res file. 

-- Load trained model: 

(Dynamic Constraints PR)
```
python train.py --data data/wb_aligned/ --load path/to/loaded/model --save_out path/to/output/file --data_mode real --optim_algo 1 --L 8 --decoder crf --cuda --one_rnn --sep_attn --option beam --posterior_reg 1  --layers 2 --train_q_epoch 5 --weight_decay 0.0 --full_independence 3 --pr_reg_style wb:global --bsz 10 --pr_coef 15 --hard_code no --decoder_constraint no --encoder_constraint yes --tagset_size 70 --max_mbs_per_epoch 25000 --use_elmo no --elmo_style 1 --seed 10 --thresh 1000 --hidden_dim 500 --embedding_dim 400 --lr_p 0.0005 --lr_q 0.001 --sample_size 3 --dual_attn yes --trans_unif yes  --test --task wb_global

```
(Static Constraints PR)
```
python train.py --data data/wb_aligned/ --load path/to/loaded/model --save_out path/to/output/file --data_mode real --optim_algo 1 --L 8 --decoder crf --cuda --one_rnn --sep_attn --option beam --posterior_reg 1  --layers 2 --train_q_epoch 5 --weight_decay 0.0 --full_independence 3 --pr_reg_style wb:cluster --bsz 10 --pr_coef 15 --hard_code no --decoder_constraint no --encoder_constraint yes --tagset_size 70 --max_mbs_per_epoch 25000 --use_elmo no --elmo_style 1 --seed 10 --thresh 1000 --hidden_dim 500 --embedding_dim 400 --lr_p 0.0005 --lr_q 0.001 --sample_size 3 --dual_attn yes --trans_unif yes  --test --task wb_lex

```
