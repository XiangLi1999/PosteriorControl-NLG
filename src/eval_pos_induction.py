from sklearn.metrics.cluster import v_measure_score
import argparse
from Chunking_Reader import Data_Loader_Chunk, Data_Loader_ptb
import numpy as np
from collections import defaultdict, Counter

parser = argparse.ArgumentParser(description='')
parser.add_argument('--tagset_size', type=int, default=45, help='path to save the final model')
parser.add_argument('--max_seqlen', type=int, default=70, help='')
parser.add_argument('--thresh', default=3, type=int, help='do not decay learning rate for at least this many epochs')
parser.add_argument('--data_ptb_path', type=str, default='../data/ptb', help='path to saved model')
parser.add_argument('--induced_path', type=str, default='', help='path to saved model')
parser.add_argument('--test', action='store_true', help='use test data')
args = parser.parse_args()


def purity_measure(dict_true, dict_pred, word_count, state_true, state_pred):
	''' on average what's the most centered states '''
	total_measure = []
	correct_measure = []
	for word, (state_t, count_t) in dict_true.items():
		count_total = word_count[word]
		state_p, count_p = dict_pred[word]

		total_measure.append(count_t/count_total)
		correct_measure.append(count_p/count_total)

	purity_score = np.array(total_measure).mean() 
	true_purity_score = np.array(correct_measure).mean()

	print('pred purity_score is {}, and true purity score is {}'.format(purity_score, true_purity_score))


	''' state variety and state entropy '''
	# threshold > 10 in all occurences. 
	state_count_dict = {}
	all_occ_count = 0
	type_occ_count = 0
	total_occur = 0
	for state, val_lst in state_true.items():
		state_count_dict[state] = len(val_lst)
		total_occur += len(val_lst)

		if len(val_lst) > 10:
			all_occ_count += 1

		if len(set(val_lst)) > 10:
			type_occ_count += 1

	state_count_dict2 = {}
	all_occ_count2 = 0
	type_occ_count2 = 0
	for state, val_lst in state_pred.items():
		state_count_dict2[state] = len(val_lst)
		if len(val_lst) > 10:
			all_occ_count2 += 1

		if len(set(val_lst)) > 10:
			type_occ_count2 += 1


	print('pred', state_count_dict2)
	print('true', state_count_dict)

	print('pred', all_occ_count2)
	print('true', all_occ_count)

	print('pred', type_occ_count2)
	print('true', type_occ_count)

	# entropy over states. 
	entr = 0
	for elem, val in state_count_dict.items():
		prob = val/total_occur
		entr -= prob * np.log(prob)

	entr2 = 0
	for elem, val in state_count_dict2.items():
		# state_count_dict[elem] += 1
		prob = val/total_occur
		entr2 -= prob * np.log(prob)

	print('entropy of true', entr)
	print('entropy of pred', entr2)

	return 

def dict_renew(word_dict):
	new_word_dict = {}
	for word, lst in word_dict.items():
		temp_dict = Counter()
		for elem in lst:
			temp_dict[elem] += 1
		top_x =temp_dict.most_common(1)[0]
		# print(top_x)
		new_word_dict[word] = top_x

	return new_word_dict


def induction_reader(path, dropend=False):
	state_pred = defaultdict(list)
	dict_pred = defaultdict(list)

	sent_full = []
	tags_full = []
	with open(path, 'r') as f:
		for line in f:
			# print(line)
			split_line = line.split('|||')

			try:
				sent = split_line[0].split()
				tags = split_line[1].split()
			except:
				continue
			if dropend:
				sent = sent[:-1]
				tags = tags[:-1]
				tags = [int(x) for x in tags]

				for x, y in zip(sent, tags):
					dict_pred[x].append(y)
					state_pred[y].append(x)

			else:
				tags = [int(x) for x in tags]
			sent_full.append(sent)
			tags_full.append(tags)

	print(len(sent_full))
	print(len(tags_full))

	return tags_full, sent_full, state_pred, dict_pred


def sanity_check(sent_ours, sent_gold):
	for our, gold in zip(sent_ours, sent_gold):
		try:
			assert len(our) == len(gold)
			assert tuple(our) == tuple(gold)
		except:
			print(our)
			print(gold)
			print()

	print('the ordering is okay')


def idx_tags(pos_lst):
	temp = []
	punct = corpus.pos_vocab['PUNCT']
	for pos_sent in pos_lst:
		temp.append([corpus.pos_vocab[pos] if pos in corpus.pos_vocab else punct for pos in pos_sent])
	return temp



def eval_avg(tags_ours, tags_gold):
	assert  len(tags_ours) == len(tags_gold)
	record = []
	for our, gold in zip(tags_ours, tags_gold):
		v_score = v_measure_score(our, gold)
		record.append(v_score)
	v_score_final = np.array(record).mean()
	return v_score_final

def eval_full(tags_ours, tags_gold):
	our_lst = []
	for elem in tags_ours:
		our_lst += elem

	gold_lst = []
	for elem in tags_gold:
		gold_lst += elem

	assert len(our_lst) == len(gold_lst)
	v_score = v_measure_score(our_lst, gold_lst)
	return v_score

def gold_reader(corpus, dropend=False):

	state_true = defaultdict(list)
	dict_true = defaultdict(list)
	word_count = defaultdict(int)

	if args.test:
		corpora = corpus.test
	else:
		corpora = corpus.valid
	full_pos = []
	full_x = []
	for i in range(len(corpora)):
		x, pos, chk, ner, x_str = corpora[i]
		# print(x_str)
		# print(pos)
		seqlen = len(x[0])
		if seqlen > args.max_seqlen:
			continue
		# print(x_str)
		if dropend:
			x_str = [y[:-1] for y in x_str]
			pos = [y for y in pos]

			for xx, yy in zip(x_str, pos):

				for x, y in zip(xx, yy):
					dict_true[x].append(y)
					state_true[y].append(x)
					word_count[x] += 1
		else:
			x_str = [y for y in x_str]
			pos = [y for y in pos]
		full_x.extend(x_str)
		full_pos.extend(pos)

	print(len(full_pos), len(full_x))
	return full_pos, full_x, state_true, dict_true, word_count


# need the measurement as follows. dict_true, dict_pred, word_count, state_true, state_pred

corpus = Data_Loader_ptb(args.data_ptb_path, 20, 1, args.tagset_size, max_len=args.max_seqlen, thresh=args.thresh,
						 test=args.test)

gold_pos, gold_x, state_true, dict_true, word_count = gold_reader(corpus, dropend=True)

ind_pos, ind_x, state_pred, dict_pred = induction_reader(args.induced_path, dropend=True)

sanity_check(ind_x, gold_x)

dict_true = dict_renew(dict_true)
dict_pred = dict_renew(dict_pred)

purity_measure(dict_true, dict_pred, word_count, state_true, state_pred)

# for elem in [set(val) for key, val in state_pred.items()]:
# 	print(elem)
# 	print()
# print()
# print()
# print()
# print(state_true)


print(gold_pos[0])
gold_pos = idx_tags(gold_pos)

print(gold_pos[0])
print(ind_pos[0])

score = eval_avg(ind_pos, gold_pos)
print('the average v score is {}'.format(score))
score = eval_full(ind_pos, gold_pos)
print('the full v score is {}'.format(score))





