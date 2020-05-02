'''
    Xiang Li
    xli150@jhu.edu
'''

import sys
import re
from collections import Counter
from collections import defaultdict
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from allennlp.modules.elmo import Elmo, batch_to_ids
import os
from elmoformanylangs import Embedder

class Embedding_Weight():
    def __init__(self, embedding_source, path=None, data_loader=None, num_sent=None):
        self.embedding_source = 'elmo'

    def file_reader(self, path):
        with open(path, 'r') as f:
            for line in f:
                line = line.split()
                self.word_map[line[0]] = np.array(line[1:])
        self.embedding_size = len(self.word_map[line[0]])
        return

    def gen_word_embed(self, data_loader):
        '''
            return a Numpy matrix.
        '''
        result = np.zeros([len(data_loader.word_dict), self.embedding_size])
        for word, val in data_loader.word_dict.items():
            if word in self.word_map:
                result[val, :] = self.word_map[word]
            else:
                result[val, :] = np.zeros(self.embedding_size)
        return result

        embed = nn.Embedding(num_embeddings, embedding_dim)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def elmo_embeddings(self, corpus, number_sent, lang='en', args=None):
        print('complicated_layers_extraction')
        # with open(temp_file, 'w')
        if lang == 'en':
            system_config = '/home-4/xli150@jhu.edu/mutualinfo/others/elmo_2x4096_512_2048cnn_2xhighway_options.json'
            system_weight = '/home-4/xli150@jhu.edu/mutualinfo/others/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
            config_isfile = os.path.isfile(system_config)
            weight_isfile = os.path.isfile(system_weight)
            options_file = system_config if config_isfile else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = system_weight if weight_isfile else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            save_embeddings = []
            # Compute two different representation for each token.
            # Each representation is a linear weighted combination for the
            # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))

        elmo = Elmo(options_file, weight_file, 1, dropout=0)

        '''
            (batch, sequence_length, 50)
            returns dict 
        '''
        # elmo._elmo_bilm._token_embedder()

        first_layer_lst = []
        second_layer_lst = []
        third_layer_lst = []
        for index, sent in enumerate(corpus[:number_sent]):


            # print(sentences)
            character_ids = batch_to_ids(sent)
            # embeddings = elmo._elmo_lstm._token_embedder(character_ids)
            bilm_output = elmo._elmo_lstm(character_ids)
            temp = bilm_output['activations']
            first_layer = temp[0].squeeze(0)[1:].unsqueeze(1)
            second_layer = temp[1].squeeze(0)[1:].unsqueeze(1)
            third_layer = temp[2].squeeze(0)[1:].unsqueeze(1)

            first_layer_lst.append(first_layer)
            second_layer_lst.append(second_layer)
            third_layer_lst.append(third_layer)

            if index % 1000 == 0:
                sys.stdout.write('-')
                sys.stdout.flush()
        sys.stdout.write('\n')

        return first_layer_lst, second_layer_lst, third_layer_lst

    def elmo_embeddings_first(self, processed_sent, number_sent,
                              store_file=None):
        # with open(temp_file, 'w')
        try:
            with open(store_file, 'rb') as f:
                save_embeddings = pickle.load(f)

            return save_embeddings



        except:
            system_config = '/home-4/xli150@jhu.edu/mutualinfo/others/elmo_2x4096_512_2048cnn_2xhighway_options.json'
            system_weight = '/home-4/xli150@jhu.edu/mutualinfo/others/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
            config_isfile = os.path.isfile(system_config)
            weight_isfile = os.path.isfile(system_weight)
            options_file = system_config if config_isfile else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = system_weight if weight_isfile else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            save_embeddings = []
            # Compute two different representation for each token.
            # Each representation is a linear weighted combination for the
            # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
            elmo = Elmo(options_file, weight_file, 1, dropout=0)

            '''
                (batch, sequence_length, 50)
                returns dict 
            '''
            for index, sent in enumerate(processed_sent[:number_sent]):
                sentences = [list(sent)]  # create a list of size one of a list.
                character_ids = batch_to_ids(sentences)
                embeddings = elmo._elmo_lstm._token_embedder(character_ids)
                if index % 1000 == 0:
                    sys.stdout.write('-')
                    sys.stdout.flush()
                save_embeddings.append(embeddings["token_embedding"].squeeze(0)[1:-1].unsqueeze(1))
            sys.stdout.write('\n')

            return save_embeddings

    def helper(self, ):
        options_file = 'ELMo/elmo_2x1024_128_2048cnn_1xhighway_weights.json'
        weight_file = 'ELMo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        elmo = Elmo(options_file, weight_file, 2, dropout=0)

        # use batch_to_ids to convert sentences to character ids
        sentences = [['First', 'sentence', '.'], ['Another', '.']]
        character_ids = batch_to_ids(sentences)

        embeddings = elmo(character_ids)

        # embeddings['elmo_representations'] is length two list of tensors.
        # Each element contains one layer of ELMo representations with shape
        # (2, 3, 1024).
        #   2    - the batch size
        #   3    - the sequence length of the batch
        #   1024 - the length of each ELMo vector

if __name__ == '__main__':
    print('simple experiment with ELMo and allenNLP library')
    embedder = Embedding_Weight('elmo')

