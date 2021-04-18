import os
import io
import json
import torch
import pickle
import time
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from multiprocessing import cpu_count
from utils import to_var, idx2word, expierment_name
from torch.utils.data import DataLoader
from nltk.tokenize import TweetTokenizer

from collections import OrderedDict, defaultdict

from utils import OrderedCounter
from tqdm import tqdm

from model import SentenceVAE

import argparse


class Yelpd(Dataset):

    def __init__(self, split, create_data, have_vocab=False,  **kwargs):

        super().__init__()
        self.data_dir = "./data/yelp/"
        self.save_model_path = "./bin"
        self.split = split

        self.bow_hidden_dim = 7526

        # self.max_sequence_length = 1165
        self.max_sequence_length = 116
        self.min_occ = kwargs.get('min_occ', 3)

        # self.num_lines = 560000
        self.num_lines = 56000

        self.have_vocab = have_vocab

        self.raw_data_path = os.path.join(self.data_dir, 'yelp.'+split+'.csv')
        self.preprocessed_data_file = 'yelpd.'+split+'.json'
        self.vocab_file = 'yelpd.vocab.json'
         
         # load bow vocab
        with open("./bow.json") as f:
            self.bow_filtered_vocab_indices = json.load(f)


        if create_data:
            print("Creating new %s ptb data." % split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.preprocessed_data_file)):
            print("%s preprocessed file not found at %s. Creating new." % (
                split.upper(), os.path.join(self.data_dir, self.preprocessed_data_file)))
            self._create_data()

        else:
            print(" found preprocessed files, no need tooo create data!")
            self._load_data()
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'bow': self._get_bow_representations(self.data[idx]['input']),
            # 'label': np.asarray(self.data[idx]['label']),
            'label': np.asarray([1-self.data[idx]['label'], self.data[idx]['label']]), # we need to make it 2 dim to match predicted label dim.
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):

        print("loading preprocessed json data...")

        with open(os.path.join(self.data_dir, self.preprocessed_data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_data_2(self, vocab=True):

        print("loading preprocessed pickle data...")

        with open(os.path.join(self.data_dir, self.preprocessed_data_file), 'rb') as file:
            self.data = pickle.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'rb') as file:
                vocab = pickle.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if not self.have_vocab and self.split == 'train':
            print("creating vocab for train!")
            self._create_vocab()
            print("finished creating vocab!")
        else:
            self._load_vocab()
            print("loaded vocab from mem!")

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(tqdm(file, total=self.num_lines)):

                if(i == self.num_lines):
                    break

                # separate the label and the line
                label = float(line[1])
                line = line[4:]

                words = tokenizer.tokenize(line)

                input = ['<sos>'] + words
                input = input[:self.max_sequence_length]

                target = words[:self.max_sequence_length-1]
                target = target + ['<eos>']
        

                assert len(input) == len(target), "%i, %i" % (len(input), len(target))

                length = len(input)

                input.extend(['<pad>'] * (self.max_sequence_length-length))
                target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['label'] = label
                data[id]['target'] = target
                data[id]['length'] = length

                # we get batch['bow'] in get_item() instead as bow is too expensive to store

        with io.open(os.path.join(self.data_dir, self.preprocessed_data_file), 'wb') as preprocessed_data_file:
            data = json.dumps(data, ensure_ascii=False)
            preprocessed_data_file.write(data.encode('utf8', 'replace'))

            # using pickle instead of json
            # pickle.dump(data, preprocessed_data_file)
            # preprocessed_data_file.close()

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        print("here")

        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(tqdm(file, total=self.num_lines)):
                if(i == self.num_lines):
                    break
                words = tokenizer.tokenize(line)
                w2c.update(words)


            print("done creating w2c")
            for w, c in tqdm(w2c.items()):
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

            print("done creating w2i")

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." % len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()

    def convert_(self): #does not seemed to be used

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(file):

                words = tokenizer.tokenize(line)

                input = ['<sos>'] + words
                input = input[:self.max_sequence_length]

                target = words[:self.max_sequence_length-1]
                target = target + ['<eos>']

                assert len(input) == len(target), "%i, %i" % (
                    len(input), len(target))
                length = len(input)

                input.extend(['<pad>'] * (self.max_sequence_length-length))
                target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

        with io.open(os.path.join(self.data_dir, self.preprocessed_data_file), 'wb') as preprocessed_data_file:
            data = json.dumps(data, ensure_ascii=False)
            preprocessed_data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)
    
    def _get_bow_representations(self, text_sequence):
        """
        Returns BOW representation of every sequence of the batch
        """

        sequence_bow_representation = np.zeros(shape=self.bow_hidden_dim, dtype=np.float32)
        
        # Iterate over each word in the sequence
        for index in text_sequence:

            if index in self.bow_filtered_vocab_indices:
                bow_index = self.bow_filtered_vocab_indices[index]
                sequence_bow_representation[bow_index] += 1
        
        sequence_bow_representation /= np.max([np.sum(sequence_bow_representation), 1])

        return np.asarray(sequence_bow_representation)
