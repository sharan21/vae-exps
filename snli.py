import os
import io
import json
import jsonlines
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from collections import OrderedDict, defaultdict
from utils import OrderedCounter

class SNLI(Dataset):

    def __init__(self, split, create_data=False, **kwargs):

        super().__init__()
        self.data_dir = './data/snli/'
        self.save_model_path = './bin'
        self.split = split

        self.max_sequence_length = 50 #to avoid CUDA out of memory
        self.min_occ = kwargs.get('min_occ', 3)
        self.num_lines = 56000
        self.bow_hidden_dim = 7526

        self.raw_data_path = self.data_dir + 'snli_1.0_' + self.split + '.jsonl'
        self.data_file = 'snli.'+ split +'.json'
        self.vocab_file = 'snli.vocab.json'

        with open("./bow.json") as f:
            self.bow_filtered_vocab_indices = json.load(f)

        # if create_data:
        #     print("Creating new %s snli data."%split.upper())
        #     self._create_data()

        if not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            print(" found preprocessed files, no need tooo create data!")
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)
        one_hot = np.zeros(4)
        one_hot[self.data[idx]['label']]=1

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'bow': self._get_bow_representations(self.data[idx]['input']),
            'length': self.data[idx]['length'],
            'label': one_hot
            # 'label': np.asarray([1-self.data[idx]['label'], self.data[idx]['label']]),
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

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
        

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']


    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()
        # print("from_create_Data")
        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        with jsonlines.open(self.raw_data_path, 'r') as file:
            for i, line in enumerate(file):

                if(i == self.num_lines):
                    break

                strline=line['sentence1']+line['sentence2']

                words = tokenizer.tokenize(strline)

                input = ['<sos>'] + words
                input = input[:self.max_sequence_length]

                target = words[:self.max_sequence_length-1]
                target = target + ['<eos>']

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)

                input.extend(['<pad>'] * (self.max_sequence_length-length))
                target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]
                
                
                labels=['entailment','neutral','contradiction','-']
                labl=labels.index(line['gold_label'])
                

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length
                data[id]['label'] =labl

        
        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))
        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."
        # print("from _create_vocab_")

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with jsonlines.open(self.raw_data_path, 'r') as file:
            print("reading for vocan {0}".format(self.raw_data_path))
            for i, line in enumerate(file):
                words = tokenizer.tokenize(line['sentence1'])
                w2c.update(words)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)
        
        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()\
    
    def _get_bow_representations(self, text_sequence):
        """
        Returns BOW representation of every sequence of the batch
        """

        sequence_bow_representation = np.zeros(shape=self.bow_hidden_dim, dtype=np.float32)
     
        # Iterate over each word in the sequence
        for index in text_sequence:

            if str(index) in self.bow_filtered_vocab_indices:
                bow_index = self.bow_filtered_vocab_indices[str(index)]
                sequence_bow_representation[bow_index] += 1
        
        # removing normalisation because the loss becomes too low with it, anyway it wont change correctness
        sequence_bow_representation /= np.max([np.sum(sequence_bow_representation), 1])

        return np.asarray(sequence_bow_representation)
        
if __name__ == "__main__":
    # prepare dataset
    splits = ['train', 'test']

    # create dataset object
    datasets = OrderedDict()

    # create test and train split in data, also preprocess
    for split in splits:
        # print("creating dataset for: {}".format(split))
        datasets[split] = Snli(
            split=split,
            create_data=False,
            min_occ=2
        )

    # get training params
    # params = dict(
    #     vocab_size=datasets['train'].vocab_size,
    #     sos_idx=datasets['train'].sos_idx,
    #     eos_idx=datasets['train'].eos_idx,
    #     pad_idx=datasets['train'].pad_idx,
    #     unk_idx=datasets['train'].unk_idx,
    #     max_sequence_length=datasets['train'].max_sequence_length,
    #     embedding_size=args.embedding_size,
    #     rnn_type=args.rnn_type,
    #     hidden_size=args.hidden_size,
    #     word_dropout=args.word_dropout,
    #     embedding_dropout=args.embedding_dropout,
    #     latent_size=args.latent_size,
    #     num_layers=args.num_layers,
    #     bidirectional=args.bidirectional,
    #     ortho=ortho,
    #     attention=attention,
    #     hspace_classifier=hspace_classifier
    # )
    
