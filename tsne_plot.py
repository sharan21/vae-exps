import os
import json
import torch
import argparse
from torch.utils.data import Dataset
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from model_yelp_ortho import SentenceVaeStyleOrtho
from yelpd import Yelpd
from utils import to_var, idx2word, interpolate, load_model_params_from_checkpoint


def main(args):

    # load params
    params = load_model_params_from_checkpoint(args.load_params)

    # create model
    model = SentenceVaeStyleOrtho(**params)

    print(model)
    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)
    # splits = ['train', 'test']
    splits = ['train']

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    datasets = OrderedDict()
    tsne_values = np.empty((0,256), int)
    tsne_labels = np.empty((0,2), int)


    for split in splits:
        print("creating dataset for: {}".format(split))
        datasets[split] = Yelpd(
            split=split,
            create_data=args.create_data,
            min_occ=args.min_occ
        )

    for split in splits:

            # create dataloader
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split == 'train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
            for iteration, batch in enumerate(data_loader):

                # get batch size
                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)
                
                hidden_emb=model.tsne_plot(batch['input'])
                batch_labels=batch['label'].cpu().detach().numpy()
                hidden_emb=hidden_emb.squeeze().cpu().detach().numpy()
                tsne_values= np.append(tsne_values, hidden_emb, axis=0)
                tsne_labels=np.append(tsne_labels, batch_labels, axis=0)

                if iteration==3:
                    break
            
    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(tsne_values)
    tsne = TSNE(n_components=2, verbose = 1)
    tsne_results = tsne.fit_transform(pca_result[:])
    color_map = np.argmax(tsne_labels, axis=1)
    plt.figure(figsize=(10,10))
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl)
    plt.legend()
    plt.savefig('trial1.png')
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-p', '--load_params', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=40)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--min_occ', type=int, default=2)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
