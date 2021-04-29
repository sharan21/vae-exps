import os
import json
import torch
import argparse
from collections import defaultdict
from model_multitask import SentenceVaeStyleOrtho
from utils import to_var, idx2word, interpolate, load_model_params_from_checkpoint
from snli_yelp import SnliYelp
# import num


def main(args):

    # check args
    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    if not os.path.exists(args.load_params):
        raise FileNotFoundError(args.load_params)


    # load dataset
    split = 'test'
    datasets = defaultdict(dict)

    datasets[split] = SnliYelp(
            split=split,
            create_data=False,
            min_occ=2)
    
    # load pretrained vocab
    with open('./data/snli_yelp/snli_yelp.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']


    # load params
    params = load_model_params_from_checkpoint(args.load_params)

    # create model
    model = SentenceVaeStyleOrtho(**params)

    print(model)
    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    # get random sentence 1 from snli
    sent1 = datasets[split].__getitem__(6)
    # get random sentence 1 from yelp
    sent2 = datasets[split].__getitem__(55)

    
    # get the lspace vectors for sent1 and sent2
    sent1_tokens = torch.tensor(sent1['input']).unsqueeze(0)
    sent2_tokens = torch.tensor(sent2['input']).unsqueeze(0)
    batch = torch.cat((sent1_tokens, sent1_tokens), 0).cuda()
    style_z, content_z = model.encode_to_lspace(batch)

    # print sent1 and 2
    print("sent 1:")
    print(*idx2word(sent1_tokens, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    print("sent 1:")
    print(*idx2word(sent2_tokens, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    # contact style_z of sent 1 to content_z of sent_2
    final_z = torch.cat((style_z[1], content_z[0]), -1).unsqueeze(0)
    samples, _ = model.inference(z=final_z)
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    # print(final_z.shape)    

    exit()
    
    # samples, z = model.inference(n=args.num_samples)
    # print('----------SAMPLES----------')
    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    # print('-------INTERPOLATION-------')
    # z1 = torch.randn([params['latent_size']]).numpy()
    # z2 = torch.randn([params['latent_size']]).numpy()
    # z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    # samples, _ = model.inference(z=z)
    
    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-p', '--load_params', type=str)
    # parser.add_argument('-n', '--num_samples', type=int, default=10)

    # parser.add_argument('-dd', '--data_dir', type=str, default='data')
    # parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    # parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    # parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    # parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    # parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    # parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    # parser.add_argument('-ls', '--latent_size', type=int, default=40)
    # parser.add_argument('-nl', '--num_layers', type=int, default=1)
    # parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    # args.rnn_type = args.rnn_type.lower()

    # assert args.rnn_type in ['rnn', 'lstm', 'gru']
    # assert 0 <= args.word_dropout <= 1

    main(args)
