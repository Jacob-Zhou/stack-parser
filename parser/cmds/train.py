# -*- coding: utf-8 -*-

import os
from parser import BiaffineParser, Model
from parser.utils import Corpus, Embedding, Vocab
from parser.utils.data import TextDataset, batchify

import torch

from config import Config


class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--buckets', default=64, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--ftrain', default='data/train.auto.conllx',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/dev.auto.conllx',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/test.auto.conllx',
                               help='path to test file')
        subparser.add_argument('--fembed', default='data/giga.100.txt',
                               help='path to pretrained embedding file')
        subparser.set_defaults(func=self)

        return subparser

    def __call__(self, args):
        print("Preprocess the data")
        train = Corpus.load(args.ftrain)
        dev = Corpus.load(args.fdev)
        test = Corpus.load(args.ftest)
        if os.path.exists(args.vocab):
            vocab = torch.load(args.vocab)
        else:
            vocab = Vocab.from_corpus(corpus=train, min_freq=2)
            vocab.read_embeddings(embed=Embedding.load(args.fembed))
            torch.save(vocab, args.vocab)
        print(vocab)

        print("Load the dataset")
        trainset = TextDataset(vocab.numericalize(train))
        devset = TextDataset(vocab.numericalize(dev))
        testset = TextDataset(vocab.numericalize(test))
        # set the data loaders
        train_loader = batchify(dataset=trainset,
                                batch_size=Config.batch_size,
                                n_buckets=args.buckets,
                                shuffle=True)
        dev_loader = batchify(dataset=devset,
                              batch_size=Config.batch_size,
                              n_buckets=args.buckets)
        test_loader = batchify(dataset=testset,
                               batch_size=Config.batch_size,
                               n_buckets=args.buckets)
        print(f"{'train:':6} {len(trainset):5} sentences in total, "
              f"{len(train_loader):3} batches provided")
        print(f"{'dev:':6} {len(devset):5} sentences in total, "
              f"{len(dev_loader):3} batches provided")
        print(f"{'test:':6} {len(testset):5} sentences in total, "
              f"{len(test_loader):3} batches provided")

        print("Create the model")
        params = {
            'n_words': vocab.n_train_words,
            'n_embed': Config.n_embed,
            'n_tags': vocab.n_tags,
            'n_tag_embed': Config.n_tag_embed,
            'embed_dropout': Config.embed_dropout,
            'n_lstm_hidden': Config.n_lstm_hidden,
            'n_lstm_layers': Config.n_lstm_layers,
            'lstm_dropout': Config.lstm_dropout,
            'n_mlp_arc': Config.n_mlp_arc,
            'n_mlp_rel': Config.n_mlp_rel,
            'mlp_dropout': Config.mlp_dropout,
            'n_rels': vocab.n_rels,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        }
        for k, v in params.items():
            print(f"  {k}: {v}")
        network = BiaffineParser(params, vocab.embeddings)
        if torch.cuda.is_available():
            network = network.cuda()
        print(f"{network}\n")

        model = Model(vocab, network)
        model(loaders=(train_loader, dev_loader, test_loader),
              epochs=Config.epochs,
              patience=Config.patience,
              lr=Config.lr,
              betas=(Config.beta_1, Config.beta_2),
              epsilon=Config.epsilon,
              annealing=lambda x: Config.decay ** (x / Config.decay_steps),
              file=args.file)
