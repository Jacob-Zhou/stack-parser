# -*- coding: utf-8 -*-

from parser import BiaffineParser, Model
from parser.utils import Corpus
from parser.utils.data import TextDataset, batchify

import torch


class Evaluate(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--batch-size', default=5000, type=int,
                               help='batch size')
        subparser.add_argument('--buckets', default=64, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--include-punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--fdata', default='data/test.gold.conllx',
                               help='path to dataset')
        subparser.set_defaults(func=self)

        return subparser

    def __call__(self, args):
        print("Load the model")
        vocab = torch.load(args.vocab)
        network = BiaffineParser.load(args.file)
        model = Model(vocab, network)

        print("Load the dataset")
        corpus = Corpus.load(args.fdata)
        dataset = TextDataset(vocab.numericalize(corpus), args.buckets)
        # set the data loader
        loader = batchify(dataset, args.batch_size, args.buckets)

        print("Evaluate the dataset")
        loss, metric_t, metric_p = model.evaluate(loader, args.include_punct)
        print(f"Loss: {loss:.4f} {metric_t}, {metric_p}")
