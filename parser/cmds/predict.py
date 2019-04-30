# -*- coding: utf-8 -*-

from parser import BiaffineParser, Model
from parser.utils import Corpus
from parser.utils.data import TextDataset, batchify

import torch


class Predict(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--batch-size', default=5000, type=int,
                               help='batch size')
        subparser.add_argument('--fdata', default='data/test.gold.conllx',
                               help='path to dataset')
        subparser.add_argument('--fpred', default='pred.conllx',
                               help='path to predicted result')

        return subparser

    def __call__(self, config):
        print("Load the model")
        vocab = torch.load(config.vocab)
        network = BiaffineParser.load(config.file)
        model = Model(vocab, network)

        print("Load the dataset")
        corpus = Corpus.load(config.fdata)
        dataset = TextDataset(vocab.numericalize(corpus, False))
        # set the data loader
        loader = batchify(dataset, config.batch_size)

        print("Predict the dataset")
        corpus.heads, corpus.rels = model.predict(loader)

        print(f"Save the predicted result")
        corpus.save(config.fpred)
