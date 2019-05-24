# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
from parser import BiaffineParser, Model
from parser.metrics import AttachmentMethod
from parser.utils import Corpus, Embedding, Vocab
from parser.utils.data import TextDataset, batchify

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--pos', default=0, type=int,
                               help='max num of sentences in fpos to use')
        subparser.add_argument('--buckets', default=64, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--fpos', default='data/pku/train',
                               help='path to pos file')
        subparser.add_argument('--ftrain', default='data/conll09/train.conllx',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/conll09/dev.conllx',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/conll09/test.conllx',
                               help='path to test file')
        subparser.add_argument('--fembed', default='data/giga.100.txt',
                               help='path to pretrained embedding file')

        return subparser

    def __call__(self, config):
        print("Preprocess the data")
        tag_train = Corpus.load(config.fpos, columns=[1, 4], length=config.pos)
        dep_train = Corpus.load(config.ftrain)
        dev = Corpus.load(config.fdev)
        test = Corpus.load(config.ftest)
        if os.path.exists(config.vocab):
            vocab = torch.load(config.vocab)
        else:
            vocab = Vocab.from_corpora(tag_train, dep_train, 2)
            vocab.read_embeddings(Embedding.load(config.fembed))
            torch.save(vocab, config.vocab)
        config.update({
            'n_words': vocab.n_train_words,
            'n_chars': vocab.n_chars,
            'n_pos_tags': vocab.n_pos_tags,
            'n_dep_tags': vocab.n_dep_tags,
            'n_rels': vocab.n_rels,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })
        print(vocab)

        print("Load the dataset")
        tag_trainset = TextDataset(vocab.numericalize(tag_train, False))
        dep_trainset = TextDataset(vocab.numericalize(dep_train))
        devset = TextDataset(vocab.numericalize(dev))
        testset = TextDataset(vocab.numericalize(test))
        print(f"Set the data loaders")
        # set the data loaders
        tag_train_loader = batchify(dataset=tag_trainset,
                                    batch_size=config.pos_batch_size,
                                    n_buckets=config.buckets,
                                    shuffle=True)
        dep_train_loader = batchify(dataset=dep_trainset,
                                    batch_size=config.batch_size,
                                    n_buckets=config.buckets,
                                    shuffle=True)
        dev_loader = batchify(dataset=devset,
                              batch_size=config.batch_size,
                              n_buckets=config.buckets)
        test_loader = batchify(dataset=testset,
                               batch_size=config.batch_size,
                               n_buckets=config.buckets)
        print(f"{'tag_train:':10} {len(tag_trainset):7} sentences in total, "
              f"{len(tag_train_loader):4} batches provided")
        print(f"{'dep_train:':10} {len(dep_trainset):7} sentences in total, "
              f"{len(dep_train_loader):4} batches provided")
        print(f"{'dev:':10} {len(devset):7} sentences in total, "
              f"{len(dev_loader):4} batches provided")
        print(f"{'test:':10} {len(testset):7} sentences in total, "
              f"{len(test_loader):4} batches provided")

        print("Create the model")
        parser = BiaffineParser(config, vocab.embeddings)
        if torch.cuda.is_available():
            parser = parser.cuda()
        print(f"{parser}\n")

        model = Model(vocab, parser)

        total_time = timedelta()
        best_e, best_metric = 1, AttachmentMethod()
        model.optimizer = Adam(model.parser.parameters(),
                               config.lr,
                               (config.mu, config.nu),
                               config.epsilon)
        model.scheduler = ExponentialLR(model.optimizer,
                                        config.decay ** (1 / config.steps))
        model.count = 0

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            model.train(tag_train_loader, dep_train_loader)
            print(f"Epoch {epoch} / {config.epochs}:")
            loss, metric_t, metric_p = model.evaluate(dep_train_loader)
            print(f"{'train:':6} Loss: {loss:.4f} {metric_t} {metric_p}")
            loss, dev_metric_t, dev_metric_p = model.evaluate(dev_loader)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric_t} {dev_metric_p}")
            loss, metric_t, metric_p = model.evaluate(test_loader)
            print(f"{'test:':6} Loss: {loss:.4f} {metric_t} {metric_p}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric_p > best_metric and epoch > config.patience:
                best_e, best_metric = epoch, dev_metric_p
                model.parser.save(config.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break
        model.parser = BiaffineParser.load(config.model)
        loss, metric_t, metric_p = model.evaluate(test_loader)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric_p.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
