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
        subparser.add_argument('--pos-batch-size', default=5000, type=int,
                               help='num of tokens per training update')
        subparser.add_argument('--patience', default=100, type=int,
                               help='patience for early stop')
        subparser.add_argument('--buckets', default=64, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--fptrain', default='data/pku/train',
                               help='path to pos train file')
        subparser.add_argument('--fpdev', default='data/pku/dev',
                               help='path to pos dev file')
        subparser.add_argument('--fptest', default='data/pku/test',
                               help='path to pos test file')
        subparser.add_argument('--ftrain', default='data/conll09/train.conllx',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/conll09/dev.conllx',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/conll09/test.conllx',
                               help='path to test file')
        subparser.add_argument('--fembed', default='data/giga.100.txt',
                               help='path to pretrained embedding file')
        subparser.add_argument('--weight', action='store_true',
                               help='whether to weighted sum the layers')

        return subparser

    def __call__(self, config):
        if config.preprocess:
            print("Preprocess the corpus")
            pos_train = Corpus.load(config.fptrain, [1, 4], config.pos)
            dep_train = Corpus.load(config.ftrain)
            pos_dev = Corpus.load(config.fpdev, [1, 4])
            dep_dev = Corpus.load(config.fdev)
            pos_test = Corpus.load(config.fptest, [1, 4])
            dep_test = Corpus.load(config.ftest)
            print("Create the vocab")
            vocab = Vocab.from_corpora(pos_train, dep_train, 2)
            vocab.read_embeddings(Embedding.load(config.fembed))
            print("Load the dataset")
            pos_trainset = TextDataset(vocab.numericalize(pos_train, False),
                                       config.buckets)
            dep_trainset = TextDataset(vocab.numericalize(dep_train),
                                       config.buckets)
            pos_devset = TextDataset(vocab.numericalize(pos_dev, False),
                                     config.buckets)
            dep_devset = TextDataset(vocab.numericalize(dep_dev),
                                     config.buckets)
            pos_testset = TextDataset(vocab.numericalize(pos_test, False),
                                      config.buckets)
            dep_testset = TextDataset(vocab.numericalize(dep_test),
                                      config.buckets)
            torch.save(vocab, config.vocab)
            torch.save(pos_trainset, os.path.join(config.file, 'pos_trainset'))
            torch.save(dep_trainset, os.path.join(config.file, 'dep_trainset'))
            torch.save(pos_devset, os.path.join(config.file, 'pos_devset'))
            torch.save(dep_devset, os.path.join(config.file, 'dep_devset'))
            torch.save(pos_testset, os.path.join(config.file, 'pos_testset'))
            torch.save(dep_testset, os.path.join(config.file, 'dep_testset'))
            return

        print("Load the vocab")
        vocab = torch.load(config.vocab)
        print("Load the datasets")
        pos_trainset = torch.load(os.path.join(config.file, 'pos_trainset'))
        dep_trainset = torch.load(os.path.join(config.file, 'dep_trainset'))
        pos_devset = torch.load(os.path.join(config.file, 'pos_devset'))
        dep_devset = torch.load(os.path.join(config.file, 'dep_devset'))
        pos_testset = torch.load(os.path.join(config.file, 'pos_testset'))
        dep_testset = torch.load(os.path.join(config.file, 'dep_testset'))
        config.update({
            'n_words': vocab.n_init,
            'n_chars': vocab.n_chars,
            'n_pos_tags': vocab.n_pos_tags,
            'n_dep_tags': vocab.n_dep_tags,
            'n_rels': vocab.n_rels,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })
        # set the data loaders
        pos_train_loader = batchify(pos_trainset,
                                    config.pos_batch_size//config.update_steps,
                                    True)
        dep_train_loader = batchify(dep_trainset,
                                    config.batch_size//config.update_steps,
                                    True)
        pos_dev_loader = batchify(pos_devset, config.pos_batch_size)
        dep_dev_loader = batchify(dep_devset, config.batch_size)
        pos_test_loader = batchify(pos_testset, config.pos_batch_size)
        dep_test_loader = batchify(dep_testset, config.batch_size)

        print(vocab)
        print(f"{'pos_train:':10} {len(pos_trainset):7} sentences in total, "
              f"{len(pos_train_loader):4} batches provided")
        print(f"{'dep_train:':10} {len(dep_trainset):7} sentences in total, "
              f"{len(dep_train_loader):4} batches provided")
        print(f"{'pos_dev:':10} {len(pos_devset):7} sentences in total, "
              f"{len(pos_dev_loader):4} batches provided")
        print(f"{'dep_dev:':10} {len(dep_devset):7} sentences in total, "
              f"{len(dep_dev_loader):4} batches provided")
        print(f"{'pos_test:':10} {len(pos_testset):7} sentences in total, "
              f"{len(pos_test_loader):4} batches provided")
        print(f"{'dep_test:':10} {len(dep_testset):7} sentences in total, "
              f"{len(dep_test_loader):4} batches provided")

        print("Create the model")
        parser = BiaffineParser(config, vocab.embed).to(config.device)
        print(f"{parser}\n")

        model = Model(config, vocab, parser)

        total_time = timedelta()
        best_e, best_metric = 1, AttachmentMethod()
        model.optimizer = Adam(model.parser.parameters(),
                               config.lr,
                               (config.mu, config.nu),
                               config.epsilon)
        model.scheduler = ExponentialLR(model.optimizer,
                                        config.decay**(1/config.decay_steps))

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            model.train(pos_train_loader, dep_train_loader)
            print(f"Epoch {epoch} / {config.epochs}:")
            lp, ld, mp, mdt, mdp = model.evaluate(None,
                                                  dep_train_loader)
            print(f"{'train:':6} LP: {lp:.4f} LD: {ld:.4f} {mp} {mdt} {mdp}")
            lp, ld, mp, mdt, dev_m = model.evaluate(pos_dev_loader,
                                                    dep_dev_loader)
            print(f"{'dev:':6} LP: {lp:.4f} LD: {ld:.4f} {mp} {mdt} {dev_m}")
            lp, ld, mp, mdt, mdp = model.evaluate(pos_test_loader,
                                                  dep_test_loader)
            print(f"{'test:':6} LP: {lp:.4f} LD: {ld:.4f} {mp} {mdt} {mdp}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_m > best_metric and epoch > config.patience:
                best_e, best_metric = epoch, dev_m
                model.parser.save(config.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break
        model.parser = BiaffineParser.load(config.model)
        lp, ld, mp, mdt, mdp = model.evaluate(pos_test_loader, dep_test_loader)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {mdp.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
