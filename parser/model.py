# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from parser.metrics import AccuracyMethod, AttachmentMethod
from parser.parser import BiaffineParser

import torch
import torch.nn as nn
import torch.optim as optim


class Model(object):

    def __init__(self, vocab, network):
        super(Model, self).__init__()

        self.vocab = vocab
        self.network = network
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, loaders, epochs, patience,
                 lr, betas, epsilon, annealing, file):
        total_time = timedelta()
        best_e, best_metric = 1, AttachmentMethod()
        train_loader, dev_loader, test_loader = loaders
        self.optimizer = optim.Adam(params=self.network.parameters(),
                                    lr=lr, betas=betas, eps=epsilon)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                     lr_lambda=annealing)

        for epoch in range(1, epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            self.train(train_loader)

            print(f"Epoch {epoch} / {epochs}:")
            loss, metric_t, metric_p = self.evaluate(train_loader)
            print(f"{'train:':6} Loss: {loss:.4f} {metric_t} {metric_p}")
            loss, metric_t, dev_metric_p = self.evaluate(dev_loader)
            print(f"{'dev:':6} Loss: {loss:.4f} {metric_t} {dev_metric_p}")
            loss, metric_t, metric_p = self.evaluate(test_loader)
            print(f"{'test:':6} Loss: {loss:.4f} {metric_t} {metric_p}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric_p > best_metric and epoch > patience:
                best_e, best_metric = epoch, dev_metric_p
                self.network.save(file + f".{best_e}")
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= patience:
                break
        self.network = BiaffineParser.load(file + f".{best_e}")
        loss, metric_t, metric_p = self.evaluate(test_loader)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric_p.score:.2%}")
        print(f"mean time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

    def train(self, loader):
        self.network.train()

        for words, tags, arcs, rels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag, s_arc, s_rel = self.network(words)
            s_tag, s_arc, s_rel = s_tag[mask], s_arc[mask], s_rel[mask]
            gold_tags = tags[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]

            loss = self.get_loss(s_tag, s_arc, s_rel,
                                 gold_tags, gold_arcs, gold_rels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.network.eval()

        loss, metric_t, metric_p = 0, AccuracyMethod(), AttachmentMethod()

        for words, tags, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag, s_arc, s_rel = self.network(words)
            s_tag, s_arc, s_rel = s_tag[mask], s_arc[mask], s_rel[mask]
            gold_tags, pred_tags = tags[mask], s_tag.argmax(dim=-1)
            gold_arcs, gold_rels = arcs[mask], rels[mask]
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)

            loss += self.get_loss(s_tag, s_arc, s_rel,
                                  gold_tags, gold_arcs, gold_rels)
            metric_t(pred_tags, gold_tags)
            metric_p(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(loader)

        return loss, metric_t, metric_p

    @torch.no_grad()
    def predict(self, loader):
        self.network.eval()

        all_arcs, all_rels = [], []
        for words, tags in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_tag, s_arc, s_rel = self.network(words)
            s_tag, s_arc, s_rel = s_tag[mask], s_arc[mask], s_rel[mask]
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)

            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels

    def get_loss(self, s_tag, s_arc, s_rel, gold_tags, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]

        tag_loss = self.criterion(s_tag, gold_tags)
        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = tag_loss + arc_loss + rel_loss

        return loss

    def decode(self, s_arc, s_rel):
        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)

        return pred_arcs, pred_rels
