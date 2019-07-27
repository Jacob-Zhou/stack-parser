# -*- coding: utf-8 -*-

from parser.metrics import AccuracyMethod, AttachmentMethod

import torch
import torch.nn as nn


class Model(object):

    def __init__(self, config, vocab, parser):
        super(Model, self).__init__()

        self.config = config
        self.vocab = vocab
        self.parser = parser

    def train(self, pos_loader, dep_loader):
        self.parser.train()

        for i, (words, chars, tags, arcs, rels) in enumerate(dep_loader):
            try:
                pos_words, pos_chars, pos_tags = next(self.pos_iter)
            except Exception:
                self.pos_iter = iter(pos_loader)
                pos_words, pos_chars, pos_tags = next(self.pos_iter)
            mask = pos_words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag = self.parser(pos_words, pos_chars, False)
            loss = self.parser.criterion(s_tag[mask], pos_tags[mask])
            loss = loss / self.config.update_steps
            loss.backward()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag, s_arc, s_rel = self.parser(words, chars)
            s_tag, s_arc, s_rel = s_tag[mask], s_arc[mask], s_rel[mask]
            gold_tags = tags[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]

            loss = self.parser.get_loss(s_tag, s_arc, s_rel,
                                        gold_tags, gold_arcs, gold_rels)
            loss = loss / self.config.update_steps
            loss.backward()
            if (i + 1) % self.config.update_steps == 0:
                nn.utils.clip_grad_norm_(self.parser.parameters(),
                                         self.config.clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, pos_loader, dep_loader):
        self.parser.eval()

        pos_loss, dep_loss = 0, 0
        pos_metric = AccuracyMethod()
        dep_metric_t, dep_metric_p = AccuracyMethod(), AttachmentMethod()

        for words, chars, tags, arcs, rels in dep_loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag, s_arc, s_rel = self.parser(words, chars)
            s_tag, s_arc, s_rel = s_tag[mask], s_arc[mask], s_rel[mask]
            gold_tags, pred_tags = tags[mask], s_tag.argmax(dim=-1)
            gold_arcs, gold_rels = arcs[mask], rels[mask]
            pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel)
            dep_loss += self.parser.get_loss(s_tag, s_arc, s_rel,
                                             gold_tags, gold_arcs, gold_rels)
            dep_metric_t(pred_tags, gold_tags)
            dep_metric_p(pred_arcs, pred_rels, gold_arcs, gold_rels)
        dep_loss /= len(dep_loader)

        if pos_loader:
            for words, chars, tags in pos_loader:
                mask = words.ne(self.vocab.pad_index)
                # ignore the first token of each sentence
                mask[:, 0] = 0
                s_tag = self.parser(words, chars, False)
                gold_tags, pred_tags = tags[mask], s_tag.argmax(dim=-1)[mask]
                pos_loss += self.parser.criterion(s_tag[mask], tags[mask])
                pos_metric(pred_tags, gold_tags)
            pos_loss /= len(pos_loader)
        return pos_loss, dep_loss, pos_metric, dep_metric_t, dep_metric_p

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()

        all_tags, all_arcs, all_rels = [], [], []
        for words, chars in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_tag, s_arc, s_rel = self.parser(words, chars)
            s_tag, s_arc, s_rel = s_tag[mask], s_arc[mask], s_rel[mask]
            pred_tags = s_tag.argmax(dim=-1)
            pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel)

            all_tags.extend(torch.split(pred_tags, lens))
            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_tags, all_arcs, all_rels
