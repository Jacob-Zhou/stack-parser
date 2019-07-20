# -*- coding: utf-8 -*-

from collections import namedtuple

import torch


Sentence = namedtuple(typename='Sentence',
                      field_names=['ID', 'FORM', 'LEMMA', 'CPOS',
                                   'POS', 'FEATS', 'HEAD', 'DEPREL',
                                   'PHEAD', 'PDEPREL'],
                      defaults=[None]*10)


class Corpus(object):
    root = '<ROOT>'

    def __init__(self, sentences):
        super(Corpus, self).__init__()

        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(
            '\n'.join('\t'.join(map(str, i))
                      for i in zip(*(f for f in sentence if f))) + '\n'
            for sentence in self
        )

    def __getitem__(self, index):
        return self.sentences[index]

    @property
    def words(self):
        if not self.sentences[0].FORM:
            raise AttributeError
        return [[self.root] + list(sentence.FORM) for sentence in self]

    @property
    def tags(self):
        if not self.sentences[0].POS:
            raise AttributeError
        return [[self.root] + list(sentence.POS) for sentence in self]

    @property
    def heads(self):
        if not self.sentences[0].HEAD:
            raise AttributeError
        return [[0] + list(map(int, sentence.HEAD)) for sentence in self]

    @property
    def rels(self):
        if not self.sentences[0].DEPREL:
            raise AttributeError
        return [[self.root] + list(sentence.DEPREL) for sentence in self]

    @tags.setter
    def tags(self, sequences):
        self.sentences = [sentence._replace(CPOS=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @heads.setter
    def heads(self, sequences):
        self.sentences = [sentence._replace(HEAD=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @rels.setter
    def rels(self, sequences):
        self.sentences = [sentence._replace(DEPREL=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @classmethod
    def load(cls, fname, columns=range(10), length=0):
        start, sentences = 0, []
        names = [Sentence._fields[col] for col in columns]
        with open(fname, 'r') as f:
            lines = [line.strip() for line in f]
        for i, line in enumerate(lines):
            if not line:
                values = zip(*[l.split() for l in lines[start:i]])
                sentence = Sentence(**dict(zip(names, values)))
                sentences.append(sentence)
                start = i + 1
        if length > 0:
            indices = torch.randperm(len(sentences)).tolist()[:length]
            sentences = [sentences[i] for i in sorted(indices)]
        corpus = cls(sentences)

        return corpus

    def save(self, fname):
        with open(fname, 'w') as f:
            f.write(f"{self}\n")
