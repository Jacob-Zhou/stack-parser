# -*- coding: utf-8 -*-

import unicodedata
from collections import Counter

import torch


class Vocab(object):
    pad = '<pad>'
    unk = '<unk>'

    def __init__(self, words, chars, pos_tags, dep_tags, rels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.pad, self.unk] + sorted(words)
        self.chars = [self.pad, self.unk] + sorted(chars)
        self.pos_tags = sorted(pos_tags)
        self.dep_tags = sorted(dep_tags)
        self.rels = sorted(rels)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.char_dict = {char: i for i, char in enumerate(self.chars)}
        self.pos_tag_dict = {tag: i for i, tag in enumerate(self.pos_tags)}
        self.dep_tag_dict = {tag: i for i, tag in enumerate(self.dep_tags)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}

        # ids of punctuation that appear in words
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if self.is_punctuation(word))

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_pos_tags = len(self.pos_tags)
        self.n_dep_tags = len(self.dep_tags)
        self.n_rels = len(self.rels)
        self.n_init = self.n_words

    def __repr__(self):
        s = f"{self.__class__.__name__}: "
        s += f"{self.n_words} words, "
        s += f"{self.n_chars} chars, "
        s += f"{self.n_pos_tags} pos_tags, "
        s += f"{self.n_dep_tags} dep_tags, "
        s += f"{self.n_rels} rels"

        return s

    def word2id(self, sequence):
        return torch.tensor([self.word_dict.get(word.lower(), self.unk_index)
                             for word in sequence])

    def char2id(self, sequence, max_length=20):
        char_ids = torch.zeros(len(sequence), max_length, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor([self.char_dict.get(c, self.unk_index)
                                for c in word[:max_length]])
            char_ids[i, :len(ids)] = ids

        return char_ids

    def pos_tag2id(self, sequence):
        return torch.tensor([self.pos_tag_dict.get(tag, 0)
                             for tag in sequence])

    def dep_tag2id(self, sequence):
        return torch.tensor([self.dep_tag_dict.get(tag, 0)
                             for tag in sequence])

    def rel2id(self, sequence):
        return torch.tensor([self.rel_dict.get(rel, 0)
                             for rel in sequence])

    def id2tag(self, ids):
        return [self.dep_tags[i] for i in ids]

    def id2rel(self, ids):
        return [self.rels[i] for i in ids]

    def read_embeddings(self, embed, smooth=True):
        words = [word.lower() for word in embed.tokens]
        # if the `unk` token has existed in the pretrained,
        # then replace it with a self-defined one
        if embed.unk:
            words[embed.unk_index] = self.unk

        self.extend(words)
        self.embed = torch.zeros(self.n_words, embed.dim)
        self.embed[self.word2id(words)] = embed.vectors

        if smooth:
            self.embed /= torch.std(self.embed)

    def extend(self, words):
        self.words += sorted(set(words).difference(self.word_dict))
        self.chars += sorted(set(''.join(words)).difference(self.char_dict))
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if self.is_punctuation(word))
        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    def numericalize(self, corpus, dep=True, training=True):
        words = [self.word2id(seq) for seq in corpus.words]
        chars = [self.char2id(seq) for seq in corpus.words]
        if not training:
            return words, chars
        if not dep:
            tags = [self.pos_tag2id(seq) for seq in corpus.tags]
            return words, chars, tags
        else:
            tags = [self.dep_tag2id(seq) for seq in corpus.tags]
            arcs = [torch.tensor(seq) for seq in corpus.heads]
            rels = [self.rel2id(seq) for seq in corpus.rels]
            return words, chars, tags, arcs, rels

    @classmethod
    def from_corpora(cls, tag_corpus, dep_corpus, min_freq=1):
        word_seqs = tag_corpus.words + dep_corpus.words
        words = Counter(word.lower() for seq in word_seqs for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in word_seqs for char in ''.join(seq)})
        pos_tags = list({tag for seq in tag_corpus.tags for tag in seq})
        dep_tags = list({tag for seq in dep_corpus.tags for tag in seq})
        rels = list({rel for seq in dep_corpus.rels for rel in seq})
        vocab = cls(words, chars, pos_tags, dep_tags, rels)

        return vocab

    @classmethod
    def is_punctuation(cls, word):
        return all(unicodedata.category(char).startswith('P') for char in word)
