# -*- coding: utf-8 -*-

from collections import Counter

import regex
import torch


class Vocab(object):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, words, chars, p_tags, d_tags, rels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.PAD, self.UNK] + sorted(words)
        self.chars = [self.PAD, self.UNK] + sorted(chars)
        self.p_tags = sorted(p_tags)
        self.d_tags = sorted(d_tags)
        self.rels = sorted(rels)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.char_dict = {char: i for i, char in enumerate(self.chars)}
        self.p_tag_dict = {tag: i for i, tag in enumerate(self.p_tags)}
        self.d_tag_dict = {tag: i for i, tag in enumerate(self.d_tags)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}

        # ids of punctuation that appear in words
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_p_tags = len(self.p_tags)
        self.n_d_tags = len(self.d_tags)
        self.n_rels = len(self.rels)
        self.n_train_words = self.n_words

    def __repr__(self):
        info = f"{self.__class__.__name__}: "
        info += f"{self.n_words} words, "
        info += f"{self.n_chars} chars, "
        info += f"{self.n_p_tags} p_tags, "
        info += f"{self.n_d_tags} d_tags, "
        info += f"{self.n_rels} rels"

        return info

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

    def p_tag2id(self, sequence):
        return torch.tensor([self.p_tag_dict.get(tag, 0)
                             for tag in sequence])

    def d_tag2id(self, sequence):
        return torch.tensor([self.d_tag_dict.get(tag, 0)
                             for tag in sequence])

    def rel2id(self, sequence):
        return torch.tensor([self.rel_dict.get(rel, 0)
                             for rel in sequence])

    def id2tag(self, ids):
        return [self.tags[i] for i in ids]

    def id2rel(self, ids):
        return [self.rels[i] for i in ids]

    def read_embeddings(self, embed, smooth=True):
        # if the UNK token has existed in the pretrained,
        # then use it to replace the one in the vocab
        if embed.unk:
            self.UNK = embed.unk

        self.extend(embed.tokens)
        self.embeddings = torch.zeros(self.n_words, embed.dim)

        for i, word in enumerate(self.words):
            if word in embed:
                self.embeddings[i] = embed[word]
        if smooth:
            self.embeddings /= torch.std(self.embeddings)

    def extend(self, words):
        self.words += sorted(set(words).difference(self.word_dict))
        self.chars += sorted(set(''.join(words)).difference(self.char_dict))
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))
        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    def numericalize(self, corpus, dep=True, training=True):
        words = [self.word2id(seq) for seq in corpus.words]
        chars = [self.char2id(seq) for seq in corpus.words]
        if not training:
            return words, chars
        if not dep:
            tags = [self.p_tag2id(seq) for seq in corpus.tags]
            return words, chars, tags
        else:
            tags = [self.d_tag2id(seq) for seq in corpus.tags]
            arcs = [torch.tensor(seq) for seq in corpus.heads]
            rels = [self.rel2id(seq) for seq in corpus.rels]
            return words, chars, tags, arcs, rels

    @classmethod
    def from_corpora(cls, tag_corpus, dep_corpus, min_freq=1):
        word_seqs = tag_corpus.words + dep_corpus.words
        words = Counter(word.lower() for seq in word_seqs for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in word_seqs for char in ''.join(seq)})
        p_tags = list({tag for seq in tag_corpus.tags for tag in seq})
        d_tags = list({tag for seq in dep_corpus.tags for tag in seq})
        rels = list({rel for seq in dep_corpus.rels for rel in seq})
        vocab = cls(words, chars, p_tags, d_tags, rels)

        return vocab
