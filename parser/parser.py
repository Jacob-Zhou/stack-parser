# -*- coding: utf-8 -*-

from parser.modules import (CHAR_LSTM, MLP, Biaffine, BiLSTM,
                            IndependentDropout, SharedDropout)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class BiaffineParser(nn.Module):

    def __init__(self, config, embeddings):
        super(BiaffineParser, self).__init__()

        self.config = config
        # the embedding layer
        self.pretrained = nn.Embedding.from_pretrained(embeddings)
        self.embed = nn.Embedding(num_embeddings=config.n_words,
                                  embedding_dim=config.n_embed)
        # the char-lstm layer
        self.char_lstm = CHAR_LSTM(n_chars=config.n_chars,
                                   n_embed=config.n_char_embed,
                                   n_out=config.n_char_out)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        # the word-lstm layer
        self.lstm = BiLSTM(input_size=config.n_embed+config.n_char_out,
                           hidden_size=config.n_lstm_hidden,
                           num_layers=config.n_lstm_layers,
                           dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        # the MLP layers
        self.mlp_tag = MLP(n_in=config.n_lstm_hidden*2,
                           n_hidden=config.n_mlp_arc,
                           dropout=config.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)

        self.ffn_tag = nn.Linear(config.n_mlp_arc,
                                 config.n_tags)
        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=config.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=config.n_mlp_rel,
                                 n_out=config.n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.pad_index = config.pad_index
        self.unk_index = config.unk_index

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.embed.weight)

    def forward(self, words, chars):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # get outputs from embedding layers
        embed = self.pretrained(words)
        embed += self.embed(
            words.masked_fill_(words.ge(self.embed.num_embeddings),
                               self.unk_index)
        )
        char_embed = self.char_lstm(chars[mask])
        char_embed = pad_sequence(torch.split(char_embed, lens.tolist()), True)
        embed, char_embed = self.embed_dropout(embed, char_embed)
        # concatenate the word and char representations
        x = torch.cat((embed, char_embed), dim=-1)

        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x = self.lstm(x)
        x_tag, _ = pad_packed_sequence(x[0], True)
        x_tag = self.lstm_dropout(x_tag)[inverse_indices]
        x, _ = pad_packed_sequence(x[-1], True)
        x = self.lstm_dropout(x)[inverse_indices]

        # apply MLPs to the BiLSTM output states
        s_tag = self.mlp_tag(x_tag)
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        s_tag = self.ffn_tag(s_tag)
        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_tag, s_arc, s_rel

    @classmethod
    def load(cls, fname):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state = torch.load(fname, map_location=device)
        network = cls(state['config'], state['embeddings'])
        network.load_state_dict(state['state_dict'])
        network.to(device)

        return network

    @classmethod
    def load_checkpoint(cls, fname):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state = torch.load(fname, map_location=device)

        return state

    def save(self, fname):
        state = {
            'config': self.config,
            'embeddings': self.pretrained.weight,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)

    def save_checkpoint(self, fname, epoch, optimizer, scheduler):
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(state, fname)
