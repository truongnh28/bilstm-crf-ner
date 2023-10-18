import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from preprocess import Vocab
import utils


class Bi_LSTM_CRF(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, dropout_rate=0.5, embedding_size=256, hidden_size=256):
        super(Bi_LSTM_CRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab
        self.dropout_rate = nn.Dropout(dropout_rate)
        self.embedding_size = nn.Embedding(len(sent_vocab), embedding_size)
        self.encoder = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=True)
        self.hidden2emit_score = nn.Linear(hidden_size*2, len(self.tag_vocab))
        self.transition = nn.Parameter(torch.randn(len(self.tag_vocab), len(self.tag_vocab)))
