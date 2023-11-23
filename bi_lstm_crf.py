import io
import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from preprocess import Vocab


class Bi_LSTM_CRF(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, dropout_rate=0.5, embedding_size=300, hidden_size=300):
        super(Bi_LSTM_CRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab

        # embedding_matrix = self.build_embedding_matrix(sent_vocab, embedding_size)
        # embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        # embedding_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=True)
        # self.embedding = embedding_layer

        embedding_size = 300
        gloveembeddings_index = {}
        with io.open('glove.840B.300d.txt', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray([item for item in values[1:] if '.' not in item], dtype='float32')
                gloveembeddings_index[word] = coefs

        # using vocab and Xtrain, Xvalid, get pretrained glove word embeddings
        glove_embeds = np.zeros((len(sent_vocab), embedding_size))
        for word in sent_vocab.get_words():
            print(word)
            if word in gloveembeddings_index.keys():
                # for the pad word let the embedding be all zeros
                glove_embeds[sent_vocab[word]] = gloveembeddings_index[word]
            else:
                glove_embeds[sent_vocab[word]] = np.random.randn(embedding_size)
        word_embeds = torch.Tensor(glove_embeds)
        # print(glove_embeds.shape) # shape (vocab_length , embedding dim)
        self.embedding = nn.Embedding.from_pretrained(word_embeds, freeze = False)

        # self.embedding = nn.Embedding(len(sent_vocab), embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=True, num_layers=5)
        self.hidden2emit_score = nn.Linear(hidden_size * 2, len(self.tag_vocab))
        self.transition = nn.Parameter(torch.randn(len(self.tag_vocab), len(self.tag_vocab)))

    def forward(self, sentences, tags, sen_lengths):
        """
            Đối số:
                sentences (tensor): các câu, hình dạng (b, len). Độ dài được sắp xếp theo thứ tự giảm dần, len là độ dài
                                    của câu dài nhất
                tags (tensor): các thẻ tương ứng, hình dạng (b, len)
                sen_lengths (danh sách): độ dài của các câu
            Trả về:
                loss (tensor): mất mát trên batch, hình dạng (b,)
        """
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD]).to(self.device)
        sentences = sentences.transpose(0, 1)
        sentences = self.embedding(sentences)
        emit_score = self.encode(sentences, sen_lengths)
        loss = self.cal_loss(tags, mask, emit_score)
        return loss

    def encode(self, sentences, sent_lengths):
        padded_sentences = pack_padded_sequence(sentences, sent_lengths)
        hidden_states, _ = self.encoder(padded_sentences)
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True)
        emit_score = self.hidden2emit_score(hidden_states)
        emit_score = self.dropout(emit_score)
        return emit_score

    def cal_loss(self, tags, mask, emit_score):
        """
        Tính toán mất mát CRF
        Đối số:
        tags (tensor): một batch của tags, kích thước (b, len)
        mask (tensor): mặt nạ cho các tags, kích thước (b, len), giá trị ở vị trí PAD là 0
        emit_score (tensor): ma trận phát ra, kích thước (b, len, K)
        Trả về:
        loss (tensor): mất mát của batch, kích thước (b,)
        """
        batch_size, sent_len = tags.shape
        # tính điểm cho các tags
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)
        # tính toán yếu tố chia tỷ lệ
        d = torch.unsqueeze(emit_score[:, 0], dim=1)
        for i in range(1, sent_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]
            emit_and_transition = emit_score[: n_unfinished, i].unsqueeze(dim=1) + self.transition
            log_sum = d_uf.transpose(1, 2) + emit_and_transition
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)
            log_sum = log_sum - max_v
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)

        d = d.squeeze(dim=1)
        max_d = d.max(dim=-1)[0]
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)
        llk = total_score - d
        loss = -llk
        return loss

    def predict(self, sentences, sen_lengths):
        """
        Đối số:
        sentences (tensor): các câu, kích thước (b, len). Độ dài được sắp xếp theo thứ tự giảm dần, len là độ dài
        của câu dài nhất
        sen_lengths (list): độ dài của các câu
        Trả về:
        tags (list[list[str]]): các tag dự đoán cho batch
        """
        batch_size = sentences.shape[0]
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD])
        sentences = sentences.transpose(0, 1)
        sentences = self.embedding(sentences)
        emit_score = self.encode(sentences, sen_lengths)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size
        d = torch.unsqueeze(emit_score[:, 0], dim=1)
        for i in range(1, sen_lengths[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]
            emit_and_transition = self.transition + emit_score[: n_unfinished, i].unsqueeze(dim=1)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)
        _, max_idx = torch.max(d, dim=1)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

    def save(self, filepath):
        params = {
            'sent_vocab': self.sent_vocab,
            'tag_vocab': self.tag_vocab,
            'args': dict(dropout_rate=self.dropout_rate, embedding_size=self.embedding_size, hidden_size=self.hidden_size),
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = Bi_LSTM_CRF(params['sent_vocab'], params['tag_vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model

    @property
    def device(self):
        return self.embedding.weight.device


def main():
    sent_vocab = Vocab.load('./ignore/vocab/sent_vocab.json')
    tag_vocab = Vocab.load('./ignore/vocab/tag_vocab.json')
    # device = torch.device('cpu')
    model = Bi_LSTM_CRF(sent_vocab, tag_vocab)
    print(model)
    print(model.embedding)
    # model.to(device)
    # model.save('./model.pth')


if __name__ == '__main__':
    main()
