import json
from collections import Counter
from itertools import chain

from utils import read_corpus


class Vocab:
    def __init__(self, word2id, id2word):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.START = '<START>'
        self.END = '<END>'
        self.__word2id = word2id
        self.__id2word = id2word

    def get_word2id(self):
        return self.__word2id

    def get_id2word(self):
        return self.__id2word

    def __getitem__(self, item):
        if self.UNK in self.__word2id:
            return self.__word2id.get(item, self.__word2id[self.UNK])
        return self.__word2id[item]

    def __len__(self):
        return len(self.__word2id)

    def id2word(self, idx):
        return self.__id2word[idx]

    @staticmethod
    def build(data, max_dict_size, freq_cutoff, is_tags):
        word_counts = Counter(chain(*data))
        valid_words = [w for w, d in word_counts.items() if d >= freq_cutoff]
        valid_words = sorted(valid_words, key=lambda x: word_counts[x], reverse=True)
        valid_words = valid_words[: max_dict_size]
        valid_words += ['<PAD>']
        word2id = {w: idx for idx, w in enumerate(valid_words)}
        if not is_tags:
            word2id['<UNK>'] = len(word2id)
            valid_words += ['<UNK>']
        return Vocab(word2id=word2id, id2word=valid_words)

    def save(self, file_path):
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump({'word2id': self.__word2id, 'id2word': self.__id2word}, f, ensure_ascii=False)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r', encoding='uft8') as f:
            j = json.load(f)
        return Vocab(word2id=j['word2id'], id2word=j['id2word'])


def main():
    sentences, tags = read_corpus('./dataset/conll2003/test.txt')
    sent_vocab = Vocab.build(data=sentences, max_dict_size=9490, freq_cutoff=1, is_tags=False)
    tag_vocab = Vocab.build(data=tags, max_dict_size=9490, freq_cutoff=1, is_tags=True)
    sent_vocab.save('./vocab/sent_vocab.json')
    tag_vocab.save('./vocab/tag_vocab.json')


if __name__ == '__main__':
    main()
