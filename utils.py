import random
import torch


def read_corpus(filepath):
    """ Đọc bộ dữ liệu từ đường dẫn tệp đã cho.
        Tham số:
            filepath: đường dẫn tệp của bộ dữ liệu
        Trả về:
            sentences: danh sách các câu, mỗi câu là một danh sách chuỗi
            tags: các nhãn tương ứng
    """
    sentences, tags = [], []
    sent, tag = ['<START>'], ['<START>']
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            if '-DOCSTART-' in line:
                continue
            if line == '\n':
                if len(sent) > 1:
                    sentences.append(sent + ['<END>'])
                    tags.append(tag + ['<END>'])
                sent, tag = ['<START>'], ['<START>']
            else:
                line = line.split()
                sent.append(line[0])
                tag.append(line[3])
    return sentences, tags


def generate_train_dev_dataset(filepath, sent_vocab, tag_vocab, train_proportion=0.8):
    """ Đọc bộ dữ liệu từ đường dẫn tệp đã cho và chia nó thành các phần dành cho huấn luyện và phát triển
        Tham số:
            filepath: đường dẫn tệp
            sent_vocab: từ điển câu
            tag_vocab: từ điển nhãn
            train_proportion: tỉ lệ dữ liệu dành cho huấn luyện
        Trả về:
            train_data: dữ liệu dành cho huấn luyện, danh sách các bộ, mỗi bộ chứa một câu và nhãn tương ứng.
            dev_data: dữ liệu dành cho phát triển, danh sách các bộ, mỗi bộ chứa một câu và nhãn tương ứng.
    """

    sentences, tags = read_corpus(filepath)
    sentences = words2indices(sentences, sent_vocab)
    tags = words2indices(tags, tag_vocab)
    data = list(zip(sentences, tags))
    random.shuffle(data)
    n_train = int(len(data) * train_proportion)
    train_data, dev_data = data[: n_train], data[n_train:]
    return train_data, dev_data


def batch_iter(data, batch_size=32, shuffle=True):
    """ Trả về lô (batch) của (câu, nhãn), theo thứ tự ngược của độ dài nguồn.
        Tham số:
            data: danh sách các bộ, mỗi bộ chứa một câu và nhãn tương ứng.
            batch_size: kích thước lô
            shuffle: giá trị kiểu boolean, quyết định xem có nên xáo trộn dữ liệu một cách ngẫu nhiên hay không
    """
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sentences = [x[0] for x in batch]
        tags = [x[3] for x in batch]
        yield sentences, tags


def words2indices(origin, vocab):
    """ Chuyển đổi một câu hoặc một danh sách các câu từ kiểu chuỗi sang kiểu số nguyên
        Tham số:
            origin: một câu có kiểu list[str], hoặc một danh sách các câu có kiểu list[list[str]]
            vocab: thực thể của Vocab
        Trả về:
            một câu hoặc một danh sách các câu được biểu diễn bằng số nguyên
    """
    if isinstance(origin[0], list):
        result = [[vocab[w] for w in sent] for sent in origin]
    else:
        result = [vocab[w] for w in origin]
    return result


def indices2words(origin, vocab):
    """ Chuyển đổi một câu hoặc một danh sách các câu từ kiểu số nguyên sang chuỗi
        Tham số:
            origin: một câu có kiểu list[int], hoặc một danh sách các câu có kiểu list[list[int]]
            vocab: thực thể của Vocab
        Trả về:
            một câu hoặc một danh sách các câu được biểu diễn bằng chuỗi
    """
    if isinstance(origin[0], list):
        result = [[vocab.id2word(w) for w in sent] for sent in origin]
    else:
        result = [vocab.id2word(w) for w in origin]
    return result


def pad(data, padded_token, device):
    """ đệm dữ liệu để mỗi câu có độ dài giống như câu dài nhất
        Tham số:
            data: danh sách các câu, List[List[từ]]
            padded_token: token được đệm
            device: thiết bị lưu trữ dữ liệu
        Trả về:
            padded_data: dữ liệu sau khi được đệm, một tensor có hình dạng (max_len, b)
            lengths: độ dài của các lô, một danh sách có độ dài b.
    """
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths


def print_var(**kwargs):
    for k, v in kwargs.items():
        print(k, v)


def main():
    sentences, tags = read_corpus('dataset/conll2003/valid.txt')
    print(len(sentences), len(tags))
    print(sentences[0], tags[0])


if __name__ == '__main__':
    main()
