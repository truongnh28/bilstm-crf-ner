import random
import torch
import os.path
from gensim.models import KeyedVectors
import numpy as np


def read_corpus(filepaths):
    """ Đọc bộ dữ liệu từ đường dẫn tệp đã cho.
        Tham số:
            filepath: đường dẫn tệp của bộ dữ liệu
        Trả về:
            sentences: danh sách các câu, mỗi câu là một danh sách chuỗi
            tags: các nhãn tương ứng
    """
    sentences, tags = [], []
    for filepath in filepaths:
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
    """ Đọc bộ dữ liệu từ đường dẫn tệp đã cho và chia nó thành các phần dành cho train và dev
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
        tags = [x[1] for x in batch]
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


def build_embedding_matrix(sent_vocab, embedding_size=300):
    embedding_matrix_path = './ignore/embedding_matrix/embedding_matrix.npy'
    if os.path.exists(embedding_matrix_path):
        embedding_matrix = np.load(embedding_matrix_path)
        return embedding_matrix
    word2vec_pretrain_path = './pre-trained/glove.42B.300d_word2vec.txt'
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_pretrain_path, binary=False)

    embedding_matrix = np.random.randn(len(sent_vocab), embedding_size) * 0.01
    word2id = sent_vocab.get_word2id().items()
    for word, idx in word2id:
        if word in word2vec_model:
            embedding_matrix[idx] = word2vec_model[word]
    np.save('ignore/embedding_matrix/embedding_matrix.npy', embedding_matrix)
    return embedding_matrix

def print_var(**kwargs):
    for k, v in kwargs.items():
        print(k, v)


# Hàm chuyển đổi ner results sang cặp (text, entity)
def ner_to_tuples(ner_results, original_text):
    # Sắp xếp ner_results theo chỉ số start
    sorted_entities = sorted(ner_results, key=lambda x: x['start'])

    # Lưu kết quả cuối cùng
    final_results = []
    last_end = 0

    # Duyệt qua từng entity và tạo cặp
    for entity in sorted_entities:
        # Thêm văn bản không phải entity
        if entity['start'] > last_end:
            final_results.append((original_text[last_end:entity['start']], None))

        # Chuẩn bị word để gộp ## tokens nếu có
        word = entity['word'].replace('##', '')

        # Kiểm tra xem có phải tiếp tục của entity cũ không
        if final_results and final_results[-1][1] == entity['entity'][2:]:
            # Gộp với tuple cuối cùng nếu cùng entity
            prev_text, prev_label = final_results.pop()
            word = f'{prev_text} {word}'

        # Tạo một cặp mới cho từ entity
        final_results.append((word, entity['entity'][2:]))

        # Cập nhật vị trí kết thúc cuối cùng
        last_end = entity['end']

    # Thêm phần cuối cùng của văn bản nếu còn
    if last_end < len(original_text):
        final_results.append((original_text[last_end:], None))

    # Gộp các từ cùng entity liên tiếp
    merged_results = []
    for text, entity in final_results:
        if merged_results and merged_results[-1][1] == entity:
            merged_text, _ = merged_results.pop()
            text = f'{merged_text} {text}'
        merged_results.append((text, entity))

    return merged_results



def convert_ner_results_to_tuples(ner_results):
    converted_results = []
    current_entity = None
    current_word = ""

    for word, tag in ner_results:
        if tag == 'O':  # Outside any named entity
            if current_entity:  # Finish the current entity if there is one
                converted_results.append((current_word.strip(), current_entity))
                current_word, current_entity = "", None
            converted_results.append((word, None))
        elif tag.startswith('B-'):  # Beginning of a named entity
            if current_entity:  # Finish the previous entity if there is one
                converted_results.append((current_word.strip(), current_entity))
            current_entity = tag.split('-')[1]  # Set the new entity type
            current_word = word  # Start the word for the new entity
        elif tag.startswith('I-') and current_entity:  # Inside a named entity
            current_word += " " + word  # Add word to the current entity
        elif tag.startswith('I-') and not current_entity:  # Entity without a beginning
            # Handle case where I- tag is without B- tag, treat it as beginning
            current_entity = tag.split('-')[1]
            current_word = word
        elif word == '<UNK>':  # Special handling for unknown tokens
            # Decide how to handle <UNK> based on context
            # Here we just ignore it, but you might need different logic
            pass

    # Add last entity if there was one when the loop ended
    if current_entity:
        converted_results.append((current_word.strip(), current_entity))

    # Add non-entity text if it's trailing after last entity
    if current_word and not current_entity:
        converted_results.append((current_word, None))

    converted_ner_results = converted_results
    filtered_ner_results = [item for item in converted_ner_results if item[0] not in ('<UNK>', '<END>')]

    return filtered_ner_results


def process_ner_results(ner_results):
    # Khởi tạo các biến
    processed_results = []
    current_phrase = ""
    current_type = None
    buffer_text = ""  # Dùng để lưu văn bản không phải là entity giữa các entities

    # Duyệt qua tất cả các kết quả
    for word, entity in ner_results:
        if entity == 'O':  # Nếu không phải là entity
            # Nếu có một entity đang mở, thêm nó vào trước khi tiếp tục với văn bản
            if current_type:
                processed_results.append((current_phrase, current_type))
                current_phrase, current_type = "", None  # Đặt lại sau khi thêm entity
            # Xử lý văn bản không phải là entity, thêm vào buffer
            if buffer_text:  # Nếu buffer không rỗng, thêm một khoảng trắng trước từ
                buffer_text += " "
            buffer_text += word
        else:
            # Xử lý trường hợp buffer text trước entity
            if buffer_text:
                processed_results.append((buffer_text, None))  # Thêm buffer text vào kết quả
                buffer_text = ""  # Làm trống buffer sau khi thêm

            tag_type = entity.split('-')[1] if '-' in entity else entity
            # Bắt đầu một entity mới hoặc tiếp tục nối chuỗi của entity hiện tại
            if entity.startswith('B-') or current_type is None or current_type != tag_type:
                # Thêm current entity vào nếu đã có
                if current_phrase:
                    processed_results.append((current_phrase, current_type))
                current_phrase, current_type = word, tag_type
            elif entity.startswith('I-') and current_type == tag_type:
                current_phrase += " " + word  # Nối thêm từ vào entity hiện tại

    # Thêm entity cuối cùng vào nếu có
    if current_phrase:
        processed_results.append((current_phrase, current_type))

    # Nếu còn buffer text sau entity cuối cùng, thêm nó vào
    if buffer_text:
        processed_results.append((buffer_text, None))

    return processed_results

def main():
    sentences, tags = read_corpus('dataset/conll2003/valid.txt')
    print(len(sentences), len(tags))
    print(sentences[0], tags[0])


if __name__ == '__main__':
    main()
