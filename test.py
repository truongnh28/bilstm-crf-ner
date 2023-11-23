import bi_lstm_crf
import utils
from preprocess import Vocab
import torch
import torch.nn as nn

def ev():
    model_path = './model/model.pth'
    result_path = './result.txt'
    test_path = './ignore/dataset/conll2003/test.txt'
    sent_path = './ignore/vocab/sent_vocab.json'
    tag_path = './ignore/vocab/tag_vocab.json'
    batch_size = 32

    sent_vocab = Vocab.load(sent_path)
    tag_vocab = Vocab.load(tag_path)
    sentences, tags = utils.read_corpus([test_path])
    sentences = utils.words2indices(sentences, sent_vocab)
    tags = utils.words2indices(tags, tag_vocab)
    test_data = list(zip(sentences, tags))
    print('num of test samples: %d' % (len(test_data)))

    device = torch.device('cpu')
    model = bi_lstm_crf.Bi_LSTM_CRF.load(model_path, device)
    print('start testing...')
    print('using device', device)

    result_file = open(result_path, 'w')
    model.eval()
    with torch.no_grad():
        for sentences, tags in utils.batch_iter(test_data, batch_size=batch_size, shuffle=False):
            padded_sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            predicted_tags = model.predict(padded_sentences, sent_lengths)
            for sent, true_tags, pred_tags in zip(sentences, tags, predicted_tags):
                sent, true_tags, pred_tags = sent[1: -1], true_tags[1: -1], pred_tags[1: -1]
                for token, true_tag, pred_tag in zip(sent, true_tags, pred_tags):
                    result_file.write(' '.join([sent_vocab.id2word(token), tag_vocab.id2word(true_tag),
                                                tag_vocab.id2word(pred_tag)]) + '\n')
                result_file.write('\n')

def predict():
    return p

def p(sentence):
    model_path = './model/model.pth'
    sent_path = './ignore/vocab/sent_vocab.json'
    tag_path = './ignore/vocab/tag_vocab.json'

    # Load vocab
    sent_vocab = Vocab.load(sent_path)
    tag_vocab = Vocab.load(tag_path)

    # Convert sentence to indices
    sentences = utils.words2indices([sentence.split()], sent_vocab)

    # Load model
    device = torch.device('cpu')
    model = bi_lstm_crf.Bi_LSTM_CRF.load(model_path, device)
    model.eval()

    # Predict
    with torch.no_grad():
        padded_sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
        predicted_tags = model.predict(padded_sentences, sent_lengths)
        for sent, pred_tags in zip(sentences, predicted_tags):
            return [(sent_vocab.id2word(token), tag_vocab.id2word(tag)) for token, tag in zip(sent, pred_tags)]



# Use the function to convert ner results

if __name__ == '__main__':
    ev()
    c = predict()
    a = c("The United Nations headquarters is located in New York City and was established after the end of the Second World War.")
    print(a)
    # converted_ner_results = utils.convert_ner_results_to_tuples(a)
    # print(converted_ner_results)


