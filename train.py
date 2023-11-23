import os.path
import time
import random

import torch
import torch.nn as nn

import bi_lstm_crf
import utils
from bi_lstm_crf import Bi_LSTM_CRF
from preprocess import Vocab
import pandas as pd


def train():
    sent_vocab = Vocab.load('./ignore/vocab/sent_vocab.json')
    tag_vocab = Vocab.load('./ignore/vocab/tag_vocab.json')
    train_data, _ = utils.generate_train_dev_dataset('./ignore/dataset/conll2003/train.txt', sent_vocab, tag_vocab, train_proportion=1.0)
    _, validate_data = utils.generate_train_dev_dataset('./ignore/dataset/conll2003/valid.txt', sent_vocab, tag_vocab,
                                                     train_proportion=0.0)
    print('num of training examples: %d' % (len(train_data)))
    print('num of validate examples: %d' % (len(validate_data)))
    # device = torch.device('cpu')
    max_epoch = 20
    log_every = 10
    validation_every = 250
    model_save_path = 'model.pth'
    optimizer_save_path = './optimizer.pth'
    min_validation_loss = float('inf')
    device = torch.device('cuda')
    patience, decay_num = 0, 0
    learning_rate = 0.001
    batch_size = 32
    clip_max_norm = 5.0
    patience_threshold = 0.98
    max_patience = 5
    max_decay = 5
    lr_decay = 0.5
    recovery = True

    model = Bi_LSTM_CRF(sent_vocab, tag_vocab, 0.5, 300, 300).to(device)

    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, 0, 0.01)
        else:
            nn.init.constant_(param.data, 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_iter = 0
    record_loss_sum, record_target_word_sum, record_batch_size = 0, 0, 0
    # cum = cumulative -> tích lũy
    cum_loss_sum, cum_target_word_sum, cum_batch_size = 0, 0, 0
    record_start, cum_start = time.time(), time.time()

    # loss
    loss_path = 'loss.csv'
    losses = pd.read_csv(loss_path).values.tolist() if recovery and os.path.exists(loss_path) else []

    print('start training...')
    for epoch in range(max_epoch):
        for sentences, tags in utils.batch_iter(train_data, batch_size=batch_size):
            train_iter += 1
            current_batch_size = len(sentences)
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags, _ = utils.pad(tags, tag_vocab[tag_vocab.PAD], device)

            # back propagation
            optimizer.zero_grad()
            batch_loss = model(sentences, tags, sent_lengths)
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            optimizer.step()

            record_loss_sum += batch_loss.sum().item()
            record_batch_size += current_batch_size
            record_target_word_sum += sum(sent_lengths)

            cum_loss_sum += batch_loss.sum().item()
            cum_batch_size += current_batch_size
            cum_target_word_sum += sum(sent_lengths)

            if train_iter % log_every == 10:
                print('log: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, record_target_word_sum / (time.time() - record_start),
                       record_loss_sum / record_batch_size, time.time() - record_start))
                record_loss_sum, record_batch_size, record_target_word_sum = 0, 0, 0
                record_start = time.time()

            if train_iter % validation_every == 0:
                print('dev: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, cum_target_word_sum / (time.time() - cum_start),
                       cum_loss_sum / cum_batch_size, time.time() - cum_start))
                train_loss = cum_loss_sum / cum_batch_size
                cum_loss_sum, cum_batch_size, cum_target_word_sum = 0, 0, 0

                validation_loss = cal_validation_loss(model, validate_data, 64, sent_vocab, tag_vocab, device)
                if validation_loss < min_validation_loss * patience_threshold:
                    min_validation_loss = validation_loss
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), optimizer_save_path)
                    patience = 0
                else:
                    patience += 1
                    if patience == max_patience:
                        decay_num += 1
                        if decay_num == max_decay:
                            print('Early stop. Save result model to %s' % model_save_path)
                            return
                        lr = optimizer.param_groups[0]['lr'] * lr_decay
                        model = bi_lstm_crf.Bi_LSTM_CRF.load(model_save_path, device)
                        optimizer.load_state_dict(torch.load(optimizer_save_path))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        patience = 0
                losses.append([epoch + 1, train_iter, train_loss, validation_loss])
                print('dev: epoch %d, iter %d, validation_loss %f, patience %d, decay_num %d' %
                      (epoch + 1, train_iter, validation_loss, patience, decay_num))
                cum_start = time.time()
                if train_iter % log_every == 0:
                    record_start = time.time()
    __save_loss(losses, loss_path)
    print('Reached %d epochs, Save result model to %s' % (max_epoch, model_save_path))


def cal_validation_loss(model, dev_data, batch_size, sent_vocab, tag_vocab, device):
    is_training = model.training
    model.eval()
    loss, n_sentences = 0, 0
    with torch.no_grad():
        for sentences, tags in utils.batch_iter(dev_data, batch_size, shuffle=False):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags, _ = utils.pad(tags, tag_vocab[sent_vocab.PAD], device)
            batch_loss = model(sentences, tags, sent_lengths)  # shape: (b,)
            loss += batch_loss.sum().item()
            n_sentences += len(sentences)
    model.train(is_training)
    return loss / n_sentences

def __save_loss(losses, file_path):
    pd.DataFrame(data=losses, columns=["epoch", "iter", "train_loss", "validation_loss"]).to_csv(file_path, index=False)


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    train()
