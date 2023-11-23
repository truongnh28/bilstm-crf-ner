import argparse
import json

import numpy as np
import torch

from v2.bi_lstm_crf.app.preprocessing import load_json_file, Preprocessor
from v2.bi_lstm_crf.app.utils import build_model, running_device, arguments_filepath
from v2.data import dict_to_object

args_default = {
        "corpus_dir": "/kaggle/input/connll2003-v2",
        "model_dir": ".",
        "num_epoch": 20,
        "lr": 1e-3,
        "weight_decay": 0.,
        "batch_size": 64,
        "device": "cuda:0",
        "max_seq_len": 300,
        "val_split": 0.2,
        "test_split": 0.1,
        "recovery": False,
        "save_best_val_model": True,
        "embedding_dim": 100,
        "hidden_dim": 128,
        "num_rnn_layers": 2,
        "rnn_type": "lstm"
    }

class WordsTagger:
    def __init__(self, model_dir, device=None):
        args_ = args_default
        args = argparse.Namespace(**args_)
        args.model_dir = model_dir
        self.args = args

        self.preprocessor = Preprocessor(config_dir=model_dir, verbose=False)
        self.model = build_model(self.args, self.preprocessor, load=True, verbose=False)
        self.device = running_device(device)
        self.model.to(self.device)

        self.model.eval()

    def __call__(self, sentences, begin_tags="BS"):
        """predict texts

        :param sentences: a text or a list of text
        :param begin_tags: begin tags for the beginning of a span
        :return:
        """
        if not isinstance(sentences, (list, tuple)):
            raise ValueError("sentences must be a list of sentence")

        try:
            sent_tensor = np.asarray([self.preprocessor.sent_to_vector(s) for s in sentences])
            sent_tensor = torch.from_numpy(sent_tensor).to(self.device)
            with torch.no_grad():
                lo, tags = self.model(sent_tensor)
                print('alo')
                print(tags)
                print()
                print('alo1')
                print(lo)
                print()
            tags = self.preprocessor.decode_tags(tags)
        except RuntimeError as e:
            print("*** runtime error: {}".format(e))
            raise e
        print("cc")
        print(self.tokens_from_tags(sentences, tags, begin_tags=begin_tags))
        return tags, self.tokens_from_tags(sentences, tags, begin_tags=begin_tags)

    @staticmethod
    def tokens_from_tags(sentences, tags_list, begin_tags):
        """extract entities from tags

        :param sentences: a list of sentence
        :param tags_list: a list of tags
        :param begin_tags:
        :return:
        """
        if not tags_list:
            return []

        def _tokens(sentence, ts):
            # begins: [(idx, label), ...]
            all_begin_tags = begin_tags + "O"
            begins = [(idx, t[2:]) for idx, t in enumerate(ts) if t[0] in all_begin_tags]
            begins = [
                         (idx, label)
                         for idx, label in begins
                         if ts[idx] != "O" or (idx > 0 and ts[idx - 1] != "O")
                     ] + [(len(ts), "")]

            tokens_ = [(sentence[s:e], label) for (s, label), (e, _) in zip(begins[:-1], begins[1:]) if label]
            return [((t, tag) if tag else t) for t, tag in tokens_]

        tokens_list = [_tokens(sentence, ts) for sentence, ts in zip(sentences, tags_list)]
        return tokens_list


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("sentence", type=str, help="the sentence to be predicted")
    # parser.add_argument('--model_dir', type=str, required=True, help="the model directory for model files")
    # parser.add_argument('--device', type=str, default=None,
    #                     help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    #
    # args = parser.parse_args()

    model = WordsTagger('./sample_corpus', "cpu")
    results = model(["The United Nations headquarters is located in New York City and was established after the end of the Second World War".split()])
    print(results)
    for objs in results:
        print(json.dumps(objs[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
