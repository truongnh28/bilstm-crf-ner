import json
def process_and_write_to_file(input_file_path, output_file_path):
    lines = []
    for f in input_file_path:
        with open(f, 'r', encoding='utf-8') as file:
            lines.extend(file.readlines())

    with open(output_file_path, 'w', encoding='utf-8') as file:
        words = []       # a list to store words of the current sentence
        ner_tags = []    # a list to store NER tags of the current sentence

        for line in lines:
            line = line.strip()
            if line:  # If line is not empty
                parts = line.split()
                words.append(parts[0])  # append the word
                ner_tags.append(parts[-1])  # append the NER tag
            else:  # Empty line indicates the end of a sentence
                if words:  # Check if there are words collected for the current sentence
                    # Write the words and NER tags lists to file in the desired format
                    file.write(json.dumps(words) + "\t" + json.dumps(ner_tags) + "\n")
                    words, ner_tags = [], []  # Reset for the next sentence

        # Don't forget to write the last sentence if file doesn't end with a newline
        if words:
            file.write(json.dumps(words) + "\t" + json.dumps(ner_tags) + "\n")


def create_vocab_and_tags_json(input_file_paths, vocab_json_path, tags_json_path):
    vocab = set()  # A set to store vocabulary without duplicates
    tags = set()   # A set to store tags without duplicates

    # Iterate over all file paths provided
    for file_path in input_file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Non-empty line
                    parts = line.split()
                    word = parts[0]
                    tag = parts[-1]
                    vocab.add(word)  # Add word to the vocab set
                    tags.add(tag)    # Add tag to the tags set

    # Convert sets to sorted lists
    vocab_list = sorted(vocab)
    tags_list = sorted(tags - {'O'})  # Remove 'O' to append it in the beginning later
    tags_list.insert(0, 'O')          # Insert 'O' at the beginning of the list

    # Write vocabulary and tags to respective JSON files
    with open(vocab_json_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_list, vocab_file, ensure_ascii=False, indent=2)

    with open(tags_json_path, 'w', encoding='utf-8') as tags_file:
        json.dump(tags_list, tags_file, ensure_ascii=False, indent=2)


def build_dataset_from_conll2003():
    input_file_path = ['../ignore/dataset/conll2003/train.txt', '../ignore/dataset/conll2003/valid.txt']
    output_file_path = 'bi_lstm_crf/app/sample_corpus/dataset.txt'
    process_and_write_to_file(input_file_path, output_file_path)

def build_vocab():
    # Usage
    input_file_path = ['../ignore/dataset/conll2003/train.txt', '../ignore/dataset/conll2003/valid.txt', '../ignore/dataset/conll2003/test.txt']
    vocab_json_path = 'vocab.json'
    tags_json_path = 'tags.json'
    create_vocab_and_tags_json(input_file_path, vocab_json_path, tags_json_path)

from collections import namedtuple
def dict_to_object(dict_obj):
    return namedtuple('ObjectName', dict_obj.keys())(*dict_obj.values())

def te():
    default_config = {
        "corpus_dir": "/kaggle/input/connll2003-v2",
        "model_dir": ".",
        "num_epoch": 20,
        "lr": 1e-3,
        "weight_decay": 0.,
        "batch_size": 1000,
        "device": "cuda:0",
        "max_seq_len": 100,
        "val_split": 0.2,
        "test_split": 0.2,
        "recovery": False,
        "save_best_val_model": False,
        "embedding_dim": 100,
        "hidden_dim": 128,
        "num_rnn_layers": 1,
        "rnn_type": "lstm"
    }
    args = dict_to_object(default_config)


    print(args.model_dir)

if __name__ == '__main__':
    # build_vocab()
    # te()
    build_dataset_from_conll2003()
