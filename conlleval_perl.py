#!/usr/bin/python
#### Original Perl Script
# conlleval: evaluate result of processing CoNLL-2000 shared task
# usage:     conlleval [-l] [-r] [-d delimiterTag] [-o oTag] < file
#            README: http://cnts.uia.ac.be/conll2000/chunking/output.html
# options:   l: generate LaTeX output for tables like in
#               http://cnts.uia.ac.be/conll2003/ner/example.tex
#            r: accept raw result tags (without B- and I- prefix;
#                                       assumes one word per chunk)
#            d: alternative delimiter tag (default is white space or tab)
#            o: alternative outside tag (default is O)
# note:      the file should contain lines with items separated
#            by $delimiter characters (default space). The final
#            two items should contain the correct tag and the
#            guessed tag in that order. Sentences should be
#            separated from each other by empty lines or lines
#            with $boundary fields (default -X-).
# url:       http://lcg-www.uia.ac.be/conll2000/chunking/
# started:   1998-09-25
# version:   2004-01-26
# author:    Erik Tjong Kim Sang <erikt@uia.ua.ac.be>
#### Now in Python
# author:    sighsmile.github.io
# version:   2017-05-18

from __future__ import division, print_function, unicode_literals
import argparse
import sys
from collections import defaultdict

# sanity check
def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-l", "--latex",
        default=False, action="store_true",
        help="generate LaTeX output"
    )
    argparser.add_argument(
        "-r", "--raw",
        default=False, action="store_true",
        help="accept raw result tags"
    )
    argparser.add_argument(
        "-d", "--delimiter",
        default=None,
        help="alternative delimiter tag (default: single space)"
    )
    argparser.add_argument(
        "-o", "--oTag",
        default="O",
        help="alternative delimiter tag (default: O)"
    )
    args = argparser.parse_args()
    return args



"""
• IOB1: I is a token inside a chunk, O is a token outside a chunk and B is the
beginning of chunk immediately following another chunk of the same Named Entity.
• IOB2: It is same as IOB1, except that a B tag is given for every token, which exists at
the beginning of the chunk.
• IOE1: An E tag used to mark the last token of a chunk immediately preceding another
chunk of the same named entity.
• IOE2: It is same as IOE1, except that an E tag is given for every token, which exists at
the end of the chunk.
• START/END: This consists of the tags B, E, I, S or O where S is used to represent a
chunk containing a single token. Chunks of length greater than or equal to two always
start with the B tag and end with the E tag.
• IO: Here, only the I and O labels are used. This therefore cannot distinguish between
adjacent chunks of the same named entity.

"""
# endOfChunk: checks if a chunk ended between the previous and current word
# arguments:  previous and current chunk tags, previous and current types
# note:       this code is capable of handling other chunk representations
#             than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
#             Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
def end_of_chunk(prev_tag, tag, prev_type, type):
    """
    checks if a chunk ended between the previous and current word;
    arguments:  previous and current chunk tags, previous and current types
    """
    return ((prev_tag == "B" and tag == "B") or
            (prev_tag == "B" and tag == "O") or
            (prev_tag == "I" and tag == "B") or
            (prev_tag == "I" and tag == "O") or

            (prev_tag == "E" and tag == "E") or
            (prev_tag == "E" and tag == "I") or
            (prev_tag == "E" and tag == "O") or
            (prev_tag == "I" and tag == "O") or

            (prev_tag != "O" and prev_tag != "." and prev_type != type) or
            (prev_tag == "]" or prev_tag == "["))
        # corrected 1998-12-22: these chunks are assumed to have length 1


# startOfChunk: checks if a chunk started between the previous and current word
# arguments:    previous and current chunk tags, previous and current types
# note:         this code is capable of handling other chunk representations
#               than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
#               Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
def start_of_chunk(prev_tag, tag, prev_type, type):
    """
    checks if a chunk started between the previous and current word;
    arguments:  previous and current chunk tags, previous and current types
    """
    chunk_start = ((prev_tag == "B" and tag == "B") or
                  (prev_tag == "B" and tag == "B") or
                  (prev_tag == "I" and tag == "B") or
                  (prev_tag == "O" and tag == "B") or
                  (prev_tag == "O" and tag == "I") or

                  (prev_tag == "E" and tag == "E") or
                  (prev_tag == "E" and tag == "I") or
                  (prev_tag == "O" and tag == "E") or
                  (prev_tag == "O" and tag == "I") or

                  (tag != "O" and tag != "." and prev_type != type) or
                  (tag == "]" or tag == "["))
        # corrected 1998-12-22: these chunks are assumed to have length 1

    #print("startOfChunk?", prevTag, tag, prevType, type)
    #print(chunkStart)
    return chunk_start

def calc_metrics(TP, P, T, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = TP / P if P else 0
    recall = TP / T if T else 0
    FB1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * FB1
    else:
        return precision, recall, FB1

def split_tag(chunk_tag, o_tag ="O", raw = False):
    """
    Split chunk tag into IOB tag and chunk type;
    return (iob_tag, chunk_type)
    """
    if chunk_tag == "O" or chunk_tag == o_tag:
        tag, type = "O", None
    elif raw:
        tag, type = "B", chunk_tag
    else:
        try:
            # split on first hyphen, allowing hyphen in type
            tag, type = chunk_tag.split('-', 1)
        except ValueError:
            tag, type  = chunk_tag, None
    return tag, type

def count_chunks(path, raw = False, o_tag = 'O'):
    """
    Process input in given format and count chunks using the last two columns;
    return correctChunk, foundGuessed, foundCorrect, correct_tags, tokenCounter
    """
    boundary = "-X-"     # sentence boundary
    delimiter = None
    raw = False
    o_tag = 'O'

    correct_chunk = defaultdict(int)     # number of correctly identified chunks
    found_correct = defaultdict(int)     # number of chunks in corpus per type
    found_guessed = defaultdict(int)     # number of identified chunks per type

    token_counter = 0     # token counter (ignores sentence breaks)
    correct_tags = 0      # number of correct chunk tags

    last_type = None # temporary storage for detecting duplicates
    in_correct = False # currently processed chunk is correct until now
    last_correct, last_correct_type = "O", None    # previous chunk tag in corpus
    last_guessed, last_guessed_type = "O", None  # previously identified chunk tag
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        # each non-empty line must contain >= 3 columns
        features = line.strip().split(delimiter)
        if not features or features[0] == boundary:
            features = [boundary, "O", "O"]
        elif len(features) < 3:
             raise IOError("conlleval: unexpected number of features in line %s\n" % line)

        # extract tags from last 2 columns
        guessed, guessed_type = split_tag(features[-1], o_tag=o_tag, raw=raw)
        correct, correct_type = split_tag(features[-2], o_tag=o_tag, raw=raw)

        # 1999-06-26 sentence breaks should always be counted as out of chunk
        first_item = features[0]
        if first_item == boundary:
            guessed, guessed_type = "O", None

        # decide whether current chunk is correct until now
        if in_correct:
            end_of_guessed = end_of_chunk(last_correct, correct, last_correct_type, correct_type)
            end_of_correct = end_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)
            if end_of_guessed and end_of_correct and last_guessed_type == last_correct_type:
                in_correct = False
                correct_chunk[last_correct_type] += 1
            elif end_of_guessed != end_of_correct or guessed_type != correct_type:
                in_correct = False

        start_of_guessed = start_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)
        start_of_correct = start_of_chunk(last_correct, correct, last_correct_type, correct_type)
        if start_of_correct and start_of_guessed and guessed_type == correct_type:
            in_correct = True
        if start_of_correct:
            found_correct[correct_type] += 1
        if start_of_guessed:
            found_guessed[guessed_type] += 1

        if first_item != boundary:
            if correct == guessed and guessed_type == correct_type:
                correct_tags += 1
            token_counter += 1

        last_guessed, last_guessed_type = guessed, guessed_type
        last_correct, last_correct_type = correct, correct_type

    if in_correct:
        correct_chunk[last_correct_type] += 1

    return correct_chunk, found_guessed, found_correct, correct_tags, token_counter

def evaluate(correct_chunk, found_guessed, found_correct, correct_tags, token_counter):
    # sum counts
    correct_chunk_sum = sum(correct_chunk.values())
    found_guessed_sum = sum(found_guessed.values())
    found_correct_sum = sum(found_correct.values())

    # sort chunk type names
    sorted_types = list(found_correct) + list(found_guessed)
    sorted_types = list(set(sorted_types))
    sorted_types.sort()
    ans = {}
    # print overall performance, and performance per chunk type
    # compute overall precision, recall and FB1 (default values are 0.0)
    precision, recall, FB1 = calc_metrics(correct_chunk_sum, found_guessed_sum, found_correct_sum)
    # print overall performance
    print("processed %i tokens with %i phrases; " % (token_counter, found_correct_sum), end='')
    print("found: %i phrases; correct: %i.\n" % (found_guessed_sum, correct_chunk_sum), end='')
    if token_counter:
        print("accuracy: %6.2f%%; " % (100 * correct_tags / token_counter), end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
              (precision, recall, FB1))
        ans['Accuracy'] = [precision, recall, FB1]

    for i in sorted_types:
        precision, recall, FB1 = calc_metrics(correct_chunk[i], found_guessed[i], found_correct[i])
        print("%17s: " % i, end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
              (precision, recall, FB1), end='')
        print("  %d" % found_guessed[i])
        ans[i] = [precision, recall, FB1]

    return ans


def call_loss_f1(path):
    correct_chunk, found_guessed, found_correct, correct_tags, token_counter = count_chunks(path)
    # compute metrics and print
    return evaluate(correct_chunk, found_guessed, found_correct, correct_tags, token_counter)

