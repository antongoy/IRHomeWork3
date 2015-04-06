#!/usr/bin/env python
from __future__ import print_function

import sys
import numpy as np

from time import time
from lxml import etree
from random import sample, choice

__author__ = 'anton-goy'


def arg_parse():
    return sys.argv[1]


def generate_features_endchar(char, pos, paragraph):
    n_sentences = len(paragraph)

    current_sentence = paragraph[pos]
    next_sentence = None if pos == n_sentences - 1 else paragraph[pos + 1]

    feature_vector = [ord(char), 0]

    if len(current_sentence) > 1:
        feature_vector.append(int(current_sentence[-2].isupper()))
    else:
        feature_vector.append(0)

    if next_sentence:
        feature_vector.append(int(next_sentence[0].isupper()))
    else:
        feature_vector.append(0)

    for i, c in enumerate(reversed(current_sentence)):
        if c == ' ':
                feature_vector.append(i)
                break
    else:
        if pos == 0:
            feature_vector.append(-1)
        else:
            feature_vector.append(i + 1)

    if pos == n_sentences - 1:
        feature_vector.append(-1)
    else:
        feature_vector.append(1)

    feature_vector.append(1)

    return feature_vector


def generate_features_commonchar(char, char_pos, sentence_pos, paragraph):

    current_sentence = paragraph[sentence_pos]

    feature_vector = [ord(char), int(current_sentence[char_pos + 1].isupper())]

    if char_pos == 0:
        feature_vector.append(0)
    else:
        feature_vector.append(int(current_sentence[char_pos - 1].isupper()))

    if char_pos == len(current_sentence) - 2:
        feature_vector.append(0)
    else:
        feature_vector.append(int(current_sentence[char_pos + 2].isupper()))

    for i, c in enumerate(reversed(current_sentence[:char_pos + 1])):
        if c == ' ':
                feature_vector.append(i)
                break
    else:
        if sentence_pos == 0:
            feature_vector.append(-1)
        else:
            feature_vector.append(i + 1)

    for i, c in enumerate(current_sentence[char_pos:]):
        if c == ' ':
                feature_vector.append(i)
                break
    else:
        if sentence_pos == len(paragraph) - 1:
            feature_vector.append(-1)
        else:
            feature_vector.append(i + 1)

    feature_vector.append(0)

    return feature_vector


def main():

    output_filename = arg_parse()

    time1 = time()
    paragraph_context = etree.iterparse('sentences.xml', events=('start', 'end',), tag=('paragraph', 'source'))

    paragraph_sentences = []
    data_set = []

    print('Reading sentences.xml...')
    for action, element in paragraph_context:
        if action == 'end' and element.tag == 'source':
            paragraph_sentences.append(element.text.strip())
            continue

        if action == 'start' and element.tag == 'paragraph':
            paragraph_sentences = []
            continue

        if action == 'end' and element.tag == 'paragraph':
            for i, sentence in enumerate(paragraph_sentences):
                sentence_len = len(sentence)

                if sentence_len == 1:
                    continue

                need_characters = [k for k, char in enumerate(sentence) if char in '.!?' and k != sentence_len - 1]
                if not need_characters:
                    continue

                positions = sample(need_characters, 1 if len(need_characters) < 3 else 2 if len(need_characters) < 2 else 1)

                for pos in positions:
                    data_set.append(generate_features_commonchar(sentence[pos], pos, i, paragraph_sentences))

                end_char = choice('.!?') if sentence[-1] not in '.!?' else sentence[-1]

                paragraph_sentences[i] = sentence[:-1] + end_char

                data_set.append(generate_features_endchar(end_char, i, paragraph_sentences))

            continue

    print('Converting data set...')
    data_set = np.array(data_set, dtype=np.int32)

    print('Saving data set...')
    np.savetxt(output_filename, data_set, fmt='%d', delimiter=',', comments='')

    print('Elapsed time =', time() - time1)

if __name__ == '__main__':
    main()
