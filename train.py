#!/usr/bin/env python
from __future__ import print_function

import numpy as np

from time import time
from lxml import etree
from random import randint, sample

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

__author__ = 'anton-goy'


def generate_features_endchar(char, pos, paragraph):
    n_sentences = len(paragraph)

    current_sentence = paragraph[pos]
    next_sentence = None if pos == n_sentences - 1 else paragraph[pos + 1]

    feature_vector = [ord(char)]

    if next_sentence:
        feature_vector.append(ord(' '))
    else:
        feature_vector.append(ord('\n'))

    if len(current_sentence) > 1:
        feature_vector.append(ord(current_sentence[-2]))
    else:
        if pos != 0:
            feature_vector.append(ord(' '))
        else:
            feature_vector.append(-1)

    if next_sentence:
        feature_vector.append(ord(next_sentence[0]))
    else:
        feature_vector.append(ord('\n'))

    for i, c in enumerate(reversed(current_sentence)):
        if c == ' ':
                feature_vector.append(i)
                break
    else:
        if pos == 0:
            feature_vector.append(-1)
        else:
            feature_vector.append(i + 1)

    if char == ' ':
        feature_vector.append(0)
    else:
        if pos == n_sentences - 1:
            feature_vector.append(-1)
        else:
            feature_vector.append(1)

    assert len(feature_vector) == 6

    return feature_vector


def generate_features_commonchar(char, char_pos, sentence_pos, paragraph):

    current_sentence = paragraph[sentence_pos]

    # ord for the current character
    # ord for the next character
    feature_vector = [ord(char), ord(current_sentence[char_pos + 1])]

    # ord for the previous character
    # if the current character is the first in the sentence
    if char_pos == 0:
        if sentence_pos != 0:
            feature_vector.append(ord(' '))
        else:
            feature_vector.append(-1)
    else:
        feature_vector.append(ord(current_sentence[char_pos - 1]))

    # char_pos points to last by one character
    if char_pos == len(current_sentence) - 2:
        if sentence_pos == len(paragraph) - 1:
            feature_vector.append(ord('\n'))
        else:
            feature_vector.append(ord(' '))
    else:
        feature_vector.append(ord(current_sentence[char_pos + 2]))

    # Distance to a previous space character
    for i, c in enumerate(reversed(current_sentence[:char_pos + 1])):
        if c == ' ':
                feature_vector.append(i)
                break
    else:
        if sentence_pos == 0:
            feature_vector.append(-1)
        else:
            feature_vector.append(i + 1)

    # Distance to the next space character
    for i, c in enumerate(current_sentence[char_pos:]):
        if c == ' ':
                feature_vector.append(i)
                break
    else:
        if sentence_pos == len(paragraph) - 1:
            feature_vector.append(-1)
        else:
            feature_vector.append(i + 1)

    assert len(feature_vector) == 6

    return feature_vector


def main():

    time1 = time()
    paragraph_context = etree.iterparse('sentences.xml', events=('start', 'end',), tag=('paragraph', 'source'))

    paragraph_sentences = []
    data_set = []
    targets = []

    for action, element in paragraph_context:
        if action == 'end' and element.tag == 'source':
            paragraph_sentences.append(element.text)
            continue

        if action == 'start' and element.tag == 'paragraph':
            paragraph_sentences = []
            continue

        if action == 'end' and element.tag == 'paragraph':
            for i, sentence in enumerate(paragraph_sentences):
                sentence_len = len(sentence)

                if sentence_len == 1:
                    continue

                pos = randint(0, sentence_len - 2)

                data_set.append(generate_features_commonchar(sentence[pos], pos, i, paragraph_sentences))
                targets.append(1)

                end_char = ' ' if sentence[-1].isalpha() else sentence[-1]

                data_set.append(generate_features_endchar(end_char, i, paragraph_sentences))
                targets.append(0)

            continue

    data_set = np.array(data_set, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    classifier = DecisionTreeClassifier(max_depth=5)
    classifier.fit(data_set, targets)

    joblib.dump(classifier, 'classifier.plk', compress=3)

    print(time() - time1)

if __name__ == '__main__':
    main()
