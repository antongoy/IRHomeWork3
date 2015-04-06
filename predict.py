#!/usr/bin/env python
from __future__ import print_function

import sys
import numpy as np

from itertools import izip, tee
from codecs import open

from sklearn.externals import joblib

__author__ = 'anton-goy'


def generate_features(char, pos, text):
    text_len = len(text)
    feature_vector = [ord(char)]

    if pos != text_len - 1:
        feature_vector.append(ord(text[pos + 1]))
    else:
        feature_vector.append(-1)

    if pos != 0:
        feature_vector.append(ord(text[pos - 1]))
    else:
        feature_vector.append(-1)

    if pos < text_len - 2:
        feature_vector.append(ord(text[pos + 2]))
    else:
        feature_vector.append(0)

    for i, c in enumerate(reversed(text[:pos + 1])):
        if c == ' ':
            feature_vector.append(i)
            break
    else:
        if pos == 0:
            feature_vector.append(-1)
        else:
            feature_vector.append(i + 1)

    for i, c in enumerate(text[pos:]):
        if c == ' ' or c == '\n':
            feature_vector.append(i)
            break
    else:
        if pos == text_len - 1:
            feature_vector.append(-1)
        else:
            feature_vector.append(i + 1)

    assert len(feature_vector) == 6

    return feature_vector


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def main():

    input_filename = sys.argv[1]

    classifier = joblib.load('classifier.plk')

    with open(input_filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            samples = []

            for i, char in enumerate(line):
                samples.append(generate_features(char, i, line))

            samples = np.array(samples, dtype=np.int32)

            predict_targets = classifier.predict(samples)

            split_positions = [i for i, predict in enumerate(predict_targets) if predict == 0]

            print(line[:split_positions[0]+1])
            for a, b in pairwise(split_positions):
                print(line[a+1:b+1].strip())

            print()


if __name__ == '__main__':
    main()