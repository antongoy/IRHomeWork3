#!/usr/bin/env python
from __future__ import print_function

import sys
import numpy as np

from time import time

from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.externals.joblib import dump

__author__ = 'anton-goy'


def arg_parse():
    dataset_filename = sys.argv[1]
    mode = sys.argv[2]

    return dataset_filename, mode


def main():
    input_filename, mode = sys.argv[1]

    time1 = time()

    print('Reading dataset...')
    data_set = np.loadtxt(input_filename, dtype=np.int32, delimiter=',')

    targets = data_set[:, -1]
    data_set = data_set[:, :-1]

    classifier = DecisionTreeClassifier(max_depth=5)

    if mode == 'train':
        print('Model training...')
        classifier.fit(data_set, targets)

        print('Saving model...')
        dump(classifier, 'classifier.plk', compress=3)
    elif mode == 'assess':
        n_folds = 5
        kfold_cross_validation = KFold(data_set.shape[0], n_folds=n_folds)

        precision = 0.0
        accuracy = 0.0
        recall = 0.0

        print('Start the cross validation...')
        for train_index, test_index in kfold_cross_validation:
            classifier.fit(data_set[train_index], targets[train_index])

            predicts = classifier.predict(data_set[test_index])

            accuracy += accuracy_score(targets[test_index], predicts)
            precision += precision_score(targets[test_index], predicts)
            recall += recall_score(targets[test_index], predicts)

        print('Accuracy =', accuracy / n_folds,
              'Precision =', precision / n_folds,
              'Recall =', recall / n_folds, sep='\n')

    else:
        print('Cannot recognize mode...')

    print('Elapsed time =', time() - time1)


if __name__ == '__main__':
    main()