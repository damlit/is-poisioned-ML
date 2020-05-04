#!/usr/bin/env python
import pandas
import pickle
import numpy as np
from collections import defaultdict


def is_poisonous(p):
    if p == 'p':
        return 1
    return 0


def change_letters_on_numbers(Y):
    Y_0_1 = []
    for y in Y:
        Y_0_1.append(is_poisonous(y))
    return Y_0_1


def calculate_prior_prob(Y, classes):
    count = [sum(1 if y == c else 0 for y in Y.T.tolist()[0]) for c in classes]
    prior_prob = [float(count[c]) / float(Y.shape[0]) for c in classes]
    return count, prior_prob


def train():
    data = pandas.read_csv('./train_data/mushrooms-train.tsv',
                           names=['tr', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12',
                                  'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22'],
                           sep='\t')

    m, n_plus_1 = data.values.shape
    n = n_plus_1 - 1
    Xn = data.values[:, 1:n_plus_1].reshape(m, n)
    Yn = np.matrix(change_letters_on_numbers(data.values[:, 0])).reshape(m, 1)

    classes = [0, 1]
    count, prior_prob = calculate_prior_prob(Yn, classes)

    prob_e = [defaultdict(int) for i in range(n)]
    prob_p = [defaultdict(int) for i in range(n)]
    prob_all = [defaultdict(int) for i in range(n)]

    for i, x in enumerate(Xn):
        for i_at, at in enumerate(x):
            if Yn[[i]] == 0:
                prob_e[i_at][at] += 1
            else:
                prob_p[i_at][at] += 1
            prob_all[i_at][at] += 1

    model = (prob_all, prob_e, prob_p, prior_prob)
    with open('./models/model.pkl', 'wb') as pickle_file:
        pickle.dump(model, pickle_file)