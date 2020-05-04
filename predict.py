#!/usr/bin/env python
import pickle
import pandas as pd
import numpy as np


def prob(ix, c, idx, prob_e, prob_p, prob_all):
    if ix is None:
        return 1
    if prob_all[idx][ix] == 0:
        return 1
    if prob_p[idx][ix] == 0 or prob_e[idx][ix] == 0:
        return 0.0001
    if c == 1:
        return prob_p[idx][ix]/prob_all[idx][ix]
    else:
        return prob_e[idx][ix]/prob_all[idx][ix]


def class_prob(x, c, prob_e, prob_p, prob_all):
    res = 1
    for idx, xi in enumerate(x[0]):
        res *= prob(xi, c, idx, prob_e, prob_p, prob_all)
    return res


def is_pois_prob(x, prior_prob, prob_e, prob_p, prob_all):
    normalizer = class_prob(x, 0, prob_e, prob_p, prob_all) * prior_prob[0] \
                 + class_prob(x, 1, prob_e, prob_p, prob_all) * prior_prob[1]
    p_prob = class_prob(x, 1, prob_e, prob_p, prob_all) * prior_prob[1] / normalizer
    e_prob = class_prob(x, 0, prob_e, prob_p, prob_all) * prior_prob[0] / normalizer
    if e_prob > p_prob:
        return False, e_prob
    else:
        return True, p_prob


def predict(X):
    (prob_all, prob_e, prob_p, prior_prob) = pickle.load(open('./models/model.pkl', 'rb'))
    return is_pois_prob(X, prior_prob, prob_e, prob_p, prob_all)