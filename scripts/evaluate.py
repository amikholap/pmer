#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pmer.elo import EloRater
from pmer.trueskill import TrueskillRater
from pmer.evaluation import logloss_for_dataset

Ks = np.linspace(0.001, 0.1, 5)
scales = np.linspace(0.05, 0.2, 5)

import numpy as np
errors = np.zeros((len(Ks), len(scales)))

# raters = []
# for i, K in enumerate(Ks):
    # raters.append([])
    # for j, scale in enumerate(scales):
        # r = EloRater(initial_rating_value=1, K=K, scale=scale)
        # raters[-1].append(r)
# raters = np.array(raters)

betas = [25/12, 25/6, 25/3, 25/2, 25]
taus = [25/3000, 25/1000, 25/500, 25/300, 25/100, 25/50]

raters = []
for i, beta in enumerate(betas):
    raters.append([])
    for j, tau in enumerate(taus):
        r = TrueskillRater(beta=beta, tau=tau)
        raters[-1].append(r)
raters = np.array(raters)


loglosses = logloss_for_dataset(raters.flatten(), 'datasets/dota2.csv')

import ipdb; ipdb.set_trace()
