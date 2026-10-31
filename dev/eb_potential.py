"""Quick upper-bound estimate: learned prior over pair rates (empirical Bayes) vs Beta(1,1)-ish S1.

Uses pair rates from rates.parquet (pairs with >= 80 items). 2-fold split by benchmark:
prior histogram learned on one fold, scored on the other. Expected Brier computed exactly
with a binomial model of the revealed labels (item heterogeneity adds the same constant to both
methods, so it is left out; differences are what matter).

Report: the error part due to not knowing the pair rate (about 0.031 of S1's ALC of about 0.20,
hence "about 85% comes from differences between questions") and the empirical-Bayes check.
Run:  .venv/bin/python dev/eb_potential.py
"""
import math
import random
import sys

import numpy as np

sys.path.insert(0, "dev")
from rates_table import load_rates

rates = load_rates()
rates = rates[(rates.n_items >= 80) & (rates.n_items <= 1000)]
benches = sorted(rates.bench.unique())
random.Random(0).shuffle(benches)
print("pairs", len(rates), "benchmarks", len(benches))

GRID = np.linspace(0.005, 0.995, 100)
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]


def learn_prior(theta, smooth=0.02):
    # histogram of pair rates, smoothed with a small Gaussian kernel, one vote per BENCHMARK-pair equally
    w = np.zeros(len(GRID))
    for t in theta:
        w = w + np.exp(-0.5 * ((GRID - t) / smooth) ** 2)
    return w / w.sum()


def eb_mean(s, n, prior):
    like = GRID ** s * (1 - GRID) ** (n - s)
    post = prior * like
    return float(np.sum(post * GRID) / np.sum(post))


def s1_mean(s, n, p0=0.5, k=2.0):
    return (k * p0 + s) / (k + n)


def expected_sq_error(theta, n, predictor):
    # E over s ~ Binomial(n, theta) of (prediction - theta)^2
    total = 0.0
    for s in range(n + 1):
        prob = math.comb(n, s) * theta ** s * (1 - theta) ** (n - s)
        total += prob * (predictor(s, n) - theta) ** 2
    return total


folds = [benches[0::2], benches[1::2]]
for smooth in [0.02, 0.05]:
    alc_s1 = []
    alc_eb = []
    per_budget = {b: [0.0, 0.0] for b in BUDGETS}
    count = 0
    for f in range(2):
        train = rates[rates.bench.isin(folds[1 - f])]
        test = rates[rates.bench.isin(folds[f])]
        # one value per pair, but cap each benchmark's influence (draws prefer distinct benchmarks)
        prior = learn_prior(train.rate.values, smooth)
        for theta in test.rate.values:
            a = 0.0
            e = 0.0
            for b, w in zip(BUDGETS, WEIGHTS):
                es1 = expected_sq_error(theta, b, s1_mean)
                eeb = expected_sq_error(theta, b, lambda s, n: eb_mean(s, n, prior))
                a += w * es1
                e += w * eeb
                per_budget[b][0] += es1
                per_budget[b][1] += eeb
            alc_s1.append(a)
            alc_eb.append(e)
            count += 1
    d = np.array(alc_eb) - np.array(alc_s1)
    print("\nsmooth", smooth, " pairs scored", count)
    print("  ALC part from pair-rate error:  S1 %.4f   EB %.4f   diff %+.4f (se %.4f)"
          % (np.mean(alc_s1), np.mean(alc_eb), d.mean(), d.std() / math.sqrt(len(d))))
    for b in BUDGETS:
        print("  budget %2d: S1 %.4f  EB %.4f" % (b, per_budget[b][0] / count, per_budget[b][1] / count))
