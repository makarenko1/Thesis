""""NOTICE: this file has been modified"""

import itertools

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from mechanisms.cdp2adp import cdp_rho
from src.mbi import FactoredInference, Dataset, Domain

"""
This is a generalization of the winning mechanism from the 
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.  
"""


def MST_unconditional(data, epsilon, delta, s_col, o_col):
    rho = cdp_rho(epsilon, delta)
    sigma = np.sqrt(3/(2*rho))
    cliques = [(s_col,), (o_col,)]
    log1 = measure(data, cliques, sigma)
    data, log1, undo_compress_fn = compress_domain(data, log1)
    weights = compute_weights(data, log1)
    if len(weights) == 1:
        return list(weights.values())[0]
    return weights


def measure(data, cliques, sigma, weights=None):
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    for proj, wgt in zip(cliques, weights):
        x = data.project(proj).datavector()
        y = x + np.random.normal(loc=0, scale=sigma/wgt, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append( (Q, y, sigma/wgt, proj) )
    return measurements


def compress_domain(data, measurements):
    supports = {}
    new_measurements = []
    for Q, y, sigma, proj in measurements:
        col = proj[0]
        sup = y >= 3*sigma
        supports[col] = sup
        if supports[col].sum() == y.size:
            new_measurements.append( (Q, y, sigma, proj) )
        else: # need to re-express measurement over the new domain
            y2 = np.append(y[sup], y[~sup].sum())
            I2 = np.ones(y2.size)
            I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
            y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
            I2 = sparse.diags(I2)
            new_measurements.append( (I2, y2, sigma, proj) )
    undo_compress_fn = lambda data: reverse_data(data, supports)
    return transform_data(data, supports), new_measurements, undo_compress_fn


def compute_weights(data, measurement_log):
    engine = FactoredInference(data.domain, iters=1000)
    est = engine.estimate(measurement_log)

    weights = {}
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    for a, b in candidates:
        xhat = est.project([a, b]).datavector()
        x = data.project([a, b]).datavector()
        weights[a,b] = np.linalg.norm(x - xhat, 1)

    return weights


def transform_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        size = support.sum()
        newdom[col] = int(size)
        if size < support.size:
            newdom[col] += 1
        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size
        df[col] = df[col].map(mapping)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


def reverse_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        mx = support.sum()
        newdom[col] = int(support.size)
        idx, extra = np.where(support)[0], np.where(~support)[0]
        mask = df[col] == mx
        if extra.size == 0:
            pass
        else:
            df.loc[mask, col] = np.random.choice(extra, mask.sum())
        df.loc[~mask, col] = idx[df.loc[~mask, col]]
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


def run_mst_unconditional(dataset, domain, s_col, o_col):
    data = Dataset.load(dataset, domain)
    data.df.replace(["NA", "N/A", ""], pd.NA, inplace=True)
    data.df.dropna(inplace=True, subset=[s_col, o_col], how="any", ignore_index=True, axis=0)
    for attr in [s_col, o_col]:
        data.df[attr] = LabelEncoder().fit_transform(data.df[attr])

    epsilon = 10000000
    delta = 1e-32

    mi_proxy = MST_unconditional(data, epsilon, delta, s_col, o_col)
    return mi_proxy
