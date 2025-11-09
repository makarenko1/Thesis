import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from mechanisms.cdp2adp import cdp_rho
from src_for_proxy_mutual_information_mst.mbi import FactoredInference, Dataset, Domain


def MST_conditional(data, epsilon, delta, s_col, o_col, a_col):
    rho = cdp_rho(epsilon, delta)
    sigma = np.sqrt(3 / (2 * rho))

    cliques = [(s_col,), (o_col,), (a_col,)]
    log1 = measure(data, cliques, sigma)
    data, _, _ = compress_domain(data, log1)

    cliques = [(s_col, a_col), (o_col, a_col)]
    log1 = measure(data, cliques, sigma)

    dist = compute_distance_on_joint(data, log1, s_col, o_col, a_col)
    return dist


def measure(data, cliques, sigma, weights=None):
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    for proj, wgt in zip(cliques, weights):
        x = data.project(proj).datavector()
        y = x + np.random.normal(loc=0, scale=sigma / wgt, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append((Q, y, sigma / wgt, proj))
    return measurements


def compress_domain(data, measurements):
    supports = {}
    new_measurements = []
    for Q, y, sigma, proj in measurements:
        for col in proj:
            if col not in supports:
                supports[col] = y >= 3 * sigma
        new_measurements.append((Q, y, sigma, proj))
    undo_compress_fn = lambda data: reverse_data(data, supports)
    return transform_data(data, supports), new_measurements, undo_compress_fn


def compute_distance_on_joint(data, measurement_log, s_col, o_col, a_col):
    engine = FactoredInference(data.domain, iters=1000)
    est = engine.estimate(measurement_log)

    soa_est = est.project([s_col, o_col, a_col]).datavector()
    soa_true = data.project([s_col, o_col, a_col]).datavector()

    dist = np.linalg.norm(soa_true - soa_est, 1)
    return dist


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


def run_mst_conditional(dataset, domain, s_col, o_col, a_col):
    data = Dataset.load(dataset, domain)
    data.df.replace(["NA", "N/A", ""], pd.NA, inplace=True)
    data.df.dropna(inplace=True, subset=[s_col, o_col, a_col], how="any", ignore_index=True, axis=0)
    for attr in [s_col, o_col, a_col]:
        data.df[attr] = LabelEncoder().fit_transform(data.df[attr])

    epsilon = 10000000
    delta = 1e-32

    return MST_conditional(data, epsilon, delta, s_col, o_col, a_col)
