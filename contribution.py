import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


class ResidualAUCMeasure:
    """
    Computes an AUC-style concentration score for conditional dependence residuals.
    For each unique tuple (s,o,a) we compute
        R_{s,o,a} = | P(S,O|A=a) - P(S|A=a) P(O|A=a) |
    then build the cumulative residual-mass curve (sorted by R descending,
    weighted by multiplicity) and return its area under the curve (AUC in [0,1]).

    If `a_col` is None, it reduces to the unconditional residual:
        R_{s,o} = | P(S,O) - P(S) P(O) |.
    """

    def __init__(self, datapath=None, data=None):
        if (datapath is None and data is None) or (datapath is not None and data is not None):
            raise Exception("Usage: pass exactly one of datapath or data")
        self.dataset = pd.read_csv(datapath) if datapath is not None else data.copy()

    def calculate(self, s_col, o_col, a_col=None, k=None, epsilon=None):
        """
        Compute the residual AUC. If `a_col` is provided, uses conditional residuals R_{s,o,a};
        otherwise uses unconditional residuals R_{s,o}.

        Returns
        -------
        float : AUC in [0,1]
        """
        start = time.time()

        cols = [s_col, o_col] + ([a_col] if a_col is not None else [])
        self.dataset = self._encode_and_clean(self.dataset, cols)
        if k is None:
            k = len(self.dataset)

        if a_col is None:
            tuples, values, counts = self._calculate_unconditional_helper(self.dataset[s_col].to_numpy(),
                                                                          self.dataset[o_col].to_numpy())
        else:
            tuples, values, counts = self._calculate_conditional_helper(self.dataset[s_col].to_numpy(),
                                                                        self.dataset[o_col].to_numpy(),
                                                                        self.dataset[a_col].to_numpy())

        top_k = sorted(values, reverse=True)[:min(k, len(values))]
        contribution = float(np.sum(top_k))

        if epsilon is not None:
            min_a = min([self.dataset[a_col].count(a_val) for a_val in set(self.dataset[a_col])])
            sensitivity = (3 * k / (min_a - 1)) + 2
            contribution = contribution + np.random.laplace(loc=0, scale=sensitivity / epsilon)

        elapsed = time.time() - start
        print(f"Residual AUC (approximation): {contribution:.4f} "
              f"({'conditional' if a_col is not None else 'unconditional'}) "
              f"computed in {elapsed:.3f}s on {len(self.dataset)} rows.")
        return contribution

    @staticmethod
    def _encode_and_clean(df, cols):
        df = df.replace(["NA", "N/A", ""], pd.NA).dropna(subset=cols).copy()
        for c in cols:
            df[c] = LabelEncoder().fit_transform(df[c])
        return df

    @staticmethod
    def _calculate_unconditional_helper(S, O):
        """
        Returns (tuples, residual_values, multiplicities) for the unconditional case.
        tuples are (s, o)
        """
        S = np.asarray(S, dtype=np.int64)
        O = np.asarray(O, dtype=np.int64)
        nS, nO = S.max() + 1, O.max() + 1
        N = len(S)

        flat = S * nO + O
        C = np.bincount(flat, minlength=nS * nO).reshape(nS, nO).astype(float)  # counts
        p_so = C / N
        p_s = p_so.sum(axis=1, keepdims=True)     # [nS,1]
        p_o = p_so.sum(axis=0, keepdims=True)     # [1,nO]
        R = np.abs(p_so - p_s @ p_o)              # residual matrix

        # Extract only cells that appear in the data to define tuples + multiplicities
        s_idx, o_idx = np.nonzero(C)
        tuples = list(zip(s_idx.tolist(), o_idx.tolist()))
        values = R[s_idx, o_idx]
        counts = C[s_idx, o_idx]                  # multiplicity for each unique (s,o)
        return tuples, values, counts

    @staticmethod
    def _calculate_conditional_helper(s_col, o_col, a_col):
        """
        Returns (tuples, residual_values, multiplicities) for the conditional case.
        tuples are (s, o, a)
        """
        s_col = np.asarray(s_col, dtype=np.int64)
        o_col = np.asarray(o_col, dtype=np.int64)
        a_col = np.asarray(a_col, dtype=np.int64)
        nS, nO, nA = s_col.max() + 1, o_col.max() + 1, a_col.max() + 1

        flat = a_col * (nS * nO) + s_col * nO + o_col
        C = np.bincount(flat, minlength=nA * nS * nO).reshape(nA, nS, nO).astype(float)  # counts per (a,s,o)

        Na = C.sum(axis=(1, 2))                                # total per a
        mask_a = Na > 0

        Pso = np.zeros_like(C)
        Pso[mask_a] = C[mask_a] / Na[mask_a, None, None]       # P(S,O|a)
        Ps = Pso.sum(axis=2, keepdims=True)                    # P(S|a)
        Po = Pso.sum(axis=1, keepdims=True)                    # P(O|a)
        R = np.abs(Pso - Ps * Po)                              # residuals per (a,s,o)

        # Extract only observed cells to define tuples + multiplicities
        a_idx, s_idx, o_idx = np.nonzero(C)
        tuples = list(zip(s_idx.tolist(), o_idx.tolist(), a_idx.tolist()))
        values = R[a_idx, s_idx, o_idx]
        counts = C[a_idx, s_idx, o_idx]                         # multiplicity for each unique (s,o,a)
        return tuples, values, counts
