import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from z3 import Bool, Or, Not, Optimize, sat


class ProxyRepairMaxSat:
    """
    ProxyRepairMaxSat

    Implementation of the proxy repair metric based on a MaxSAT formulation.
    This version assumes that the protected, response, and admissible sets
    each contain at most one attribute (a single column).

    The method converts each fairness criterion into a set of logical
    clauses (soft and hard) representing multivalued dependency (MVD)
    constraints, then solves a weighted MaxSAT problem to estimate the
    minimal data repair needed for fairness. Laplace noise can be added
    for differential privacy.
    """

    def __init__(self, datapath=None, data=None):
        if (datapath is None and data is None) or (datapath is not None and data is not None):
            raise ValueError("Usage: Should pass either datapath or data itself")
        if datapath is not None:
            self.dataset = pd.read_csv(datapath)
        else:
            self.dataset = data

    def calculate(
        self,
        fairness_criteria,
        epsilon=None,
        encode_and_clean=False,
        chunk_size: int = 100,
    ):
        """
        Computes the proxy repair measure using a MaxSAT solver.

        Each criterion is defined by two or three column names:
          (protected, response) or (protected, response, admissible).

        The method encodes tuples as Boolean variables, builds soft and hard
        3-CNF clauses for each criterion, and solves the combined MaxSAT
        optimization. To improve scalability, the data is processed in
        chunks of `chunk_size` rows per criterion, and the repair values
        are summed across chunks.

        The resulting repair value represents the (approximate) number of
        tuple changes required for fairness. If epsilon is given, Laplace
        noise is added for privacy.

        Parameters
        ----------
        fairness_criteria : list[list[str]]
        epsilon : float or None
        encode_and_clean : bool
        chunk_size : int, default 100
            Number of rows per chunk for the MaxSAT subproblems.

        Returns
        -------
        float
            The estimated repair value (possibly noised).
        """
        start_time = time.time()
        total_repair = 0

        for criterion in fairness_criteria:
            if len(criterion) not in [2, 3]:
                raise ValueError("Invalid input")

            protected_col, response_col, admissible_col = (
                criterion[0],
                criterion[1],
                None if len(criterion) == 2 else criterion[2],
            )
            cols = [protected_col, response_col] + (
                [admissible_col] if admissible_col is not None else []
            )

            # Prepare the base dataframe for this criterion
            if encode_and_clean:
                df_base = self._encode_and_clean(self.dataset, cols)
            else:
                df_base = self.dataset[cols]

            n_rows = len(df_base)
            # Process data in chunks of `chunk_size`
            for start in range(0, n_rows, chunk_size):
                end = start + chunk_size
                df_chunk = df_base.iloc[start:end]

                if df_chunk.empty:
                    continue

                D = self._add_id(df_chunk, cols)

                opt = Optimize()
                D_star = self._conversion_to_solving_general_3cnf(
                    D, admissible_col, opt
                )
                if opt.check() != sat:
                    print(
                        f"No satisfying assignment found for criterion {criterion} "
                        f"in chunk {start}:{end}."
                    )
                    continue
                model = opt.model()

                # repair = number of mismatched tuples for this chunk
                D_set = set(D)
                DR = {t for t in D_star if model.evaluate(Bool(f"x_{t}"))}
                chunk_repair = len(D_set.symmetric_difference(DR))
                total_repair += chunk_repair

        if epsilon is not None:
            sensitivity = 2 * len(fairness_criteria)
            total_repair = total_repair + np.random.laplace(
                loc=0, scale=sensitivity / epsilon
            )

        elapsed_time = time.time() - start_time
        print(
            f"Repair MaxSAT (chunked): Proxy Repair MaxSAT for fairness criteria {fairness_criteria}: "
            f"{total_repair} with data size: {len(self.dataset)} and epsilon: "
            f"{epsilon if epsilon is not None else 'infinity'}. "
            f"Calculation took {elapsed_time:.3f} seconds."
        )
        return total_repair

    @staticmethod
    def _encode_and_clean(df, cols):
        """
        Cleans and label-encode selected columns.

        Drops rows with missing values and converts categorical entries
        into integer codes for the specified columns.

        Returns
        -------
        pandas.DataFrame
            A cleaned and encoded copy of the input data.
        """
        df = df.replace(["NA", "N/A", ""], pd.NA).dropna(subset=cols).copy()
        for c in cols:
            df[c] = LabelEncoder().fit_transform(df[c])
        return df

    @staticmethod
    def _add_id(df, cols):
        """
        Adds a local ID per distinct (S,O[,A]) and returns tuples for
        the MVD S, O, ID | A:

          - with admissible_col: (S, O, A, ID)
          - without admissible_col: (S, O, ID)
        """
        df_local = df[cols]
        ids = df_local.groupby(cols).cumcount().to_numpy() + 1

        s_col, o_col = cols[0], cols[1]
        s_vals = df_local[s_col].to_numpy()
        o_vals = df_local[o_col].to_numpy()

        if len(cols) == 3:
            a_col = cols[2]
            a_vals = df_local[a_col].to_numpy()
            return [
                (s, o, a, i)
                for s, o, a, i in zip(s_vals, o_vals, a_vals, ids)
            ]
        else:
            return [
                (s, o, i)
                for s, o, i in zip(s_vals, o_vals, ids)
            ]

    @staticmethod
    def _clause_key(t1, t2, t3):
        """
        Generates a unique key for a clause defined by three tuples.

        Used internally to avoid duplicate logical clauses.
        """
        return tuple(sorted([f"x_{t1}", f"x_{t2}", f"x_{t3}"]))

    @staticmethod
    def _conversion_to_solving_general_3cnf(D, admissible_col, opt):
        """
        Converts a dataset into a 3-CNF MaxSAT formulation.

        Constructs:
          - D_star: the set of all tuple combinations satisfying MVD structure
          - Soft clauses: one per tuple, encouraging consistency with the data
          - Hard clauses: 3-CNF constraints enforcing fairness dependencies

        Returns
        -------
            D_star
        """
        var_cache = {}

        def v(t):
            if t not in var_cache:
                var_cache[t] = Bool(f"x_{t}")
            return var_cache[t]

        D_set = set(D)

        if admissible_col is not None:
            # D elements: (s, o, a, id)
            by_a = defaultdict(lambda: (set(), set()))
            for s, o, a, i in D:
                S_set, O_set = by_a[a]
                S_set.add(s)
                O_set.add((o, i))  # treat (o, id) as the "O-slot" value

            D_star = set()
            for a, (S_set, O_set) in by_a.items():
                for s in S_set:
                    for (o, i) in O_set:
                        t = (s, o, a, i)  # canonical flat key
                        D_star.add(t)
                        x_t = v(t)
                        opt.add_soft(x_t if t in D_set else Not(x_t), weight=1)

            # hard clauses
            for a, (S_set, O_set) in by_a.items():
                pairs = list(O_set)  # each is (o, id)
                for (o1, i1), (o2, i2) in combinations(pairs, 2):
                    if (o1, i1) == (o2, i2):  # redundant but explicit
                        continue
                    for s1, s2 in combinations(S_set, 2):
                        if s1 == s2:
                            continue
                        t1 = (s1, o1, a, i1)
                        t2 = (s2, o2, a, i2)
                        t3 = (s1, o2, a, i2)
                        opt.add(Or(Not(v(t1)), Not(v(t2)), v(t3)))
        else:
            # D elements: (s, o, id)
            S_set = {s for (s, o, i) in D}
            O_set = {(o, i) for (s, o, i) in D}

            D_star = set()
            for s in S_set:
                for (o, i) in O_set:
                    t = (s, o, i)
                    D_star.add(t)
                    x_t = v(t)
                    opt.add_soft(x_t if t in D_set else Not(x_t), weight=1)

            pairs = list(O_set)
            for (o1, i1), (o2, i2) in combinations(pairs, 2):
                if (o1, i1) == (o2, i2):
                    continue
                for s1, s2 in combinations(S_set, 2):
                    if s1 == s2:
                        continue
                    t1 = (s1, o1, i1)
                    t2 = (s2, o2, i2)
                    t3 = (s1, o2, i2)
                    opt.add(Or(Not(v(t1)), Not(v(t2)), v(t3)))

        return D_star
