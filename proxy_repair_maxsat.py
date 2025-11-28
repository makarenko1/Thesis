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

    def calculate(self, fairness_criteria, epsilon=None, encode_and_clean=False):
        """
        Computes the proxy repair measure using a MaxSAT solver.

        Each criterion is defined by two or three column names:
          (protected, response) or (protected, response, admissible).

        The method encodes tuples as Boolean variables, builds soft and hard
        3-CNF clauses for each criterion, and solves the combined MaxSAT
        optimization. The resulting repair value represents the number of
        tuple changes required for fairness. If epsilon is given, Laplace
        noise is added for privacy.

        Returns
        -------
        float
            The estimated repair value (possibly noised).
        """
        start_time = time.time()
        repair = 0

        for criterion in fairness_criteria:
            if len(criterion) not in [2, 3]:
                raise ValueError("Invalid input")

            protected_col, response_col, admissible_col = (criterion[0], criterion[1],
                                                           None if len(criterion) == 2 else criterion[2])
            cols = [protected_col, response_col] + ([admissible_col] if admissible_col is not None else [])
            if encode_and_clean:
                df = self._encode_and_clean(self.dataset, cols)
            else:
                df = self.dataset
            D = self._add_id(df, cols)

            opt = Optimize()
            D_star = (
                self._conversion_to_solving_general_3cnf(D, admissible_col, opt))
            if opt.check() != sat:
                print("No satisfying assignment found.")
                continue
            model = opt.model()

            # Compute DR = satisfying assignment
            DR = set()
            for t in D_star:
                if model.evaluate(Bool(f"x_{t}")):
                    DR.add(t)

            # repair = number of mismatched tuples
            list_symmetric_difference = [row for row in D if row not in DR] + [row for row in DR if row not in D]
            repair += len(list_symmetric_difference)

        if epsilon is not None:
            sensitivity = 2 * len(fairness_criteria)
            repair = repair + np.random.laplace(loc=0, scale=sensitivity / epsilon)

        elapsed_time = time.time() - start_time
        print(
            f"Repair MaxSAT: Proxy Repair MaxSAT for fairness criteria {fairness_criteria}: "
            f"{repair:.4f} with data size: {len(self.dataset)} and epsilon: "
            f"{epsilon if epsilon is not None else 'infinity'}. Calculation took {elapsed_time:.3f} seconds."
        )
        return repair

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
        the MVD S, (O,ID) | A:

          - with admissible_col: (S, (O,ID), A)
          - without admissible_col: (S, (O,ID))

        Parameters
        ----------
        df : pandas.DataFrame
        cols : list[str]
            [protected_col, response_col] (+ [admissible_col] if present)

        Returns
        -------
        list[tuple]
            List of tuples suitable for _conversion_to_solving_general_3cnf.
        """
        df_local = df[cols].copy()
        df_local["ID"] = df_local.groupby(cols).cumcount() + 1

        s_col, o_col = cols[0], cols[1]
        ids = df_local["ID"].to_numpy()
        s_vals = df_local[s_col].to_numpy()
        o_vals = df_local[o_col].to_numpy()

        if len(cols) == 3:
            a_col = cols[2]
            a_vals = df_local[a_col].to_numpy()
            # Build list of tuples without iterrows
            D = [
                (s_vals[i], (o_vals[i], ids[i]), a_vals[i])
                for i in range(len(df_local))
            ]
        else:
            D = [
                (s_vals[i], (o_vals[i], ids[i]))
                for i in range(len(df_local))
            ]

        return D

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
            by_a = defaultdict(lambda: (set(), set()))
            for s, o, a in D:
                S_set, O_set = by_a[a]
                S_set.add(s)
                O_set.add(o)

            D_star = set()
            for a, (S_set, O_set) in by_a.items():
                for s in S_set:
                    for o in O_set:
                        t = (s, o, a)
                        D_star.add(t)
                        x_t = v(t)
                        opt.add_soft(x_t if t in D_set else Not(x_t), weight=1)

            for a, (S_set, O_set) in by_a.items():
                pairs = [(s, o) for s in S_set for o in O_set]
                for (s1, o1), (s2, o2) in combinations(pairs, 2):
                    if s1 == s2 or o1 == o2:
                        continue
                    t1 = (s1, o1, a)
                    t2 = (s2, o2, a)
                    t3 = (s1, o2, a)
                    opt.add(Or(Not(v(t1)), Not(v(t2)), v(t3)))
        else:
            S_set = {s for (s, _) in D}
            O_set = {o for (_, o) in D}

            D_star = set()
            for s in S_set:
                for o in O_set:
                    t = (s, o)
                    D_star.add(t)
                    x_t = v(t)
                    opt.add_soft(x_t if t in D_set else Not(x_t), weight=1)

            pairs = [(s, o) for s in S_set for o in O_set]
            for (s1, o1), (s2, o2) in combinations(pairs, 2):
                if s1 == s2 or o1 == o2:
                    continue
                t1 = (s1, o1)
                t2 = (s2, o2)
                t3 = (s1, o2)
                opt.add(Or(Not(v(t1)), Not(v(t2)), v(t3)))

        return D_star

        # Step 3: Enforce MVD 3CNF constraints
        # OLD MEMORY INEFFICIENT:
        # C = set()
        # if admissible_col is not None:
        #     for (s1, o1, a1) in D_star:
        #         for (s2, o2, a2) in D_star:
        #             if a1 == a2 and s1 != s2 and o1 != o2:
        #                 C.add((s1, o1, s2, o2, a1))
        # else:
        #     for (s1, o1) in D_star:
        #         for (s2, o2) in D_star:
        #             if s1 != s2 and o1 != o2:
        #                 C.add((s1, o1, s2, o2))
        #
        # used_keys = set()
        # for t in C:
        #     if admissible_col is not None:
        #         s1, o1, s2, o2, a = t
        #         t1 = (s1, o1, a)
        #         t2 = (s2, o2, a)
        #         t3 = (s1, o2, a)
        #     else:
        #         s1, o1, s2, o2 = t
        #         t1 = (s1, o1)
        #         t2 = (s2, o2)
        #         t3 = (s1, o2)
        #     x_t1 = Bool(f"x_{t1}")
        #     x_t2 = Bool(f"x_{t2}")
        #     x_t3 = Bool(f"x_{t3}")
        #     key = ProxyRepairMaxSat._clause_key(t1, t2, t3)
        #     if key not in used_keys and ProxyRepairMaxSat._clause_key(t2, t1, t3) not in used_keys:
        #         hard_clauses.add(Or(Not(x_t1), Not(x_t2), x_t3))
        #         used_keys.add(key)
        # ----------------------------------------------------------------------------------------------------
        # OLD MEMORY EFFICIENT:
        # if admissible_col is not None:
        #     # group unique (s,o) pairs by each admissible value a
        #     group_by_a_star = defaultdict(set)
        #     for (s, o, a) in D_star:
        #         group_by_a_star[a].add((s, o))
        #
        #     for a, pairs in group_by_a_star.items():
        #         uniq_pairs = list(pairs)  # unique (s,o) for this a
        #         for (s1, o1), (s2, o2) in combinations(uniq_pairs, 2):
        #             if s1 == s2 or o1 == o2:
        #                 continue
        #             t1 = (s1, o1, a)
        #             t2 = (s2, o2, a)
        #             t3 = (s1, o2, a)
        #             x_t1 = Bool(f"x_{t1}")
        #             x_t2 = Bool(f"x_{t2}")
        #             x_t3 = Bool(f"x_{t3}")
        #             hard_clauses.add(Or(Not(x_t1), Not(x_t2), x_t3))
        # else:
        #     # unconditional: unique (s,o) pairs across all D_star
        #     uniq_pairs = list(set(D_star))  # D_star already contains (s,o)
        #     for (s1, o1), (s2, o2) in combinations(uniq_pairs, 2):
        #         if s1 == s2 or o1 == o2:
        #             continue
        #         t1 = (s1, o1)
        #         t2 = (s2, o2)
        #         t3 = (s1, o2)
        #         x_t1 = Bool(f"x_{t1}")
        #         x_t2 = Bool(f"x_{t2}")
        #         x_t3 = Bool(f"x_{t3}")
        #         hard_clauses.add(Or(Not(x_t1), Not(x_t2), x_t3))
        #
        # return soft_clauses, hard_clauses, D_star
