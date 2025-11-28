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

        soft_clauses = []
        hard_clauses = set()
        D_star = set()

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

            D = list(df[cols].itertuples(index=False, name=None))
            D_shortened = list(set(D))

            soft_clauses_for_criterion, hard_clauses_for_criterion, D_star_for_criterion = (
                self._conversion_to_solving_general_3cnf(D_shortened, admissible_col))
            soft_clauses += soft_clauses_for_criterion
            hard_clauses = hard_clauses.union(hard_clauses_for_criterion)
            D_star = D_star.union(D_star_for_criterion)

        opt = Optimize()
        # Add constraints to the optimizer
        for clause in soft_clauses:
            opt.add_soft(clause, weight=1)
        for clause in hard_clauses:
            opt.add(clause)

        if opt.check() != sat:
            print("No satisfying assignment found")
            return

        model = opt.model()

        # Compute DR = satisfying assignment
        DR = set()
        for t in D_star:
            if model.evaluate(Bool(f"x_{t}")):
                DR.add(t)

        # repair = number of mismatched tuples
        list_symmetric_difference = [row for row in D if row not in DR] + [row for row in DR if row not in D]
        repair = len(list_symmetric_difference)

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
    def _clause_key(t1, t2, t3):
        """
        Generates a unique key for a clause defined by three tuples.

        Used internally to avoid duplicate logical clauses.
        """
        return tuple(sorted([f"x_{t1}", f"x_{t2}", f"x_{t3}"]))

    @staticmethod
    def _conversion_to_solving_general_3cnf(D, admissible_col):
        """
        Converts a dataset into a 3-CNF MaxSAT formulation.

        Constructs:
          - D_star: the set of all tuple combinations satisfying MVD structure
          - Soft clauses: one per tuple, encouraging consistency with the data
          - Hard clauses: 3-CNF constraints enforcing fairness dependencies

        Returns
        -------
        tuple
            (soft_clauses, hard_clauses, D_star)
        """
        soft_clauses = []
        hard_clauses = set()

        # Step 1: Create D_star
        if admissible_col is not None:
            group_by_a = defaultdict(list)
            for s, o, a in D:
                group_by_a[a].append((s, o))
            D_star = {
                (s1, o2, a)
                for a, pairs in group_by_a.items()
                for s1, _ in pairs
                for _, o2 in pairs
            }
        else:
            D_star = {
                (s1, o2)
                for s1, _ in D
                for _, o2 in D
            }

        # Step 2: Create soft clauses for each tuple
        for t in D_star:
            x_t = Bool(f"x_{t}")
            if t in D:
                soft_clauses.append(x_t)
            else:
                soft_clauses.append(Not(x_t))

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
        # NEW MEMORY EFFICIENT:
        if admissible_col is not None:
            # group unique (s,o) pairs by each admissible value a
            group_by_a_star = defaultdict(set)
            for (s, o, a) in D_star:
                group_by_a_star[a].add((s, o))

            for a, pairs in group_by_a_star.items():
                uniq_pairs = list(pairs)  # unique (s,o) for this a
                for (s1, o1), (s2, o2) in combinations(uniq_pairs, 2):
                    if s1 == s2 or o1 == o2:
                        continue
                    t1 = (s1, o1, a)
                    t2 = (s2, o2, a)
                    t3 = (s1, o2, a)
                    x_t1 = Bool(f"x_{t1}")
                    x_t2 = Bool(f"x_{t2}")
                    x_t3 = Bool(f"x_{t3}")
                    hard_clauses.add(Or(Not(x_t1), Not(x_t2), x_t3))
        else:
            # unconditional: unique (s,o) pairs across all D_star
            uniq_pairs = list(set(D_star))  # D_star already contains (s,o)
            for (s1, o1), (s2, o2) in combinations(uniq_pairs, 2):
                if s1 == s2 or o1 == o2:
                    continue
                t1 = (s1, o1)
                t2 = (s2, o2)
                t3 = (s1, o2)
                x_t1 = Bool(f"x_{t1}")
                x_t2 = Bool(f"x_{t2}")
                x_t3 = Bool(f"x_{t3}")
                hard_clauses.add(Or(Not(x_t1), Not(x_t2), x_t3))

        return soft_clauses, hard_clauses, D_star
