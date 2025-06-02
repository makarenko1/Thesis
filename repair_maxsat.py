import time
from collections import defaultdict

import pandas as pd
from z3 import Bool, Or, Not, Optimize, sat


class ProxyRepairMaxSat:
    """
    This class approximates conditional mutual information using a MaxSAT-based proxy.
    It does so by encoding constraints derived from multivalued dependencies (MVDs)
    into a weighted MaxSAT problem and solving it using Z3's optimizer.
    """

    def __init__(self, datapath):
        """
        Initializes the ProxyRepairMaxSat object by loading the dataset.

        Parameters:
        -----------
        datapath : str
            Path to the CSV dataset file.
        """
        self.dataset = pd.read_csv(datapath)

    def calculate(self, s_col, o_col, a_col):
        """
        Computes the MaxSAT-based proxy score for conditional dependence between s_col and o_col given a_col.

        Parameters:
        -----------
        s_col : str
            Name of the S attribute (e.g. sensitive).
        o_col : str
            Name of the O attribute (e.g. outcome).
        a_col : str
            Name of the A attribute (e.g. auxiliary/context variable).

        Returns:
        --------
        int
            The number of tuples violating multivalued dependency constraints.
        """
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
        self.dataset.dropna(inplace=True, subset=[s_col, o_col, a_col])

        D = list(self.dataset[[s_col, o_col, a_col]].itertuples(index=False, name=None))

        start_time = time.time()

        opt = Optimize()
        soft_clauses, hard_clauses = self.conversion_to_solving_general_3cnf(D)

        # Add constraints to the optimizer
        for clause in soft_clauses:
            opt.add_soft(clause, weight=1)
        for clause in hard_clauses:
            opt.add(clause)

        if opt.check() != sat:
            print("No satisfying assignment found.")
            return

        model = opt.model()

        # Compute DR = satisfying assignment
        DR = set()
        for t in D:
            if model.evaluate(Bool(f"x_{t}")):
                DR.add(t)

        # UR = number of mismatched tuples
        list_symmetric_difference = [row for row in D if row not in DR] + [row for row in DR if row not in D]
        UR = len(list_symmetric_difference)

        elapsed_time = time.time() - start_time
        print(f"Repair MaxSAT: The score for dependency '{s_col}' ⊥⊥ '{o_col}' | '{a_col}': {UR}. "
              f"Calculation took {elapsed_time:.3f} seconds.")
        return UR

    @staticmethod
    def clause_key(t1, t2, t3):
        return tuple(sorted([f"x_{t1}", f"x_{t2}", f"x_{t3}"]))

    @staticmethod
    def conversion_to_solving_general_3cnf(D):
        """
        Constructs soft and hard clauses for the MaxSAT solver using the 3CNF encoding of MVD constraints.

        Parameters:
        -----------
        D : np.ndarray
            Tuples of s_col values, o_col values and a_col values.

        Returns:
        --------
        tuple
            soft_clauses : list
                Soft clauses for optimization (encouraged but not required).
            hard_clauses : list
                Hard constraints encoding MVD-based 3CNF rules.
        """
        soft_clauses = list()
        hard_clauses = set()

        # Step 1: Generate all valid combinations (S, O, A) based on A-grouping
        group_by_a = defaultdict(list)
        for s, o, a in D:
            group_by_a[a].append((s, o))
        D_star = {
            (s1, o2, a)
            for a, pairs in group_by_a.items()
            for s1, _ in pairs
            for _, o2 in pairs
        }

        # Step 2: Create soft clauses for each (S, O, A) tuple
        for t in D_star:
            x_t = Bool(f"x_{t}")
            if t in D:
                soft_clauses.append(x_t)
            else:
                soft_clauses.append(Not(x_t))

        # Step 3: Enforce MVD 3CNF constraints
        C = set()
        for (s1, o1, a1) in D_star:
            for (s2, o2, a2) in D_star:
                if a1 == a2 and s1 != s2 and o1 != o2:
                    C.add((s1, o1, s2, o2, a1))

        used_keys = set()
        for (s1, o1, s2, o2, a) in C:
            t1 = (s1, o1, a)
            t2 = (s2, o2, a)
            t3 = (s1, o2, a)
            x_t1 = Bool(f"x_{t1}")
            x_t2 = Bool(f"x_{t2}")
            x_t3 = Bool(f"x_{t3}")
            key = ProxyRepairMaxSat.clause_key(t1, t2, t3)
            if key not in used_keys and ProxyRepairMaxSat.clause_key(t2, t1, t3) not in used_keys:
                hard_clauses.add(Or(Not(x_t1), Not(x_t2), x_t3))
                used_keys.add(key)

        return soft_clauses, hard_clauses
