import time
from itertools import product

import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
        self.dataset[s_col] = LabelEncoder().fit_transform(self.dataset[s_col])
        self.dataset[o_col] = LabelEncoder().fit_transform(self.dataset[o_col])
        self.dataset[a_col] = LabelEncoder().fit_transform(self.dataset[a_col])

        s_col_values = self.dataset[s_col].to_numpy()
        o_col_values = self.dataset[o_col].to_numpy()
        a_col_values = self.dataset[a_col].to_numpy()

        start_time = time.time()

        opt = Optimize()
        soft_clauses, hard_clauses, D_tuples, D_star = self.conversion_to_solving_general_3cnf(
            s_col_values, o_col_values, a_col_values
        )

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
        for t in D_tuples:
            if model.evaluate(Bool(f"x_{t}")):
                DR.add(t)

        # UR = number of mismatched tuples
        UR = len(set(D_tuples).symmetric_difference(DR))

        elapsed_time = time.time() - start_time
        print(f"MaxSAT: Proxy Mutual Information between '{s_col}' and '{o_col}': {UR:.4f}. "
              f"Calculation took {elapsed_time:.3f} seconds.")
        return UR

    def conversion_to_solving_general_3cnf(self, s_col_values, o_col_values, a_col_values):
        """
        Constructs soft and hard clauses for the MaxSAT solver using the 3CNF encoding of MVD constraints.

        Parameters:
        -----------
        s_col_values : np.ndarray
            Encoded values of the S attribute.
        o_col_values : np.ndarray
            Encoded values of the O attribute.
        a_col_values : np.ndarray
            Encoded values of the A attribute.

        Returns:
        --------
        tuple
            soft_clauses : list
                Soft clauses for optimization (encouraged but not required).
            hard_clauses : list
                Hard constraints encoding MVD-based 3CNF rules.
            D_tuples : list
                Actual observed tuples (S, O, A) in the dataset.
            D_star : set
                All syntactically valid (S, O, A) tuples based on MVD expansion.
        """
        soft_clauses = []
        hard_clauses = []

        # Step 1: Generate all valid combinations (S, O, A) based on A-grouping
        D_star = set()
        for a_val in set(a_col_values):
            S_given_A = set(s for s, a in zip(s_col_values, a_col_values) if a == a_val)
            O_given_A = set(o for o, a in zip(o_col_values, a_col_values) if a == a_val)
            for s, o in product(S_given_A, O_given_A):
                D_star.add((s, o, a_val))

        D_tuples = list(zip(s_col_values, o_col_values, a_col_values))

        # Step 2: Create soft clauses for each (S, O, A) tuple
        for t in D_star:
            x_t = Bool(f"x_{t}")
            if t in D_tuples:
                soft_clauses.append(x_t)
            else:
                soft_clauses.append(Not(x_t))

        # Step 3: Enforce MVD 3CNF constraints
        C = set()
        for (s1, o1, a1) in D_star:
            for (s2, o2, a2) in D_star:
                if a1 == a2:
                    C.add((s1, o1, s2, o2, a1))

        for (s1, o1, s2, o2, a) in C:
            t1 = (s1, o1, a)
            t2 = (s2, o2, a)
            t3 = (s1, o2, a)
            x_t1 = Bool(f"x_{t1}")
            x_t2 = Bool(f"x_{t2}")
            x_t3 = Bool(f"x_{t3}")
            hard_clauses.append(Or(Not(x_t1), Not(x_t2), x_t3))

        return soft_clauses, hard_clauses, D_tuples, D_star
