import time
from itertools import product

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from z3 import Bool, Or, Not, Optimize, sat, If


class ProxyRepairMaxSat:

    def __init__(self, datapath):
        self.dataset = pd.read_csv(datapath)

    def calculate(self, s_col, o_col, a_col):
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
        soft_clauses, hard_clauses, D_tuples = self.conversion_to_solving_general_3cnf(s_col_values, o_col_values, a_col_values)

        # Add soft clauses to objective
        for clause in soft_clauses:
            opt.add_soft(clause, weight=1)
        for clause in hard_clauses:
            opt.add(clause)

        if opt.check() != sat:
            print("No satisfying assignment found.")
            return

        model = opt.model()

        DR = set()
        for t in D_tuples:
            if model.evaluate(Bool(f"x_{t}")):
                DR.add(t)

        UR = len(set(D_tuples).symmetric_difference(DR))

        elapsed_time = time.time() - start_time
        print(f"MaxSAT: Proxy Mutual Information between '{s_col}' and '{o_col}': {UR:.4f}. "
              f"Calculation took {elapsed_time:.3f} seconds.")
        return UR

    def conversion_to_solving_general_3cnf(self, s_col_values, o_col_values, a_col_values):
        soft_clauses = []
        hard_clauses = []

        D_star = set()
        for a_val in set(a_col_values):
            S_given_A = set(s for s, a in zip(s_col_values, a_col_values) if a == a_val)
            O_given_A = set(o for o, a in zip(o_col_values, a_col_values) if a == a_val)
            for s, o in product(S_given_A, O_given_A):
                D_star.add((s, o, a_val))

        D_tuples = list(zip(s_col_values, o_col_values, a_col_values))

        for t in D_star:
            x_t = Bool(f"x_{t}")
            if t in D_tuples:
                soft_clauses.append(x_t)
            else:
                soft_clauses.append(Not(x_t))

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

        return soft_clauses, hard_clauses, D_tuples
