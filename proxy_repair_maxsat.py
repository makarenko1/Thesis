from itertools import product


def conversion_to_solving_general_3cnf(D, s_col, o_col, a_col):
    """
    Args:
        D: a pandas DataFrame representing the dataset
        s_col, o_col, a_col: column names corresponding to X, Y, Z
    Returns:
        hard_clauses: list of hard clauses (list of tuples of literals)
        soft_clauses: list of soft clauses (single literals)
    """
    hard_clauses = []
    soft_clauses = []

    # Step 1: Compute D*
    s_col_values = D[s_col].to_numpy()
    o_col_values = D[o_col].to_numpy()
    a_col_values = D[a_col].to_numpy()

    D_star = set()

    for z_val in set(a_col_values):
        X_given_Z = set(x for x, z in zip(s_col_values, a_col_values) if z == z_val)
        Y_given_Z = set(y for y, z in zip(o_col_values, a_col_values) if z == z_val)
        for x, y in product(X_given_Z, Y_given_Z):
            D_star.add((x, y, z_val))

    # Step 2: Soft clauses
    D_tuples = set(zip(s_col_values, o_col_values, a_col_values))

    for t in D_star:
        if t in D_tuples:
            soft_clauses.append(('X', t))
        else:
            soft_clauses.append(('¬X', t))

    # Step 3: Compute C
    # It's just D* again (in the original algorithm C is same as D*)

    C = D_star

    # Step 4: Hard clauses
    for (x, y, z) in C:
        t1 = (x, None, z)
        t2 = (None, y, z)
        t3 = (x, y, None)

        hard_clauses.append((('¬X', t1), ('¬X', t2), ('X', t3)))

    return hard_clauses, soft_clauses
