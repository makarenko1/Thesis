import time

from mechanisms.mst_unconditional import run_mst_unconditional
from mechanisms.mst_conditional import run_mst_conditional


class ProxyMutualInformationNistContest:
    """
    Computes a proxy for (conditional or unconditional) mutual information
    using the MST-based mechanism from the NIST Differential Privacy Synthetic Data Challenge (2018).
    """

    def __init__(self, datapath):
        """
        Initializes the proxy estimator with a dataset path.

        Parameters:
        -----------
        datapath : str
            Path to the CSV dataset file.
        """
        self.datapath = datapath

    def calculate(self, s_col, o_col, domain, a_col=None):
        """
        Calculates the proxy mutual information score.

        Parameters:
        -----------
        s_col : str
            Sensitive attribute (S).
        o_col : str
            Outcome attribute (O).
        domain : str
            Path to the domain specification (JSON format).
        a_col : str or None, optional
            Attribute to condition on (A). If None, compute unconditional mutual information.

        Returns:
        --------
        float
            Proxy (conditional) mutual information score.
        """
        start_time = time.time()

        if a_col is None:
            mi = run_mst_unconditional(self.datapath, domain, s_col, o_col)
            print(f"NIST MST: Proxy Mutual Information between '{s_col}' and '{o_col}': {mi:.4f}.", end=" ")
        else:
            mi = run_mst_conditional(self.datapath, domain, s_col, o_col, a_col)
            print(f"NIST MST: Proxy Conditional Mutual Information between '{s_col}' and '{o_col}' conditioned on "
                  f"'{a_col}': {mi:.4f}.", end=" ")

        elapsed_time = time.time() - start_time
        print(f"Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)
