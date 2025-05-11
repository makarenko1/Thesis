import time

from mechanisms.mst_unconditional import run_mst_unconditional


class ProxyMutualInformationNistContestUnconditional:
    """
    Computes a proxy for mutual information using the MST-based mechanism
    from the NIST Differential Privacy Synthetic Data Challenge (2018).
    This version is for unconditional mutual information I(S;O).
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

    def calculate(self, s_col, o_col, domain):
        """
        Calculates the proxy mutual information I(S;O)
        using the MST-based synthetic data mechanism.

        Parameters:
        -----------
        s_col : str
            Sensitive attribute (S).
        o_col : str
            Outcome attribute (O).
        domain : str
            Path to the domain specification (JSON format).

        Returns:
        --------
        float
            Proxy mutual information score.
        """
        start_time = time.time()

        mi = run_mst_unconditional(self.datapath, domain, s_col, o_col)

        elapsed_time = time.time() - start_time
        print(f"NIST MST: Proxy Mutual Information between '{s_col}' and '{o_col}': {mi:.4f}. "
              f"Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)
