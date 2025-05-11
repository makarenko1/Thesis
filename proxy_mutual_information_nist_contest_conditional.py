import time

from mechanisms.mst_conditional import run_mst_conditional


class ProxyMutualInformationNistContestConditional:
    """
    Computes a proxy for conditional mutual information using the MST-based mechanism
    from the NIST Differential Privacy Synthetic Data Challenge (2018).
    This version supports conditioning on a third attribute.
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

    def calculate(self, s_col, o_col, a_col, domain):
        """
        Calculates the proxy conditional mutual information I(S;O | A)
        using the MST-based synthetic data mechanism.

        Parameters:
        -----------
        s_col : str
            Sensitive attribute (S).
        o_col : str
            Outcome attribute (O).
        a_col : str
            Attribute to condition on (A).
        domain : str
            Path to the domain specification (JSON format).

        Returns:
        --------
        float
            Proxy conditional mutual information score.
        """
        start_time = time.time()

        mi = run_mst_conditional(self.datapath, domain, s_col, o_col, a_col)

        elapsed_time = time.time() - start_time
        print(f"NIST MST: Proxy Conditional Mutual Information between '{s_col}' and '{o_col} conditioned on {a_col}': "
              f"{mi:.4f}. Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)
