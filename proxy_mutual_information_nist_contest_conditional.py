import time

from mechanisms.mst_conditional import run_mst_conditional


class ProxyMutualInformationNistContestConditional:

    def __init__(self, datapath):
        self.datapath = datapath

    def calculate(self, s_col, o_col, a_col, domain):
        start_time = time.time()

        mi = run_mst_conditional(self.datapath, domain, s_col, o_col, a_col)

        elapsed_time = time.time() - start_time
        print(f"NIST MST: Proxy Conditional Mutual Information between '{s_col}' and '{o_col} conditioned on {a_col}': "
              f"{mi:.4f}. Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)
