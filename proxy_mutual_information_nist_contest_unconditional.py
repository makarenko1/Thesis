import time

from mechanisms.mst_unconditional import run_mst_unconditional


class ProxyMutualInformationNistContestUnconditional:

    def __init__(self, datapath):
        self.datapath = datapath

    def calculate(self, s_col, o_col, domain):
        start_time = time.time()

        mi = run_mst_unconditional(self.datapath, domain, s_col, o_col)

        elapsed_time = time.time() - start_time
        print(f"NIST MST: Proxy Mutual Information between '{s_col}' and '{o_col}': {mi:.4f}. "
              f"Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)
