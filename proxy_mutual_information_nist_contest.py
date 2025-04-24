import time

from mechanisms.mst import run_mst


class ProxyMutualInformationNistContest:

    def __init__(self, datapath):
        self.datapath = datapath

    def calculate(self, column_name_1, column_name_2, domain):
        start_time = time.time()

        mi = run_mst(self.datapath, domain)

        elapsed_time = time.time() - start_time
        print(f"NIST MST: Proxy Mutual Information between '{column_name_1}' and '{column_name_2}': {mi:.4f}. "
              f"Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)
