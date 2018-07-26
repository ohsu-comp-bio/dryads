
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../..')])

from dryad.features.cohorts.base import UniCohort
import pandas as pd


def main():
    expr_data = pd.read_csv(os.path.join(base_dir, "resources/expr.txt.gz"),
                            sep='\t', index_col=0)

    cdata = UniCohort(expr_data,
                      train_samps=list(expr_data.index[:120]),
                      test_samps=list(expr_data.index[120:]))

    assert (cdata.train_samps | cdata.test_samps) == set(expr_data.index), (
        "Cohort training and testing samples should be a partition "
        "of the original samples!"
        )

    print("All Cohort tests passed successfully!")


if __name__ == '__main__':
    main()

