
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../..')])

from dryadic.features.cohorts.base import UniCohort
from dryadic.features.cohorts import *
import pandas as pd


def main():
    data_dir = os.path.join(base_dir, "resources")

    expr_data = pd.read_csv(os.path.join(data_dir, "expr.txt.gz"),
                            sep='\t', index_col=0)
    mut_data = pd.read_csv(os.path.join(data_dir, "variants.txt.gz"),
                           sep='\t', index_col=0)

    cdata = UniCohort(expr_data,
                      train_samps=list(expr_data.index[:120]),
                      test_samps=list(expr_data.index[120:]))

    assert (cdata.train_samps | cdata.test_samps) == set(expr_data.index), (
        "Cohort training and testing samples should be a partition "
        "of the original samples!"
        )

    cdata = BaseMutationCohort(
        expr_data, mut_data, domain_dir=data_dir, mut_genes=['TP53'],
        mut_levels=['Domain_Pfam', 'Exon'], cv_prop=0.8, cv_seed=139
        )

    assert len(cdata.train_mut['None']) == 2, (
        "Exactly two training samples should have mutations "
        "not overlapping a Pfam domain!"
        )

    assert len(cdata.test_mut['PF00870']['5/11']) == 5, (
        "Exactly five testing samples should have a Pfam "
        "Domain PF00870 mutation on the 5th Exon!"
        )

    print("All Cohort tests passed successfully!")


if __name__ == '__main__':
    main()

