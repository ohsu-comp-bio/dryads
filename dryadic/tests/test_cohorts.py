
import os
import sys
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "resources")
sys.path.extend([os.path.join(base_dir, '../..')])

from dryadic.features.cohorts.base import UniCohort
from dryadic.features.cohorts import *
from dryadic.features.mutations import MuType

import numpy as np
import pandas as pd
from itertools import combinations as combn


def load_omic_data(data_lbl):
    return pd.read_csv(os.path.join(data_dir, "{}.txt.gz".format(data_lbl)),
                       sep='\t', index_col=0)


def main():

    expr_data = load_omic_data('expr')
    mut_data = load_omic_data('variants')
    cdata = UniCohort(expr_data, cv_seed=None, test_prop=0)

    assert len(cdata.get_train_samples()) == expr_data.shape[0]
    assert len(cdata.get_test_samples()) == 0
    assert len(cdata.get_samples()) == expr_data.shape[0]
    assert set(cdata.get_train_samples()) == set(expr_data.index)
    assert set(cdata.get_samples()) == set(expr_data.index)

    assert cdata.get_seed() is None
    cdata.update_seed(23)
    assert cdata.get_seed() == 23
    assert len(cdata.get_train_samples()) == expr_data.shape[0]
    assert len(cdata.get_test_samples()) == 0
    assert len(cdata.get_samples()) == expr_data.shape[0]

    cdata.update_seed(551, test_prop=1./3)
    assert cdata.get_seed() == 551
    assert len(cdata.get_samples()) == expr_data.shape[0]

    assert ((set(cdata.get_train_samples()) | set(cdata.get_test_samples()))
            == set(expr_data.index)), (
                "Cohort training and testing samples should be a partition "
                "of the original samples!"
                )

    cdata = BaseMutationCohort(expr_data, mut_data, mut_levels=['Exon'],
                               mut_genes=['TP53'], cv_seed=139, test_prop=0.2)
    assert cdata.get_seed() == 139
    assert len(cdata.get_samples()) == expr_data.shape[0]

    for exn, mut_df in mut_data[mut_data.Gene == 'TP53'].groupby('Exon'):
        assert set(cdata.mtree[exn]) == set(
            mut_data.Sample[(mut_data.Gene == 'TP53')
                            & (mut_data.Exon == exn)]
            )

    for exn1, exn2 in combn(set(mut_data.Exon[mut_data.Gene == 'TP53']), 2):
        mtype = MuType({('Exon', (exn1, exn2)): None})

        train_expr, train_phn = cdata.train_data(mtype)
        assert train_phn.shape == (len(cdata.get_train_samples()),)
        assert train_expr.shape == (len(cdata.get_train_samples()),
                                    len(cdata.get_features()))

        test_expr, test_phn = cdata.test_data(mtype)
        assert (set(train_expr.index) & set(test_expr.index)) == set()
        assert ((set(train_expr.index) | set(test_expr.index))
                == set(cdata.get_samples()))
        assert sorted(train_expr.columns) == sorted(test_expr.columns)

        train_mut_smps = np.array(cdata.get_train_samples())[train_phn]
        test_mut_smps = np.array(cdata.get_test_samples())[test_phn]
        assert (set(train_mut_smps) & set(test_mut_smps)) == set()

        assert (sorted((set(train_mut_smps) | set(test_mut_smps)))
                == sorted(set(mut_data.Sample[
                    (mut_data.Gene == 'TP53')
                    & mut_data.Exon.isin([exn1, exn2])
                    ])))

    cdata = BaseMutationCohort(
        expr_data, mut_data, mut_levels=['Domain_Pfam', 'Exon'],
        mut_genes=['TP53'], domain_dir=data_dir, test_prop=0, cv_seed=7753
        )

    assert len(cdata.mtree['None'].get_samples()) == 2, (
        "Exactly two samples should have mutations "
        "not overlapping a Pfam domain!"
        )

    assert len(cdata.mtree['PF00870']['5/11']) == 16, (
        "Exactly sixteen samples should have a Pfam "
        "Domain PF00870 mutation on the 5th Exon!"
        )

    assert (cdata.mutex_test(
        MuType({('Domain_Pfam', 'PF00870'): None}),
        MuType({('Domain_Pfam', 'None'): None}))) == (0, 1)

    print("All Cohort tests passed successfully!")


if __name__ == '__main__':
    main()

