
import os
import sys
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "resources")
sys.path.extend([os.path.join(base_dir, '../..')])

from dryadic.features.cohorts.base import UniCohort
from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.mutations import MuType

import numpy as np
import pandas as pd
from itertools import combinations as combn


def load_omic_data(data_lbl):
    return pd.read_csv(os.path.join(data_dir, "{}.txt.gz".format(data_lbl)),
                       sep='\t', index_col=0)


def load_muts(muts_lbl):
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'resources',
                     "muts_{}.tsv".format(muts_lbl)),
        engine='python', sep='\t', comment='#',
        names=['Gene', 'Form', 'Sample', 'Protein', 'Transcript', 'Exon',
               'ref_count', 'alt_count', 'PolyPhen']
        )


def check_samp_split(cdata, expr_samps):
    assert ((set(cdata.get_train_samples()) | set(cdata.get_test_samples()))
            == set(expr_samps)), (
                "Cohort training and testing samples should be a partition "
                "of the original samples!"
                )


def main():
    expr_data = load_omic_data('expr')
    cdata = UniCohort(expr_data, cv_seed=None, test_prop=0)

    assert len(cdata.get_train_samples()) == expr_data.shape[0]
    assert len(cdata.get_test_samples()) == 0
    assert len(cdata.get_samples()) == expr_data.shape[0]
    assert set(cdata.get_train_samples()) == set(expr_data.index)
    assert set(cdata.get_samples()) == set(expr_data.index)
    check_samp_split(cdata, expr_data.index)

    assert cdata.get_seed() is None
    cdata.update_split(new_seed=23)
    assert cdata.get_seed() == 23
    assert len(cdata.get_train_samples()) == expr_data.shape[0]
    assert len(cdata.get_test_samples()) == 0
    assert len(cdata.get_samples()) == expr_data.shape[0]
    check_samp_split(cdata, expr_data.index)

    cdata.update_split(new_seed=551, test_prop=1./3)
    assert cdata.get_seed() == 551
    assert len(cdata.get_samples()) == expr_data.shape[0]
    assert cdata.train_data(None)[0].shape == (expr_data.shape[0] * 2/3,
                                               expr_data.shape[1])
    check_samp_split(cdata, expr_data.index)

    cdata.update_split(new_seed=551, test_samps=expr_data.index[:20])
    assert cdata.test_data(None)[0].shape == (20, expr_data.shape[1])
    check_samp_split(cdata, expr_data.index)

    mut_data = load_omic_data('variants')
    cdata = BaseMutationCohort(
        expr_data, mut_data,
        mut_levels=[['Form', 'Exon', 'Location'], ['Exon', 'Form_base']],
        mut_genes=['TP53'], cv_seed=139, test_prop=0.2
        )

    assert cdata.get_seed() == 139
    assert len(cdata.get_samples()) == expr_data.shape[0]
    check_samp_split(cdata, expr_data.index)
    assert len(cdata.mtrees) == 2
    assert (cdata.muts.Gene == 'TP53').all()
    assert cdata.data_hash() == cdata.data_hash()

    for exn1, exn2 in combn(set(mut_data.Exon[mut_data.Gene == 'TP53']), 2):
        mtype = MuType({('Exon', (exn1, exn2)): None})

        train_expr, train_phn = cdata.train_data(mtype)
        assert len(cdata.mtrees) == 2
        assert train_phn.shape == (len(cdata.get_train_samples()),)
        assert train_expr.shape == (len(cdata.get_train_samples()),
                                    len(cdata.get_features()))

        test_expr, test_phn = cdata.test_data(mtype)
        assert len(cdata.mtrees) == 2
        assert (set(train_expr.index) & set(test_expr.index)) == set()
        assert ((set(train_expr.index) | set(test_expr.index))
                == set(cdata.get_samples()))
        assert sorted(train_expr.columns) == sorted(test_expr.columns)

        train_mut_smps = np.array(cdata.get_train_samples())[train_phn]
        test_mut_smps = np.array(cdata.get_test_samples())[test_phn]
        assert (set(train_mut_smps) & set(test_mut_smps)) == set()

        use_muts = mut_data.loc[(mut_data.Gene == 'TP53')
                                & mut_data.Exon.isin([exn1, exn2])]
        assert (sorted((set(train_mut_smps) | set(test_mut_smps)))
                == sorted(set(use_muts.Sample)))

        for samp, lf_ant in mtype.get_leaf_annot(
                cdata.mtrees['Exon', 'Form_base'], ['PolyPhen']).items():
            assert (set(lf_ant['PolyPhen'])
                    == set(use_muts.loc[use_muts.Sample
                                        == samp].PolyPhen.tolist()))

    cdata = BaseMutationCohort(
        expr_data, mut_data, mut_levels=[['Domain_Pfam', 'Exon']],
        mut_genes=['TP53'], domain_dir=data_dir, test_prop=0, cv_seed=7753
        )

    assert (len(cdata.mtrees['Domain_Pfam', 'Exon']['none'].get_samples())
            == 2), ("Exactly two samples should have mutations "
                    "not overlapping a Pfam domain!")

    assert (len(cdata.mtrees['Domain_Pfam', 'Exon']['PF00870']['5/11'])
            == 16), ("Exactly sixteen samples should have a Pfam "
                     "Domain PF00870 mutation on the 5th Exon!")

    assert (cdata.mutex_test(
        MuType({('Domain_Pfam', 'PF00870'): None}),
        MuType({('Domain_Pfam', 'none'): None}))) == (0, 1)
    assert len(cdata.mtrees) == 1

    print("All Cohort tests passed successfully!")


if __name__ == '__main__':
    main()

