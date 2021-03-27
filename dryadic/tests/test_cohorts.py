
"""
Unit tests for classes representing collections of -omic data collected on
cohorts of samples.

See Also:
    :class:`dryadic.features.cohorts`: The class tested herein.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>
"""

from ..features.mutations import MuType
from .test_mtrees import load_muts
from dryadic.features.cohorts.base import UniCohort
from dryadic.features.cohorts import BaseMutationCohort

import numpy as np
import pandas as pd

import os
import random
from string import ascii_uppercase as LETTERS
from itertools import product


def load_omic_data(data_lbl):
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources',
                                    "{}.txt.gz".format(data_lbl)),
                       sep='\t', index_col=0)


def generate_cohort(muts_lbl, mut_lvls, **coh_args):
    muts_df = load_muts(muts_lbl)
    expr_data = generate_expr_data(set(muts_df.Sample))

    return BaseMutationCohort(expr_data, muts_df, mut_lvls, **coh_args)


def generate_expr_data(samps, gene_count=1000, seed=None):
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    genes = set()
    while len(genes) < gene_count:
        new_gene = ''.join(random.sample(LETTERS, random.choice([4, 5, 6])))

        if new_gene not in genes:
            genes |= {new_gene}

    return pd.DataFrame(np.random.randn(len(samps), gene_count),
                        index=samps, columns=genes)


def pytest_generate_tests(metafunc):
    if hasattr(metafunc.cls, 'params'):
        if isinstance(metafunc.cls.params, dict):
            if set(metafunc.cls.params.keys()) == {'muts', 'mut_levels',
                                                   'cv_seed', 'test_prop'}:
                metafunc.parametrize(
                    'cdata',
                    [generate_cohort(muts_lbl, lvls,
                                     cv_seed=cv_seed, test_prop=test_prop)
                     for muts_lbl, lvls, cv_seed, test_prop
                     in product(*[metafunc.cls.params[k]
                                  for k in ['muts', 'mut_levels',
                                            'cv_seed', 'test_prop']])],
                    )


class TestCaseUni:

    def test_samps(self):
        expr_data = load_omic_data('expr')
        base_cdata = UniCohort(expr_data, cv_seed=None, test_prop=0)

        for cv_seed in [None, 0, 23, 555]:
            for test_prop in [0, 0.1, 0.2, 0.5]:
                base_cdata.update_split(new_seed=cv_seed, test_prop=test_prop)
                new_cdata = UniCohort(expr_data,
                                      cv_seed=cv_seed, test_prop=test_prop)

                for cdata in [base_cdata, new_cdata]:
                    assert cdata.get_seed() == cv_seed
                    assert len(cdata.get_samples()) == expr_data.shape[0]
                    assert set(cdata.get_samples()) == set(expr_data.index)

                    train_samps = cdata.get_train_samples()
                    test_samps = cdata.get_test_samples()
                    assert len(set(train_samps) & set(test_samps)) == 0
                    assert (set(train_samps)
                            | set(test_samps)) == set(expr_data.index)

        base_cdata.update_split(new_seed=551, test_samps=expr_data.index[:20])
        assert base_cdata.test_data(None)[0].shape == (20, expr_data.shape[1])


class TestCaseInit:

    params = {
        'muts': ['medium'],
        'mut_levels': [[['Gene', 'Form']], [['Gene', 'Protein']]],
        'cv_seed': [0, 133, 9077, 77011], 'test_prop': [0, 0.25]
        }

    def test_init(self, cdata):
        assert len(cdata.mtrees) == 1

    def test_pheno(self, cdata):
        mtype_dict = {lvls: mtree.combtypes(comb_sizes=(1, ),
                                            min_type_size=1)
                      for lvls, mtree in cdata.mtrees.items()}

        train_samps = set(cdata.get_train_samples())
        for lvls, mtypes in mtype_dict.items():
            assert len(mtypes) > 1

            for mtype in mtypes:
                mtype_samps = mtype.get_samples(cdata.mtrees[lvls])
                train_expr, train_pheno = cdata.train_data(mtype)
                assert train_pheno.sum() == len(mtype_samps & train_samps)

