
"""Unit tests for hierarchical storage of mutation cohorts.

See Also:
    :class:`..features.mutations.MuTree`: The class tested herein.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

import pytest
from ..features.mutations import MuType, MuTree
from .test_mtypes import mtype_generator
from .test_cohorts import load_muts

import pandas as pd
from itertools import product, chain
from itertools import combinations as combn


def pytest_generate_tests(metafunc):
    funcargdict = {param_key: None
                   for param_key in ('muts', 'mut_lvls', 'mtypes')}

    if metafunc.function.__code__.co_argcount == 1:
        pass

    elif metafunc.function.__code__.co_argcount >= 2:
        if hasattr(metafunc.cls, 'params'):
            if not isinstance(metafunc.cls.params, dict):
                raise ValueError("Testing case parameters must be given "
                                 "as a dictionary!")

            for param_key in funcargdict:
                if param_key in metafunc.cls.params:
                    funcargdict[param_key] = metafunc.cls.params[param_key]

            if metafunc.function.__name__ in metafunc.cls.params:
                func_params = metafunc.cls.params[metafunc.function.__name__]

                for param_key in funcargdict:
                    if param_key in func_params:
                        funcargdict[param_key] = func_params[param_key]

        else:
            raise ValueError(
                "No testing parameters defined for `{}` !".format(metafunc))

    if not funcargdict['muts']:
        raise ValueError("Every test must load a set of mutations!")

    mut_dfs = {muts_lbl: load_muts(muts_lbl)
               for muts_lbl in funcargdict['muts']}

    if metafunc.function.__code__.co_argcount == 2:
        if metafunc.function.__code__.co_varnames[1] == 'mtree':
            metafunc.parametrize(
                'mtree',
                [mtree_generator(mut_df) for mut_df in mut_dfs.values()],
                ids=mut_dfs.keys()
                )

        else:
            raise ValueError("Unrecognized singleton argument `{}` !".format(
                metafunc.function.__code__.co_varnames[1]))

    elif metafunc.function.__code__.co_argcount == 3:
        if metafunc.function.__code__.co_varnames[2] == 'mtree':
            metafunc.parametrize(
                'muts, mtree',
                [(mut_df, mtree_generator(mut_df))
                 for mut_df in mut_dfs.values()],
                ids=mut_dfs.keys()
                )

        elif metafunc.function.__code__.co_varnames[2] == 'mut_lvls':
            metafunc.parametrize(
                'mtree, mut_lvls',
                [(mtree_generator(mut_df, mut_lvls), mut_lvls)
                 for mut_df, mut_lvls in product(mut_dfs.values(),
                                                 funcargdict['mut_lvls'])],
                ids=["{} ({})".format(mut_k, '_'.join(mut_lvls))
                     for mut_k, mut_lvls in product(mut_dfs.keys(),
                                                    funcargdict['mut_lvls'])]
                )

        elif metafunc.function.__code__.co_varnames[2] == 'mtypes':
            mtype_list = [mtype_generator(mtype_lbl)
                          for mtype_lbl in funcargdict['mtypes']]

            metafunc.parametrize(
                'mtree, mtypes',
                [(mtree_generator(mut_df), mtypes)
                 for mut_df, (mtypes, _) in product(mut_dfs.values(),
                                                    mtype_list)],
                ids=["{} with <{}> muts".format(mut_k, mtypes_k)
                     for mut_k, (_, mtypes_k) in product(mut_dfs.keys(),
                                                         mtype_list)]
                )

        else:
            raise ValueError

    elif metafunc.function.__code__.co_argcount == 4:
        if 'mtypes' in metafunc.function.__code__.co_varnames:
            mtype_list = [mtype_generator(mtype_lbl)
                          for mtype_lbl in funcargdict['mtypes']]

            metafunc.parametrize(
                'mtree, mut_lvls, mtypes',
                [(mtree_generator(mut_df, mut_lvls), mut_lvls, mtypes)
                 for mut_df, (mut_lvls, (mtypes, _))
                 in product(mut_dfs.values(), zip(funcargdict['mut_lvls'],
                                                  mtype_list))],
                ids=["{} ({}) with <{}> muts".format(mut_k,
                                                     '_'.join(mut_lvls),
                                                     mtypes_k)
                     for mut_k, (mut_lvls, (_, mtypes_k))
                     in product(mut_dfs.keys(), zip(funcargdict['mut_lvls'],
                                                    mtype_list))]
                )

        else:
            metafunc.parametrize(
                'muts, mtree, mut_lvls',
                [(mut_df, mtree_generator(mut_df, mut_lvls), mut_lvls)
                 for mut_df, mut_lvls in product(mut_dfs.values(),
                                                 funcargdict['mut_lvls'])],
                ids=["{} ({})".format(mut_k, '_'.join(mut_lvls))
                     for mut_k, mut_lvls in product(mut_dfs.keys(),
                                                    funcargdict['mut_lvls'])]
                )

    elif metafunc.function.__code__.co_argcount == 5:
        mtype_list = [mtype_generator(mtype_lbl)
                      for mtype_lbl in funcargdict['mtypes']]

        metafunc.parametrize(
            'muts, mtree, mut_lvls, mtypes',
            [(mut_df, mtree_generator(mut_df, mut_lvls), mut_lvls, mtypes)
             for mut_df, (mut_lvls, (mtypes, _))
             in product(mut_dfs.values(), zip(funcargdict['mut_lvls'],
                                              mtype_list))],
            ids=["{} ({}) with <{}> muts".format(mut_k, '_'.join(mut_lvls),
                                                 mtypes_k)
                 for mut_k, (mut_lvls, (_, mtypes_k))
                 in product(mut_dfs.keys(), zip(funcargdict['mut_lvls'],
                                                mtype_list))]
            )

    else:
        raise ValueError


def mtree_generator(mut_df, mut_lvls=None):
    if mut_lvls is None:
        mut_lvls = ('Gene', 'Form')

    if isinstance(mut_lvls, (list, tuple)):
        mtree = MuTree(mut_df, levels=mut_lvls)

    else:
        raise TypeError("Unrecognized mutation tree level identifier "
                        "`{}` !".format(mut_lvls))

    return mtree


class TestCaseInit(object):
    """Tests for basic functionality of MuTrees."""

    params = {
        'muts': ['small', 'medium'],
        'mut_lvls': [('Gene', ), ('Form', ), ('Form', 'Exon'),
                     ('Gene', 'Protein'), ('Gene', 'Form', 'Protein')]
        }

    def test_levels(self, mtree, mut_lvls):
        """Does the tree correctly implement nesting of mutation levels?"""
        assert mtree.get_levels() == set(mut_lvls)

    def test_keys(self, muts, mtree, mut_lvls):
        """Does the tree correctly implement key retrieval of subtrees?"""
        if len(mtree.get_levels()) > 1:
            for vals, _ in muts.groupby(list(mut_lvls)):
                assert mtree._child[vals[0]][vals[1:]] == mtree[vals]
                assert mtree[vals[:-1]]._child[vals[-1]] == mtree[vals]

        else:
            for val in set(muts[mtree.mut_level]):
                assert mtree._child[val] == mtree[val]

    def test_structure(self, muts, mtree, mut_lvls):
        """Is the internal structure of the tree correct?"""
        assert set(mtree._child.keys()) == set(muts[mtree.mut_level])
        assert mtree.depth == 0

        lvl_sets = {i: mut_lvls[:i] for i in range(1, len(mut_lvls))}
        for i, lvl_set in lvl_sets.items():

            for vals, mut in muts.groupby(list(lvl_set)):
                assert mtree[vals].depth == i
                assert (set(mtree[vals]._child.keys())
                        == set(mut[mut_lvls[i]]))

    def test_print(self, muts, mtree, mut_lvls):
        """Can we print the tree?"""
        lvl_sets = [mut_lvls[:i] for i in range(1, len(mut_lvls))]
        for lvl_set in lvl_sets:
            for vals, _ in muts.groupby(list(lvl_set)):
                print(mtree[vals])

        print(mtree)

    def test_iteration(self, muts, mtree, mut_lvls):
        """Does the tree correctly implement iteration over subtrees?"""
        for nm, mut in mtree:
            assert nm in set(muts[mtree.mut_level])
            assert mut == mtree[nm]
            assert mut != mtree

        lvl_sets = {i: mut_lvls[:i] for i in range(1, len(mut_lvls))}
        for i, lvl_set in lvl_sets.items():

            for vals, _ in muts.groupby(list(lvl_set)):
                if isinstance(vals, str):
                    vals = (vals,)

                for nm, mut in mtree[vals]:
                    assert nm in set(muts[mut_lvls[i]])
                    assert mut == mtree[vals][nm]
                    assert mut != mtree[vals[:-1]]

    def test_samples(self, muts, mtree, mut_lvls):
        """Does the tree properly store its samples?"""
        for vals, mut in muts.groupby(list(mut_lvls)):
            assert set(mtree[vals]) == set(mut['Sample'])

    def test_get_samples(self, muts, mtree, mut_lvls):
        """Can we successfully retrieve the samples from the tree?"""
        lvl_sets = [mut_lvls[:i] for i in range(1, len(mut_lvls))]
        for lvl_set in lvl_sets:

            for vals, mut in muts.groupby(list(lvl_set)):
                assert set(mtree[vals].get_samples()) == set(mut['Sample'])

    def test_allkeys(self, muts, mtree, mut_lvls):
        """Can we retrieve the mutation set key of the tree?"""
        lvl_sets = chain.from_iterable(
            combn(mut_lvls, r)
            for r in range(1, len(mut_lvls) + 1))

        for lvl_set in lvl_sets:
            lvl_key = {}

            for vals, _ in muts.groupby(list(lvl_set)):
                cur_key = lvl_key

                if isinstance(vals, (int, str)):
                    vals = (vals,)

                for i in range(len(lvl_set) - 1):
                    if (lvl_set[i], vals[i]) not in cur_key:
                        cur_key.update({(lvl_set[i], vals[i]): {}})
                    cur_key = cur_key[(lvl_set[i], vals[i])]

                cur_key.update({(lvl_set[-1], vals[-1]): None})

            assert mtree.allkey(lvl_set) == lvl_key


class TestCaseMuTypeSamples(object):
    """Tests for using MuTypes to access samples in MuTrees."""

    params = {
        'test_small': {'muts': ['small'],
                       'mut_lvls': [('Gene', 'Form', 'Exon')],
                       'mtypes': ['small']},

        'test_medium': {'muts': ['medium'],
                        'mut_lvls': [('Gene', 'Form', 'Exon'),
                                     ('Gene', 'Form', 'Protein')],
                        'mtypes': ['small']},
        }

    def test_small(self, muts, mtree, mtypes, mut_lvls):
        """Can we use basic MuTypes to get samples in MuTrees?"""
        assert mtypes[0].get_samples(mtree) == set(muts.Sample)

        assert (mtypes[1].get_samples(mtree)
                == set(muts.Sample[muts.Form == 'Missense_Mutation']))
        assert (mtypes[2].get_samples(mtree)
                == set(muts.Sample[muts.Exon == '8/21']))

        assert (mtypes[3].get_samples(mtree)
                == set(muts.Sample[(muts.Form == 'Missense_Mutation')
                                   & ((muts.Exon == '10/21')
                                      | (muts.Exon == '2/21'))]))

        assert (mtypes[4].get_samples(mtree)
                == set(muts.Sample[(muts.Form == 'In_Frame_Del')
                                   | ((muts.Form == 'Missense_Mutation')
                                      & ((muts.Exon == '8/21')
                                         | (muts.Exon == '5/21')))]))

        assert mtypes[5].get_samples(mtree) == set(muts.Sample)
        for mtype in mtypes[6:]:
            assert mtype.get_samples(mtree) == set()

    def test_medium(self, muts, mtree, mtypes, mut_lvls):
        assert (mtypes[1].get_samples(mtree)
                == set(muts.Sample[muts.Form == 'Missense_Mutation']))
        assert (mtypes[2].get_samples(mtree)
                == set(muts.Sample[muts.Exon == '8/21']))
        assert (mtypes[7].get_samples(mtree)
                == set(muts.Sample[muts.Gene.isin(['TP53', 'KRAS', 'BRAF'])]))

        for (gn, frm), mut_data in muts.groupby(['Gene', 'Form']):
            assert (MuType({('Gene', gn): {('Form', frm): None}}).get_samples(
                mtree) == set(mut_data.Sample))

        for (gn1, gn2) in combn(set(muts.Gene), r=2):
            assert (MuType({('Gene', (gn1, gn2)): None}).get_samples(mtree)
                    == set(muts.Sample[muts.Gene.isin([gn1, gn2])]))

        for (frm1, frm2) in combn(set(muts.Form), r=2):
            assert (MuType({('Form', (frm1, frm2)): None}).get_samples(mtree)
                    == set(muts.Sample[muts.Form.isin([frm1, frm2])]))


class TestCaseCustomLevels(object):
    """Tests for custom mutation levels."""

    params = {
        'muts': ['medium', 'big'],
        'mut_lvls': [('Gene', 'Form_base'), ('Gene', 'Form_base', 'Exon'),
                     ('Form_base', 'Protein')]
        }

    def test_base(self, muts, mtree, mut_lvls):
        """Is the _base mutation level parser correctly defined?"""

        assert (
            MuType({('Form_base', 'Frame_Shift'): None}).get_samples(mtree)
            == set(muts.Sample[muts.Form.isin(['Frame_Shift_Ins',
                                               'Frame_Shift_Del'])])
            )

