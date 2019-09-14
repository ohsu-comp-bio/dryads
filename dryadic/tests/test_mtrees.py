
"""Unit tests for hierarchical storage of mutation cohorts.

See Also:
    :class:`..features.mutations.MuTree`: The class tested herein.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from ..features.mutations import MuType, MuTree
from .test_mtypes import mtype_tester
import pytest

import os
import pandas as pd
from itertools import product, chain
from itertools import combinations as combn


def load_muts(muts_lbl):
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'resources',
                     "muts_{}.tsv".format(muts_lbl)),
        engine='python', sep='\t', comment='#',
        names=['Gene', 'Form', 'Sample', 'Protein', 'Transcript', 'Exon',
               'ref_count', 'alt_count', 'PolyPhen']
        )


def pytest_generate_tests(metafunc):
    if metafunc.function.__code__.co_argcount == 1:
        pass

    if metafunc.function.__code__.co_argcount >= 2:
        if hasattr(metafunc.cls, 'params'):
            if isinstance(metafunc.cls.params, dict):
                funcarglist = metafunc.cls.params[metafunc.function.__name__]

            else:
                funcarglist = metafunc.cls.params

        else:
            funcarglist = 'ALL'

        if isinstance(funcarglist, str):
            funcarglist = [funcarglist]
    
    if metafunc.function.__code__.co_argcount == 2:
        if metafunc.function.__code__.co_varnames[1] == 'mtree':
            metafunc.parametrize(
                'mtree', [mtree_tester(funcarg) for funcarg in funcarglist],
                ids=[funcarg.replace('_', '+') for funcarg in funcarglist]
                )

        else:
            raise ValueError

    elif metafunc.function.__code__.co_argcount == 3:
        if metafunc.function.__code__.co_varnames[2] == 'mtree':
            muts = {lbl: load_muts(lbl) for lbl in funcarglist}

            metafunc.parametrize('muts, mtree',
                                 [(muts[funcarg], mtree_tester(muts[funcarg]))
                                  for funcarg in funcarglist],
                                 ids=funcarglist)

        elif metafunc.function.__code__.co_varnames[2] == 'mtypes':
            metafunc.parametrize(
                'mtree, mtypes',
                [(mtype_tester(funcarg1), mtree_tester(funcarg2))
                 for funcarg1, funcarg2 in product(*funcarglist)],
                ids=['x'.join([funcarg1, funcarg2])
                     for funcarg1, funcarg2 in product(*funcarglist)]
                )

        elif metafunc.function.__code__.co_varnames[2] == 'mut_lvls':
            if len(funcarglist) == 2 and len(funcarglist[0]) > 1:
                metafunc.parametrize(
                    'mtree, mut_lvls',
                    [(mtree_tester(mut_str, mut_lvls), mut_lvls)
                     for mut_str, mut_lvls in product(*funcarglist)],
                    ids=["{}<{}>".format(mut_str, mut_lvls)
                         for mut_str, mut_lvls in product(*funcarglist)]
                    )

            else:
                metafunc.parametrize(
                    'mtree, mut_lvls',
                    [(mtree_tester(mut_str, mut_lvls), mut_lvls)
                     for mut_str, mut_lvls in funcarglist],
                    ids=["{}: <{}>".format(mut_str, mut_lvls)
                         for mut_str, mut_lvls in funcarglist]
                    )

        else:
            raise ValueErorr

    elif metafunc.function.__code__.co_argcount == 4:
        if metafunc.function.__code__.co_varnames[3] == 'mut_lvls':
            if len(funcarglist) == 2 and len(funcarglist[0]) > 1:
                muts = {mut_str: load_muts(mut_str)
                        for mut_str in funcarglist[0]}

                metafunc.parametrize(
                    'muts, mtree, mut_lvls',
                    [(muts[mut_str],
                      mtree_tester(muts[mut_str], mut_lvls), mut_lvls)
                     for mut_str, mut_lvls in product(*funcarglist)],
                    ids=["{}: <{}>".format(mut_str, mut_lvls)
                         for mut_str, mut_lvls in product(*funcarglist)]
                    )

            else:
                muts = {mut_str: load_muts(mut_str)
                        for mut_str, _ in funcarglist}

                metafunc.parametrize(
                    'muts, mtree, mut_lvls',
                    [(muts[mut_str],
                      mtree_tester(muts[mut_str], mut_lvls), mut_lvls)
                     for mut_str, mut_lvls in funcarglist],
                    ids=["{}: <{}>".format(mut_str, mut_lvls)
                         for mut_str, mut_lvls in funcarglist]
                    )

    elif metafunc.function.__code__.co_argcount == 5:
        muts = {mut_str: load_muts(mut_str) for mut_str, _ in funcarglist[0]}

        metafunc.parametrize(
            'muts, mtree, mtypes, mut_lvls',
            [(muts[mut_str], mtree_tester(muts[mut_str], levels=mut_lvls),
              mtype_tester(mtype_str), mut_lvls)
             for (mut_str, mtype_str), mut_lvls in product(*funcarglist)],
            ids=["{} & {} : <{}>".format(mut_str, mtype_str, mut_lvls)
                 for (mut_str, mtype_str), mut_lvls in product(*funcarglist)]
            )

    else:
        raise ValueError


def mtree_tester(muts_param, levels=('Gene', 'Form')):
    if isinstance(muts_param, str):
        mtree = MuTree(load_muts(muts_param), levels=levels)
    elif hasattr(muts_param, 'shape'):
        mtree = MuTree(muts_param, levels=levels)

    else:
        raise ValueError("Unrecognized type of mutation tree "
                         "parameter `{}` !".format(muts_param))

    return mtree


class TestCaseInit(object):
    """Tests for basic functionality of MuTrees."""

    params = [
        ['small', 'medium'],
        [('Gene', ), ('Form', ), ('Form', 'Exon'), ('Gene', 'Protein'),
         ('Gene', 'Form', 'Protein')]
        ]

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
        'test_small': [
            [('small', 'small'), ],
            [('Gene', 'Form', 'Exon'), ]
            ],
        'test_medium': [
            [('medium', 'small'), ],
            [('Gene', 'Form', 'Exon'), ('Gene', 'Form', 'Protein')]
            ],
        'test_status': [
            [('small', 'small'), ('medium', 'small'), ],
            [('Gene', 'Form', 'Exon'), ('Gene', 'Form', 'Protein')]
            ],
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

    def test_status(self, muts, mtree, mtypes, mut_lvls):
        """Can we get a vector of mutation status from a MuTree?"""
        for mtype in mtypes:
            assert (mtree.status(['herpderp', 'derpherp'], mtype)
                    == [False, False])

        samp_lists = [pd.Series(tuple(set(muts.Sample))),
                      pd.Series(tuple(set(muts.Sample[::-2]))),
                      pd.Series(tuple(set(muts.Sample[2:7]))),]

        for samp_list in samp_lists:
            for mtype in mtypes:
                assert (mtree.status(samp_list, mtype)
                        == samp_list.isin(mtype.get_samples(mtree))).all()


class TestCaseCustomLevels(object):
    """Tests for custom mutation levels."""

    params = {
        'test_base': [
            ['medium', 'big'],
            [('Gene', 'Form_base'), ('Gene', 'Form_base', 'Exon'),
             ('Form_base', 'Protein')]
            ],
        }

    def test_base(self, muts, mtree, mut_lvls):
        """Is the _base mutation level parser correctly defined?"""

        assert (
            MuType({('Form_base', 'Frame_Shift'): None}).get_samples(mtree)
            == set(muts.Sample[muts.Form.isin(['Frame_Shift_Ins',
                                               'Frame_Shift_Del'])])
            )

