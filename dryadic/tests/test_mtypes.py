
"""
Unit tests for abstract representations of mutation sub-types.

See Also:
    :class:`..features.mutations.MuType`: The class tested herein.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>
"""

import pytest
from ..features.mutations import MuType
from .resources import mutypes
from ..utils import powerset_slice

from functools import reduce
from operator import or_, and_, add
from itertools import combinations as combn
from itertools import product


def generate_mkeys(lvl_lbls, use_none=False):
    if not lvl_lbls:
        mkeys = [None]

    else:
        cur_lvl, cur_lbls = lvl_lbls[0]
        sub_keys = generate_mkeys(lvl_lbls[1:], True)

        if use_none:
            mkeys = [None]
        else:
            mkeys = [{}]

        for lbl_list in powerset_slice(cur_lbls, start=1):
            mkeys += [
                {(cur_lvl, lbl): k for lbl, k in zip(lbl_list, lbl_keys)}
                for lbl_keys in product(sub_keys, repeat=len(lbl_list))
                ]

    return mkeys


def pytest_generate_tests(metafunc):
    if metafunc.function.__code__.co_argcount == 1:
        pass

    elif metafunc.function.__code__.co_argcount == 2:
        if hasattr(metafunc.cls, 'params'):
            if isinstance(metafunc.cls.params, dict):
                funcarg = metafunc.cls.params[metafunc.function.__name__]

            else:
                funcarg = metafunc.cls.params

        else:
            funcarg = 'ALL'

        if metafunc.function.__code__.co_varnames[1] == 'mtypes':
            mtypes, id_str = mtype_generator(funcarg)
            metafunc.parametrize('mtypes', [mtypes], ids=[id_str])

        else:
            raise ValueError("Unrecognized singleton argument "
                             "to unit test `{}` !".format(
                                 metafunc.function.__code__.co_varnames[1]))

    else:
        raise ValueError("MuType unit tests take at most one argument!")


def mtype_generator(mtypes_param):
    if isinstance(mtypes_param, str):
        if mtypes_param == 'ALL':
            mtypes = reduce(add, [tps for _, tps in vars(mutypes).items()
                                  if isinstance(tps, tuple)])
            id_str = 'all labeled sets'

        elif '_' in mtypes_param:
            mtypes = reduce(add, [eval('mutypes.{}'.format(mtypes))
                                  for mtypes in mtypes_param.split('_')])
            id_str = mtypes_param.replace('_', '+')

        else:
            mtypes = eval('mutypes.{}'.format(mtypes_param))
            id_str = mtypes_param

    elif isinstance(mtypes_param, list):
        mtypes = [MuType(mkey) for mkey in generate_mkeys(mtypes_param)]

        lvls_list, lbls_list = zip(*mtypes_param)
        id_str = ' '.join(['x'.join([str(len(lbls)) for lbls in lbls_list]),
                           "({})".format(','.join(lvls_list))])

    else:
        raise TypeError("Unrecognized mutation type identifier "
                        "`{}` !".format(mtypes_param))

    return mtypes, id_str


class TestCaseInit(object):
    """Tests for proper instatiation of MuTypes from type dictionaries."""

    def test_child(self):
        """Is the child attribute of a MuType properly created?"""

        assert (MuType({('Gene', 'TP53'): None})._child
                == {frozenset(['TP53']): None})
        assert (MuType({('Gene', 'TP53'): {('Form', 'Frame'): None}})._child
                == {frozenset(['TP53']): MuType({('Form', 'Frame'): None})})

        assert (MuType({('Gene', ('TP53', )): None})._child
                == {frozenset(['TP53']): None})
        assert (MuType({('Gene', ('TP53', 'KRAS')): None})._child
                == {frozenset(['TP53', 'KRAS']): None})

        assert (MuType({('Gene', 'TP53'): {('Form', 'Point'): None},
                        ('Gene', 'KRAS'): {('Form', 'Frame'): None}})._child
                == {frozenset(['TP53']): MuType({('Form', 'Point'): None}),
                    frozenset(['KRAS']): MuType({('Form', 'Frame'): None})})

        assert (MuType({('Gene', 'TP53'): {('Form', 'InDel'): None},
                        ('Gene', 'KRAS'): {('Form', 'InDel'): None}})._child
                == {frozenset(['TP53', 'KRAS']): MuType({
                    ('Form', 'InDel'): None})})

    def test_empty(self):
        """Can we correctly instantiate an empty MuType?"""

        assert MuType(None).is_empty()
        assert MuType({}).is_empty()
        assert MuType([]).is_empty()
        assert MuType(()).is_empty()

    def test_levels(self):
        """Do mutations store mutation annotation levels correctly?"""

        for gene_lbls in powerset_slice(['TP53', 'KRAS', 'BRAF'], start=1):
            mtype = MuType({('Gene', gene_lbls): None})
            assert mtype.cur_level == 'Gene'
            assert mtype.get_levels() == {'Gene'}

            for form_lbls in powerset_slice(['missense', 'nonsense'],
                                            start=1):
                mtype = MuType({('Gene', gene_lbls): {(
                    'Form', form_lbls): None}})

                assert mtype.cur_level == 'Gene'
                assert mtype.get_levels() == {'Gene', 'Form'}

    def test_synonyms(self):
        assert (MuType({('Gene', ('TP53', 'KRAS')): None})._child
                == MuType({('Gene', 'TP53'): None,
                           ('Gene', 'KRAS'): None})._child)

        assert (MuType({('Gene', 'TP53'): {('Form', 'Frame'): None},
                        ('Gene', 'KRAS'): {('Form', 'Frame'): None}})._child
                == MuType({('Gene', ('TP53', 'KRAS')): {
                    ('Form', 'Frame'): None}})._child)

        assert (MuType({('Gene', 'TP53'): {('Form',
                                            ('Point', 'Frame')): None},
                        ('Gene', 'KRAS'): {
                            ('Form', ('Frame', 'Point')): None}})._child
                == MuType({('Gene', 'TP53'): {('Form', 'Frame'): None},
                           ('Gene', 'KRAS'): {('Form', 'Frame'): None},
                           ('Gene', ('TP53', 'KRAS')): {
                               ('Form', 'Point'): None}})._child)

        assert (MuType({('Gene', ('TP53', 'KRAS', 'BRAF')): None})._child
                == MuType({('Gene', ('BRAF', 'TP53', 'KRAS')): None})._child)


class TestCaseBasic:
    """Tests for basic functionality of MuTypes."""

    params = [('Gene', ('KRAS', 'TP53')),
              ('Exon', ('7th', )),
              ('Form', ('missense', 'nonsense', 'frameshift'))]

    def test_hash(self, mtypes):
        """Can we get proper hash values of MuTypes?"""
        for mtype1, mtype2 in combn(mtypes, 2):
            if mtype1 == mtype2:
                assert hash(mtype1) == hash(mtype2)

    def test_equality(self, mtypes):
        for mtype1, mtype2 in combn(mtypes, 2):
            assert (mtype1._child == mtype2._child) == (mtype1 == mtype2)

    def test_print(self, mtypes):
        """Can we print MuTypes?"""
        for mtype in mtypes:
            assert isinstance(repr(mtype), str)
            assert isinstance(str(mtype), str)

        for mtype1, mtype2 in combn(mtypes, 2):
            if mtype1 == mtype2:
                assert str(mtype1) == str(mtype2)
                assert repr(mtype1) == repr(mtype2)

            else:
                assert repr(mtype1) != repr(mtype2)

    def test_subkeys(self, mtypes):
        """Can we get the leaf types stored in a MuType?"""
        for mtype in mtypes:
            key_mtypes = [MuType(k) for k in mtype.subkeys()]

            assert len(set(key_mtypes)) == len(key_mtypes)
            assert reduce(or_, key_mtypes, MuType({})) == mtype
            assert (sum(len(key_mtype.subkeys()) for key_mtype in key_mtypes)
                    == len(mtype.subkeys()))

            if len(key_mtypes) > 1:
                assert reduce(and_, key_mtypes).is_empty()

        for mtype1, mtype2 in combn(mtypes, 2):
            assert ((sorted(MuType(k) for k in mtype1.subkeys())
                     == sorted(MuType(k) for k in mtype2.subkeys()))
                    == (mtype1 == mtype2))

    def test_state(self, mtypes):
        for mtype in mtypes:
            assert mtype == MuType(mtype.__getstate__())


class TestCaseIter:
    """Tests for iterating over the elements of MuTypes."""

    params = {
        'test_iter': [('Gene', ('TP53', )),
                      ('Exon', ('3rd', '7th', '2nd', '4th', '1st')),
                      ('Form', ('missense', 'nonsense', 'frameshift'))],
        'test_len': 'small'
        }

    def test_iter(self, mtypes):
        """Can we iterate over the sub-types in a MuType?"""
        for mtype in mtypes:
            assert (len(mtype.subkeys()) >= len(tuple(mtype.subtype_iter()))
                    >= len(tuple(mtype.child_iter())))

    def test_len(self, mtypes):
        assert ([len(mtype) for mtype in mtypes]
                == [1, 1, 1, 1, 2, 2, 1, 3, 1, 1])


class TestCaseSorting:
    """Tests the sort order defined for MuTypes."""

    params = {'test_sort': 'sorting',
              'test_sort_invariance': 'ALL'}

    def test_sort(self, mtypes):
        """Can we correctly sort a list of MuTypes?"""
        assert sorted(mtypes) == [
            mtypes[5], mtypes[6], mtypes[7], mtypes[2], mtypes[1], mtypes[10],
            mtypes[3], mtypes[9], mtypes[11], mtypes[0], mtypes[4], mtypes[8]
            ]

        assert sorted(mtypes[0:11:2]) == [
            mtypes[6], mtypes[2], mtypes[10], mtypes[0], mtypes[4], mtypes[8]
            ]

        assert sorted(mtypes[:6]) == [
            mtypes[5], mtypes[2], mtypes[1], mtypes[3], mtypes[0], mtypes[4],
            ]

    def test_sort_invariance(self, mtypes):
        """Is the sort order for MuTypes consistent?"""
        for mtype1, mtype2, mtype3 in combn(mtypes, 3):
            assert not (mtype1 < mtype2 < mtype3 < mtype1)

        assert sorted(mtypes) == sorted(list(reversed(mtypes)))


class TestCaseBinary:
    """Tests the binary operators defined for MuTypes."""

    params = {
        'test_comparison': [('Gene', ('TP53', 'PIK3CA')),
                            ('Exon', ('3rd', '6th')),
                            ('Form', ('nonsense', ))],
        'test_invariants': [('Gene', ('TP53', 'PIK3CA')),
                            ('Exon', ('3rd', )),
                            ('Form', ('nonsense', 'frameshift', 'missense'))],
        'test_or_easy': 'small',
        'test_or_hard': 'binary',
        'test_and': 'small',
        'test_sub': 'binary'
        }

    def test_comparison(self, mtypes):
        """Are rich comparison operators correctly implemented for MuTypes?"""
        for mtype in mtypes:
            assert mtype == mtype

            assert mtype <= mtype
            assert mtype >= mtype
            assert not mtype < mtype
            assert not mtype > mtype

        for mtype1, mtype2 in combn(mtypes, 2):
            assert (mtype1 == mtype2) == (mtype2 == mtype1)
            assert (mtype1 <= mtype2) != (mtype1 > mtype2)

            if mtype1 < mtype2:
                assert mtype1 <= mtype2
                assert mtype1 != mtype2

            elif mtype1 > mtype2:
                assert mtype1 >= mtype2
                assert mtype1 != mtype2

    def test_invariants(self, mtypes):
        """Do binary operators preserve set theoretic invariants?"""
        for mtype in mtypes:
            assert mtype == (mtype & mtype)
            assert mtype == (mtype | mtype)
            assert (mtype - mtype).is_empty()

        for mtype1, mtype2 in combn(mtypes, 2):
            if mtype1.get_levels() == mtype2.get_levels():
                assert mtype1 | mtype2 == mtype2 | mtype1
                assert mtype1 & mtype2 == mtype2 & mtype1

                if mtype1 == mtype2:
                    assert (mtype1 - mtype2).is_empty()
                    assert (mtype2 - mtype1).is_empty()

                assert (mtype1 | mtype2).is_supertype(mtype1 & mtype2)
                assert mtype1 - mtype2 == mtype1 - (mtype1 & mtype2)
                assert mtype1 | mtype2 == (
                    (mtype1 - mtype2) | (mtype2 - mtype1)
                    | (mtype1 & mtype2)
                    )

            if mtype1.get_levels() <= mtype2.get_levels():
                if mtype1 == mtype2 or mtype1.is_supertype(mtype2):
                    assert mtype2 == (mtype1 & mtype2)

            if mtype1.get_levels() >= mtype2.get_levels():
                if mtype1 == mtype2 or mtype2.is_supertype(mtype1):
                    assert mtype2 == (mtype1 | mtype2)

    def test_or_easy(self, mtypes):
        """Can we take the union of two simple MuTypes?"""
        assert (mtypes[6] | mtypes[7]) == mtypes[7]
        assert (mtypes[0] | mtypes[5]) == mtypes[5]

        assert (mtypes[8] | mtypes[4]) == MuType({
            ('Form', ('Frame_Shift', 'In_Frame_Del')): None,
            ('Form', 'Missense_Mutation'): {
                ('Exon', ('8/21', '5/21')): None}
            })

    def test_or_hard(self, mtypes):
        """Can we take the union of two tricky MuTypes?"""
        assert (mtypes[0] | mtypes[1]) == mtypes[2]
        assert (mtypes[0] & mtypes[1]) == mtypes[3]

    def test_and(self, mtypes):
        """Can we take the intersection of two MuTypes?"""
        assert (mtypes[0] & mtypes[3]) == mtypes[3]
        assert (mtypes[2] & mtypes[9]) == MuType({})
        assert (mtypes[6] & mtypes[7]) == mtypes[6]

        assert ((mtypes[4] & MuType({('Form', 'Missense_Mutation'): None}))
                == MuType({('Form', 'Missense_Mutation'): {
                    ('Exon', ('8/21', '5/21')): None}}))

    def test_sub(self, mtypes):
        """Can we subtract one MuType from another?"""
        sub_mtype = MuType({
            ('Gene', 'TP53'): {('Form', 'Missense_Mutation'): None}})
        assert (mtypes[2] - mtypes[0]) == sub_mtype

