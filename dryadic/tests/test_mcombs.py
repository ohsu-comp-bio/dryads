
"""Unit tests for abstract representations of mutation combinations.

See Also:
    :class:`..features.mutations.MutComb`: The class tested herein.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from ..features.mutations import MuType, MutComb
from .resources import mutcombs
from .test_mtypes import mtype_tester
import pytest

from functools import reduce
from operator import add
from itertools import combinations as combn
from itertools import product


def pytest_generate_tests(metafunc):
    if metafunc.function.__code__.co_argcount == 1:
        pass

    if metafunc.function.__code__.co_argcount == 2:
        if hasattr(metafunc.cls, 'params'):
            if isinstance(metafunc.cls.params, dict):
                funcarglist = metafunc.cls.params[metafunc.function.__name__]

            else:
                funcarglist = metafunc.cls.params

        else:
            funcarglist = 'ALL'

        if isinstance(funcarglist, str):
            funcarglist = [funcarglist]

        if metafunc.function.__code__.co_varnames[1] == 'mtypes':
            metafunc.parametrize(
                'mtypes', [mtype_tester(funcarg) for funcarg in funcarglist],
                ids=[funcarg.replace('_', '+') for funcarg in funcarglist]
                )

        elif metafunc.function.__code__.co_varnames[1] == 'mcombs':
            metafunc.parametrize(
                'mcombs', [mcomb_tester(funcarg) for funcarg in funcarglist],
                ids=[funcarg.replace('_', '+') for funcarg in funcarglist]
                )

        else:
            raise ValueError("Unrecognized singleton argument "
                             "to unit test `{}` !".format(
                                 metafunc.function.__code__.co_varnames[1]))

    else:
        raise ValueError("MutComb unit tests take at most one argument!")


def mcomb_tester(mcombs_param):
    if mcombs_param == 'ALL':
        mcombs = reduce(
            add, [tps for _, tps in vars(mutcombs).items()
                  if isinstance(tps, tuple) and isinstance(tps[0], MutComb)]
            )

    elif '_' in mcombs_param:
        mcombs = reduce(add, [eval('mutcombs.{}'.format(mcombs))
                              for mcombs in mcombs_param.split('_')])

    else:
        mcombs = eval('mutcombs.{}'.format(mcombs_param))

    return mcombs


class TestCaseInit(object):
    """Tests for proper instatiation of MutCombs from lists of MuTypes."""

    params = ['basic_synonyms_binary']

    def test_attr(self, mtypes):
        for mtype1, mtype2 in product(mtypes, repeat=2):
            if (mtype1 & mtype2).is_empty():
                if not mtype1.is_empty() and not mtype2.is_empty():
                    test_mcomb = MutComb(mtype1, mtype2)
                    assert test_mcomb.mtypes == frozenset([mtype1, mtype2])
                    assert test_mcomb.not_mtype == None

                for mtype3 in mtypes:
                    if ((mtype1 & mtype3).is_empty()
                            and (mtype2 & mtype3).is_empty()):
                        test_mcomb = MutComb(mtype1, mtype2, not_mtype=mtype3)
                        assert test_mcomb.mtypes == frozenset(
                            [mtype1, mtype2])
                        assert test_mcomb.not_mtype == mtype3

    def test_order(self, mtypes):
        for mtype1, mtype2 in combn(mtypes, r=2):
            assert MutComb(mtype1, mtype2) == MutComb(mtype2, mtype1)

            for mtype3 in mtypes:
                assert (MutComb(mtype1, mtype2, not_mtype=mtype3)
                        == MutComb(mtype2, mtype1, not_mtype=mtype3))

    def test_overlap(self, mtypes):
        for mtype1, mtype2 in product(set(mtypes) | {MuType({})}, repeat=2):
            if mtype1.is_supertype(mtype2):
                assert MutComb(mtype1, mtype2) == mtype2

                if mtype1.get_levels() == mtype2.get_levels():
                    assert (MutComb(mtype1, not_mtype=mtype2)
                            == mtype1 - mtype2)


class TestCaseBasic(object):

    params = 'basic'

    def test_hash(self, mcombs):
        for mcomb1, mcomb2 in product(mcombs, repeat=2):
            assert (mcomb1 == mcomb2) == (hash(mcomb1) == hash(mcomb2))

    def test_print(self, mcombs):
        for mcomb1, mcomb2 in combn(mcombs, 2):
            if mcomb1 == mcomb2:
                assert repr(mcomb1) == repr(mcomb2)
                assert str(mcomb1) == str(mcomb2)

            else:
                assert repr(mcomb1) != repr(mcomb2)

