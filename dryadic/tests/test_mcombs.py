
"""Unit tests for abstract representations of mutation sub-types.

See Also:
    :class:`..features.mutations.MutComb`: The class tested herein.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from ..features.mutations import MuType, MutComb
from .utilities import pytest_generate_tests
import pytest

from itertools import combinations as combn
from itertools import product


class TestCaseInit(object):
    """Tests for proper instatiation of MutCombs from lists of MuTypes."""

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

