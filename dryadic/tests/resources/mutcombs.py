
"""Pre-defined MutCombs used for testing.

"""

from ...features.mutations import MuType, MutComb

mtype1 = MuType({('Gene', 'TP53'): None})
mtype2 = MuType({('Gene', 'KRAS'): None})
mtype3 = MuType({('Gene', 'PIK3CA'): None})


basic = (
    MutComb(mtype1, mtype2),
    MutComb(mtype1, mtype3),
    MutComb(mtype2, mtype3),
    MutComb(mtype1, mtype2, mtype3),
    MutComb(mtype1 | mtype2, mtype3),
    MutComb(mtype1, not_mtype=mtype2),
    )

