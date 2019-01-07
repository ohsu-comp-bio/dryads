
from .resources import mutypes 
from .resources import mutcombs
from ..features.mutations import MuType, MutComb

from functools import reduce
from operator import add


def mtype_tester(mtypes_param):
    if mtypes_param == 'ALL':
        mtypes = reduce(
            add, [tps for _, tps in vars(mutypes).items()
                  if isinstance(tps, tuple) and isinstance(tps[0], MuType)]
            )

    elif '_' in mtypes_param:
        mtypes = reduce(add, [eval('mutypes.{}'.format(mtypes))
                              for mtypes in mtypes_param.split('_')])

    else:
        mtypes = eval('mutypes.{}'.format(mtypes_param))

    return mtypes


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


def pytest_generate_tests(metafunc):

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

