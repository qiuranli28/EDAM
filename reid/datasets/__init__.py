from __future__ import absolute_import
from .market import Market
from .cuhk import CUHK
from .msmt import MSMT
from .randperson import RandPerson
from .clonedperson import ClonedPerson
from .prcc import PRCC
from .ltcc import LTCC
from .vcclothes import VCClothes
from .deepchange import DeepChange
from .last import LaST

__factory = {
    'market': Market,
    'cuhk03_np_labeled': CUHK,
    'cuhk03_np_detected': CUHK,
    'msmt': MSMT,
    'randperson': RandPerson,
    'clonedperson': ClonedPerson,
    'prcc':PRCC,
    'ltcc':LTCC,
    'vcclothes':VCClothes,
    'deepchange':DeepChange,
    'last':LaST
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'market', 'duke'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
