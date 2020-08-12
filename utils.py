import enum
import random
import time
import typing

import networkx as nx
import numpy as np


class NodeKind(enum.Enum):
    PEEK_ALL = 1
    PEEK_ONE = 2


Seq = typing.List[typing.Optional[int]]
SolvedSeq = typing.List[int]
SimpleGraph = typing.MutableMapping[int, typing.List[typing.Optional[int]]]
TrueGraph = typing.Mapping[int, typing.Tuple[NodeKind, typing.List[int]]]


def seq2str(seq: Seq, aligned: bool=False, align_size: int=2) -> str:
    if aligned:
        fmt = '{: >' + str(align_size) + 'd}'

        def format_element(e):
            if e is None:
                return '_' * align_size
            return fmt.format(e)
    else:
        def format_element(e):
            if e is None:
                return '_'
            return str(e)
    return '[{}]'.format(', '.join([format_element(n) for n in seq]))


def shortest_path(graph: nx.DiGraph, start: int, end: int,
                  avoid: typing.Collection[int]) ->\
        typing.Optional[typing.List[int]]:
    try:
        return nx.shortest_path(nx.restricted_view(graph, avoid, []),
                                start, end)
    except nx.NetworkXNoPath:
        return None


def length_n_path(graph: nx.DiGraph, start: int, end: int,
                  length: int, avoid: typing.Collection[int]) ->\
        typing.Optional[typing.List[int]]:
    o = [[start]]
    path = None
    while o:
        p = o.pop(0)
        if p[-1] == end:
            if len(p) == length:
                path = p
                break
            else:
                continue
        elif len(p) >= length:
            continue
        for n2 in graph.adj.get(p[-1], set()):
            if n2 in avoid or n2 in p:
                continue
            o.append(p + [n2])
    return path
