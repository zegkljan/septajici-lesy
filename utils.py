import enum
import typing

import networkx


class NodeKind(enum.Enum):
    PEEK_ALL = 1
    PEEK_ONE = 2


Seq = typing.List[typing.Optional[int]]
SolvedSeq = typing.List[int]
SimpleGraph = typing.MutableMapping[int, typing.List[typing.Optional[int]]]
TrueGraph = typing.Mapping[int, typing.Tuple[NodeKind, typing.List[int]]]


def shortest_path(graph: SimpleGraph, start: int, end: int) ->\
        typing.Optional[typing.List[int]]:
    o = [start]
    c = set()
    rec = dict()
    while o:
        n = o.pop(0)
        if n in c:
            continue
        c.add(n)
        if n == end:
            break
        for n2 in graph.get(n, []):
            if n2 is None:
                continue
            o.append(n2)
            if n2 not in rec:
                rec[n2] = n
        if not o:
            return None
    path = [end]
    while True:
        n = rec[path[-1]]
        path.append(n)
        if n == start:
            break
    path.reverse()
    return path


def length_n_path(graph: SimpleGraph, start: int, end: int, length: int) ->\
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
        for n2 in graph.get(p[-1], []):
            if n2 is None:
                continue
            o.append(p + [n2])
    return path


def tg2digraph(tg: TrueGraph) -> networkx.DiGraph:
    g = networkx.DiGraph()
    for n, (k, _) in tg.items():
        g.add_node(n, kind=k)
    for n, (_, nbs) in tg.items():
        g.add_edges_from([(n, nb) for nb in nbs])
    return g


def sg2digraph(sg: SimpleGraph) -> networkx.DiGraph:
    g = networkx.DiGraph()
    for n, nbs in sg.items():
        g.add_node(n)
        for i, nb in enumerate(nbs):
            if nb is not None:
                g.add_edge(n, nb, order=i)
    return g


def all_paths(tg: TrueGraph, start: int, end: int) -> typing.Generator:
    g = tg2digraph(tg)
    return networkx.shortest_simple_paths(g, start, end)
