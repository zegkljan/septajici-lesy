import enum
import typing

import networkx as nx


class NodeKind(enum.Enum):
    PEEK_ALL = 1
    PEEK_ONE = 2


Seq = typing.List[typing.Optional[int]]
SolvedSeq = typing.List[int]
SimpleGraph = typing.MutableMapping[int, typing.List[typing.Optional[int]]]
TrueGraph = typing.Mapping[int, typing.Tuple[NodeKind, typing.List[int]]]


def seq2str(seq: Seq) -> str:
    return '[{}]'.format(', '.join([str(n) if n is not None else '?'
                                    for n in seq]))


def shortest_path(graph: nx.DiGraph, start: int, end: int) ->\
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
        for n2 in graph.adj.get(n, set()):
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


def length_n_path(graph: nx.DiGraph, start: int, end: int,
                  length: int) -> typing.Optional[typing.List[int]]:
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
            if n2 is None:
                continue
            o.append(p + [n2])
    return path


def tg2digraph(tg: TrueGraph) -> nx.DiGraph:
    g = nx.DiGraph()
    for n, (k, _) in tg.items():
        g.add_node(n, kind=k)
    for n, (_, nbs) in tg.items():
        g.add_edges_from([(n, nb, {'order': i}) for i, nb in enumerate(nbs)])
    return g


def all_paths(tg: TrueGraph, start: int, end: int) -> typing.Generator:
    g = tg2digraph(tg)
    return nx.shortest_simple_paths(g, start, end)
