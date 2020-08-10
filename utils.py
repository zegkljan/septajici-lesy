import collections
import enum
import functools
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
T = typing.TypeVar('T')
U = typing.TypeVar('U')


class PathSetMetrics(typing.NamedTuple):
    nones_count: int
    pairwise_dists: int
    min_none_sequence_len: int
    nones_spread: float


def seq2str(seq: Seq) -> str:
    return '[{}]'.format(', '.join([str(n) if n is not None else '?'
                                    for n in seq]))


def partition(c: typing.Collection[T], key: typing.Callable[[T], U]) ->\
        typing.Mapping[U, typing.List[T]]:
    partitions = dict()
    for e in c:
        k = key(e)
        if k not in partitions:
            partitions[k] = []
        partitions[k].append(e)
    return partitions


def seq_distance(seq1: Seq, seq2: Seq) -> int:
    """
    iterative_levenshtein(s, t) -> ldist
    ldist is the Levenshtein distance between the strings
    s and t.
    For all i and j, dist[i,j] will contain the Levenshtein
    distance between the first i characters of s and the
    first j characters of t
    """

    rows = len(seq1) + 1
    cols = len(seq2) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    # source prefixes can be transformed into empty strings
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i

    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            s1r = seq1[row - 1]
            s2c = seq2[col - 1]
            if s1r == s2c and s1r is not None and s2c is not None:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                 dist[row][col - 1] + 1,  # insertion
                                 dist[row - 1][col - 1] + cost)  # substitution

    return dist[-1][-1]


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


def generate_graph(size: int, seed: typing.Optional[int] = None) -> nx.DiGraph:
    if seed is None:
        seed = int(time.time())
    random.seed(seed)

    # generate random graph
    g = nx.fast_gnp_random_graph(size, 2.5 / (size - 1),
                                 seed=seed, directed=True)

    # postprocess graph

    # remove self loops
    for n in g:
        if g.has_edge(n, n):
            g.remove_edge(n, n)

    # strongly connect all components
    comps = [list(c) for c in nx.strongly_connected_components(g)]
    nc = len(comps)
    while nc > 1:
        c1 = random.randrange(len(comps))
        c2 = random.randrange(len(comps))
        while c2 == c1:
            c2 = random.randrange(len(comps))
        c1n1 = random.choice(comps[c1])
        c1n2 = random.choice(comps[c1])
        while len(comps[c1]) > 1 and c1n1 == c1n2:
            c1n2 = random.choice(comps[c1])
        c2n1 = random.choice(comps[c2])
        c2n2 = random.choice(comps[c2])
        while len(comps[c2]) > 1 and c2n1 == c2n2:
            c2n2 = random.choice(comps[c2])
        g.add_edge(c1n1, c2n1)
        g.add_edge(c2n2, c1n2)
        comps = [list(c) for c in nx.strongly_connected_components(g)]
        nc = len(comps)

    # add meta info - node types, edge orderings
    for n in g:
        g.add_node(n, kind=NodeKind.PEEK_ALL)
        for i, nn in enumerate(g[n]):
            g.edges[(n, nn)]['order'] = i
    return g


def blank_seqs(seqs: typing.Collection[SolvedSeq]) ->\
        typing.Sequence[typing.Tuple[Seq, SolvedSeq]]:
    assert min(seqs, key=len) == max(seqs, key=len)
    arr = np.array(seqs)
    partitions = [np.arange(arr.shape[0])]
    for i in range(len(seqs.__iter__().__next__()) - 1):
        point = arr[:, i]
        partitions2 = []
        for p in partitions:
            unq = np.unique(point[p])
            if len(unq) == 1:
                if i > 0:
                    point[p] = -1
                partitions2.append(p)
            else:
                for u in unq:
                    partitions2.append(p[np.where(point[p] == u)])
        partitions = partitions2
    ret = []
    for i, orig in zip(range(arr.shape[0]), seqs):
        p = []
        for j in arr[i, :]:
            if j == -1:
                p.append(None)
            else:
                p.append(j)
        ret.append((p, orig))
    return ret


def select_paths(all_paths: typing.List[typing.Tuple[Seq, SolvedSeq]],
                 no: int) -> typing.List[typing.Tuple[Seq, SolvedSeq]]:
    return sorted(all_paths, key=lambda x: x[0].count(None))[-no:]


def compute_metrics(seq_set: typing.List[typing.Tuple[Seq, SolvedSeq]]) ->\
        PathSetMetrics:
    nones_count = sum([seq.count(None) for seq, _ in seq_set])
    pairwise_dists = sum([seq_distance(a, b)
                          for _, a in seq_set for _, b in seq_set])
    min_none_sequence_len = max(map(lambda x: len(x[0]), seq_set))
    for s, _ in seq_set:
        cnt = 0
        for e in s:
            if e is not None:
                if cnt > 0:
                    min_none_sequence_len = min(min_none_sequence_len, cnt)
                cnt = 0
            else:
                cnt += 1
    nones_spread = 0
    for s, _ in seq_set:
        l = len(s)
        positions = []
        for i, e in enumerate(s):
            if e is None:
                positions.append(i / l)
        nones_spread += np.var(positions)
    return PathSetMetrics(nones_count=nones_count,
                          pairwise_dists=pairwise_dists,
                          min_none_sequence_len=min_none_sequence_len,
                          nones_spread=nones_spread)


def nd_sort(path_sets: typing.Iterable[
        typing.List[typing.Tuple[Seq, SolvedSeq]]]) ->\
        typing.List[
            typing.List[
                typing.Tuple[
                    typing.List[
                        typing.Tuple[Seq, SolvedSeq]
                    ],
                    PathSetMetrics
                ]
            ]
        ]:
    path_sets_keys = [(s, compute_metrics(s)) for s in path_sets]
    fronts = [[]]
    path_sets_keys.sort(key=lambda x: (x[1].nones_spread, x[1].pairwise_dists),
                        reverse=True)
    for p, pk in path_sets_keys:
        x = len(fronts)
        k = 0
        while True:
            dominated = False
            for pf, pfk in reversed(fronts[k]):
                if (pfk.nones_spread >= pk.nones_spread and
                        pfk.pairwise_dists >= pk.pairwise_dists):
                    dominated = True
                    break
            if not dominated:
                fronts[k].append((p, pk))
                break
            else:
                k += 1
                if k >= x:
                    fronts.append([(p, pk)])
                    break
    return fronts


def find_sequences(g: nx.DiGraph, min_len: int, max_len: int, seqs_no: int) ->\
        typing.List[typing.Tuple[Seq, SolvedSeq]]:
    max_len = min(len(g), max_len)
    all_paths = []
    nodes = list(g.nodes)
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            u = nodes[i]
            v = nodes[j]
            all_paths.extend(nx.all_simple_paths(g, u, v, max_len))
    all_paths = [p for p in all_paths if len(p) >= min_len]
    by_length = partition(all_paths, len)
    blanked = []
    for paths in by_length.values():
        blanked.extend(blank_seqs(paths))

    by_endpoints = dict(partition(blanked, lambda e: (e[1][0], e[1][-1])))
    small = set()
    for endpoints, paths in by_endpoints.items():
        if len(paths) < seqs_no:
            small.add(endpoints)
    for endpoints in small:
        del by_endpoints[endpoints]
    for paths in by_endpoints.values():
        paths2 = select_paths(paths, seqs_no)
        paths.clear()
        paths.extend(paths2)
    by_endpoints_sorted = nd_sort(by_endpoints.values())
    first_front = by_endpoints_sorted[0]
    first_front.sort(key=lambda x: x[1].pairwise_dists)
    if len(first_front) >= 2:
        return first_front[-2][0]
    return first_front[0][0]
