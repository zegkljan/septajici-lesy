import typing

import matplotlib
import networkx as nx
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import Seq, SolvedSeq, NodeKind, seq2str, save_sequences


T = typing.TypeVar('T')
U = typing.TypeVar('U')


class PathMetrics(typing.NamedTuple):
    nones_count: int
    frac_nones_count: float
    min_none_sequence_len: int
    max_none_sequence_len: int
    nones_spread: float
    frac_nones_spread: float


class PathSetMetrics(typing.NamedTuple):
    # aggregated per-path metrics
    sum_nones_count: int
    min_nones_count: int
    sum_frac_nones_count: float
    min_min_none_sequence_len: int
    sum_min_none_sequence_len: int
    sum_nones_spread: float
    sum_frac_nones_spread: float
    # per-set metrics
    min_dist: int
    sum_dists: int


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


def partition(c: typing.Collection[T], key: typing.Callable[[T], U]) ->\
        typing.Mapping[U, typing.List[T]]:
    partitions = dict()
    for e in c:
        k = key(e)
        if k not in partitions:
            partitions[k] = []
        partitions[k].append(e)
    return partitions


def decorate_graph(g: nx.DiGraph):
    # add meta info - node types, edge orderings
    for n in g:
        g.add_node(n, kind=NodeKind.PEEK_ALL)
        for i, nn in enumerate(g[n]):
            g.edges[(n, nn)]['order'] = i


def generate_graph(size: int, extra_edge_prob: float) -> nx.DiGraph:
    # generate cyclic graph to ensure connectivity
    g1 = nx.cycle_graph(size, nx.DiGraph)

    # generate random graph for other connections
    g2 = nx.fast_gnp_random_graph(size, extra_edge_prob,
                                  seed=random.randint(0, 2**64), directed=True)

    # combine both graphs
    g = nx.compose(g1, g2)

    # remove self loops
    for n in g:
        if g.has_edge(n, n):
            g.remove_edge(n, n)

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


def filter_same_subsequences(seq: typing.Sequence[typing.Tuple[Seq,
                                                               SolvedSeq]],
                             min_k: int, max_k: int) ->\
        typing.Sequence[typing.Tuple[Seq, SolvedSeq]]:
    def find_nth_pre_blank_pos(s: Seq, n: int) -> int:
        for i, e in enumerate(s):
            if e is None:
                if s[i - 1] is not None:
                    if n == 0:
                        return i - 1
                    n -= 1
        return -1

    k = min_k
    work = seq
    filtered = []
    while work and k <= max_k:
        with_kth = [(s, find_nth_pre_blank_pos(s[0], k)) for s in work]
        has_kth = partition(with_kth, key=lambda x: x[1] != -1)
        filtered.extend([x for x, _ in has_kth.get(False, [])])
        work = has_kth.get(True, [])
        by_kth = partition(work, key=lambda x: x[0][0][x[1]])
        work = []
        for kth, sqs in by_kth.items():
            work.append(max(sqs, key=lambda x: len(x[0][0]))[0])
        k += 1

    return filtered


def select_paths(all_paths: typing.List[typing.Tuple[Seq, SolvedSeq]],
                 no: int) -> typing.List[typing.Tuple[Seq, SolvedSeq]]:
    return sorted(all_paths, key=lambda x: x[0].count(None))[-no:]


def compute_path_metrics(path: typing.Tuple[Seq, SolvedSeq]) -> PathMetrics:
    nones_count = path[0].count(None)
    frac_nones_count = nones_count / len(path[0])
    min_none_sequence_len = len(path[0])
    max_none_sequence_len = 0
    cnt = 0
    for e in path[0]:
        if e is not None:
            if cnt > 0:
                min_none_sequence_len = min(min_none_sequence_len, cnt)
                max_none_sequence_len = max(max_none_sequence_len, cnt)
            cnt = 0
        else:
            cnt += 1
    nones_spread = float(np.var([i for i, e in enumerate(path[0])
                                 if e is None]))
    frac_nones_spread = float(np.var([i / len(path[0])
                                      for i, e in enumerate(path[0])
                                      if e is None]))
    return PathMetrics(nones_count=nones_count,
                       frac_nones_count=frac_nones_count,
                       min_none_sequence_len=min_none_sequence_len,
                       max_none_sequence_len=max_none_sequence_len,
                       nones_spread=nones_spread,
                       frac_nones_spread=frac_nones_spread)


def compute_path_set_metrics(
        seq_set: typing.List[typing.Tuple[Seq, SolvedSeq]]) -> PathSetMetrics:
    dists = [seq_distance(a, b)
             for _, a in seq_set
             for _, b in seq_set]
    path_metrics = [compute_path_metrics(seq) for seq in seq_set]

    sum_nones_count = sum([pm.nones_count for pm in path_metrics])
    min_nones_count = min([pm.nones_count for pm in path_metrics])
    sum_frac_nones_count = sum([pm.frac_nones_count for pm in path_metrics])
    min_dist = min(dists)
    sum_dists = sum(dists)
    min_min_none_sequence_len = min([pm.min_none_sequence_len
                                     for pm in path_metrics])
    sum_min_none_sequence_len = sum([pm.min_none_sequence_len
                                     for pm in path_metrics])
    sum_nones_spread = sum([pm.nones_spread for pm in path_metrics])
    sum_frac_nones_spread = sum([pm.frac_nones_spread for pm in path_metrics])
    return PathSetMetrics(sum_nones_count=sum_nones_count,
                          min_nones_count=min_nones_count,\
                          sum_frac_nones_count=sum_frac_nones_count,
                          min_min_none_sequence_len=min_min_none_sequence_len,
                          sum_min_none_sequence_len=sum_min_none_sequence_len,
                          sum_nones_spread=sum_nones_spread,
                          sum_frac_nones_spread=sum_frac_nones_spread,
                          min_dist=min_dist,
                          sum_dists=sum_dists)


def dominates(a, b, keys: typing.List[typing.Tuple[str, int]]) -> bool:
    for k, f in keys:
        if (getattr(a, k) - getattr(b, k)) * f < 0:
            return False
    return True


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
    path_sets_keys = [(s, compute_path_set_metrics(s)) for s in path_sets]
    fronts = [[]]

    keys = [('nones_spread', 1), ('sum_dists', 1)]
    path_sets_keys.sort(key=lambda x: [getattr(x[1], k) * f for k, f in keys],
                        reverse=True)

    for p, pk in path_sets_keys:
        x = len(fronts)
        k = 0
        while True:
            dominated = False
            for pf, pfk in reversed(fronts[k]):
                if dominates(pfk, pk, keys):
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


def get_all_paths(g: nx.DiGraph, min_len: int, max_len: int) ->\
        typing.List[typing.List[int]]:
    all_paths = []
    nodes = list(g.nodes)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            u = nodes[i]
            v = nodes[j]
            paths = [p for p in nx.all_simple_paths(g, u, v, max_len - 1)
                     if len(p) >= min_len]
            all_paths.extend(paths)
    all_paths.sort()
    all_paths.sort(key=len)
    return all_paths


def select_sequences(all_paths: typing.List[SolvedSeq], seqs_no: int) ->\
        typing.List[typing.Tuple[Seq, SolvedSeq]]:
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
    first_front.sort(key=lambda x: x[1].sum_dists)
    if len(first_front) >= 2:
        return first_front[-2][0]
    return first_front[0][0]


def select_sequences_simple(
        all_paths: typing.List[SolvedSeq], seqs_no: int,
        min_blanks: int,
        path_score_key_factor: typing.Sequence[typing.Tuple[str, float]],
        path_set_score_key_factor: typing.Sequence[typing.Tuple[str, float]])\
        -> typing.List[typing.Tuple[Seq, SolvedSeq]]:
    tmp = dict(partition(all_paths, lambda e: (e[0], e[-1])))
    by_endpoints = []
    for endpoints, paths in tmp.items():
        by_length = partition(paths, len)
        blanked = []
        for ps in by_length.values():
            blanked.extend(blank_seqs(ps))
        # discard paths with too few blanks
        blanked[:] = [
            s for s in blanked
            if s[0].count(None) >= min_blanks
        ]

        if len(blanked) < seqs_no:
            continue
        with_metrics = [(p, compute_path_metrics(p)) for p in blanked]
        with_metrics.sort(key=lambda x: sum([getattr(x[1], m) * f
                                             for m, f
                                             in path_score_key_factor]),
                          reverse=True)
        del with_metrics[seqs_no:]
        by_endpoints.append([x[0] for x in with_metrics])
    by_endpoints.sort(
        key=lambda x: sum([getattr(compute_path_set_metrics(x), m) * f
                           for m, f in path_set_score_key_factor]),
        reverse=True)
    if by_endpoints:
        return by_endpoints[0]
    return []


def reduce_blanks(sqs: typing.List[typing.Tuple[Seq, SolvedSeq]],
                  no_blanks: int):
    max_reduced = 0
    min_reduced = float('inf')
    for s, ss in sqs:
        none_idx = [i for i, e in enumerate(s) if e is None]
        unblanked_idx = random.sample(none_idx, len(none_idx) - no_blanks)
        max_reduced = max(max_reduced, len(unblanked_idx))
        min_reduced = min(min_reduced, len(unblanked_idx))
        for i in unblanked_idx:
            s[i] = ss[i]
    print(f'Max reduction: {max_reduced}')
    print(f'Min reduction: {min_reduced}')


def generate_graph_sequences_1(graph_size: int, seq_min_len: int,
                               seq_max_len: int, no_blanks: int,
                               no_of_sequences: int) ->\
        typing.Tuple[nx.DiGraph, typing.List[typing.Tuple[Seq, SolvedSeq]]]:
    seqs = []
    freq_base = 0.5
    freq_factor = 1.01
    max_freq_base = 2
    max_max_freq_base = graph_size - 1
    while not seqs:
        print(f'Generating graph ({graph_size}): {freq_base} {freq_factor} '
              f'{max_freq_base} {max_max_freq_base}')
        g = generate_graph(graph_size, freq_base / (graph_size - 1))
        print(f'Getting all paths of length {seq_min_len} to {seq_max_len}')
        all_paths = get_all_paths(g, seq_min_len, seq_max_len)
        print(f'{len(all_paths)} paths')
        print('Finding sequences')
        # seqs = select_sequences(all_paths, no_of_sequences)
        seqs = select_sequences_simple(
            all_paths, no_of_sequences,
            min_blanks=no_blanks,
            path_score_key_factor=[('nones_count', 0), ('nones_spread', 1)],
            path_set_score_key_factor=[('min_dist', 1),
                                       ('sum_nones_spread', 0.1),
                                       ('min_nones_count', 0)])
        if seqs:
            reduce_blanks(seqs, no_blanks)
        freq_base = min(freq_base * freq_factor, max_freq_base)
        freq_factor *= 1.005
        if freq_base == max_freq_base:
            freq_base = 0.5
            freq_factor = 1.01
            max_freq_base = min(max_freq_base * freq_factor, max_max_freq_base)
            if max_freq_base == max_max_freq_base:
                raise ValueError('could not generate')
    print('Done')
    return g, seqs


def generate_graph_sequences_2(graph_size: int, seq_min_len: int,
                               seq_max_len: int,
                               seed: typing.Optional[int] = None) ->\
        typing.Tuple[nx.DiGraph, typing.List[typing.Tuple[Seq, SolvedSeq]]]:
    if seed is None:
        seed = int(time.time())
    random.seed(seed)

    matplotlib.use('TkAgg')
    plt.ion()

    print('Generating graph')
    g: nx.DiGraph = nx.cycle_graph(range(graph_size), nx.DiGraph)
    first = True
    while True:
        if not first:
            # select nodes with maximum out deg - in deg and in deg - out deg
            out = float('inf')
            min_out = []
            for n in g:
                v = g.out_degree(n)
                if v < out:
                    min_out = [n]
                    out = v
                elif v == out:
                    min_out.append(n)

            out_n = random.choice(min_out)
            target = random.choice(
                [n for n in g if n != out_n and (out_n, n) not in g.edges])
            print(out_n, target)
            g.add_edge(out_n, target)
        first = False

        for n, p in nx.kamada_kawai_layout(g).items():
            g.nodes[n]['pos'] = p

        all_paths = get_all_paths(g, seq_min_len, seq_max_len)
        sqs = select_sequences(all_paths, 20)

        draw(g, sqs)
        plt.show()
        plt.waitforbuttonpress()
    return g, [(x, None) for x in paths]


def get_graph_sequences() ->\
        typing.Tuple[nx.DiGraph, typing.List[typing.Tuple[Seq, SolvedSeq]]]:
    return generate_graph_sequences_1(30, 17, 17, 13, 20)
    # return generate_graph_sequences_2(10, 4, 6, seed)


def draw(g: nx.DiGraph, sqs: typing.List[typing.Tuple[Seq, SolvedSeq]],
         title: str = ''):
    plt.figure(0, figsize=(10, 5))
    plt.clf()
    if title:
        plt.suptitle(title)

    plt.subplot(1, 2, 1)
    nx.draw_networkx(g, pos=dict(g.nodes('pos')))
    plt.subplot(1, 2, 2)
    for i, (s, ss) in enumerate(sqs):
        plt.text(0, i / len(sqs),
                 '{: >2}: {}'.format(i, seq2str(s, aligned=True, align_size=2)),
                 fontfamily='monospace')
    plt.ylim((0, 1))
    plt.axis('off')


def main():
    seed = 0  # int(time.time())
    random.seed(seed)
    k = 0
    while True:
        #g, sqs = generate_graph_sequences_2(20, 8, 8, seed)
        g, sqs = get_graph_sequences()
        nx.write_graphml_xml(g, '/tmp/graph')
        save_sequences(sqs, '/tmp/sequences')
        for n, p in nx.kamada_kawai_layout(g).items():
            g.nodes[n]['pos'] = p
        draw(g, sqs, f'seed: {seed} k: {k} out deg: '
                     f'{min([deg for _, deg in g.out_degree])} | '
                     f'{sum([deg for _, deg in g.out_degree]) / len(g)} | '
                     f'{max([deg for _, deg in g.out_degree])}')
        plt.show()
        k += 1


if __name__ == '__main__':
    main()
