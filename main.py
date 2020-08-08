from __future__ import annotations

import copy
import logging
import logging.config
import typing

import matplotlib.pyplot as plt
import networkx
from matplotlib.backends.backend_pdf import PdfPages

from utils import (Seq, shortest_path, length_n_path,
                   SolvedSeq, NodeKind, seq2str, blank_seqs)


def get_seq(g: networkx.DiGraph) -> Seq:
    paths = []
    nodes = list(g.nodes)
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            u = nodes[i]
            v = nodes[j]
            paths.extend(networkx.all_simple_paths(g, u, v, len(g) // 2))
    paths.sort()
    paths.sort(key=len, reverse=True)
    paths = list(filter(lambda p: len(p) == len(paths[0]), paths))
    partitions = dict()
    for p in paths:
        pair = (p[0], p[-1])
        if pair not in partitions:
            partitions[pair] = []
        partitions[pair].append(p)
    partitions = list(partitions.values())
    partitions.sort(key=len, reverse=True)
    paths = partitions[0]
    blanked = blank_seqs(paths)
    #return [0, 2, None, 5]
    return blanked[0]


def get_graph() -> networkx.DiGraph:
    # gr = {
    #     0: (NodeKind.PEEK_ALL, [1, 2]),
    #     1: (NodeKind.PEEK_ALL, [0, 3, 4]),
    #     2: (NodeKind.PEEK_ALL, [0, 3, 4]),
    #     3: (NodeKind.PEEK_ALL, [0, 5]),
    #     4: (NodeKind.PEEK_ALL, [0, 5]),
    #     5: (NodeKind.PEEK_ALL, [0]),
    # }
    # return tg2digraph(gr)

    g = networkx.gnm_random_graph(20, 20 * 3, seed=0, directed=True)
    for n in g:
        if g.has_edge(n, n):
            g.remove_edge(n, n)
    for n in g:
        g.add_node(n, kind=NodeKind.PEEK_ALL)
        for i, nn in enumerate(g[n]):
            g.edges[(n, nn)]['order'] = i
    return g


def split_seq(seq: Seq) -> typing.List[Seq]:
    assert seq[0] is not None and seq[-1] is not None
    subseqs = []
    subseq = [seq[0]]
    for n in seq[1:]:
        subseq.append(n)
        if n is not None:
            subseqs.append(subseq)
            subseq = [n]
    return subseqs


def join(seq1: Seq, seq2: Seq) -> Seq:
    if seq1:
        if seq2:
            assert seq1[-1] == seq2[0]
            return seq1 + seq2[1:]
        return list(seq1)
    return list(seq2)


class Action:
    ACTION = None

    def __init__(self, node: int):
        self.node: int = node

    def __str__(self):
        return f'{self.ACTION} at {self.node}'


class NoAction(Action):
    ACTION = 'NO-ACTION'

    def __init__(self, node: int):
        super(NoAction, self).__init__(node)


class PeekNodeTypeAction(Action):
    ACTION = 'PEEK-NODE-TYPE'

    def __init__(self, node: int):
        super().__init__(node)


class PeekNeighboursAction(Action):
    ACTION = 'PEEK-NEIGHBOURS'

    def __init__(self, node: int, neighbour_nums: typing.Sequence[int]):
        super().__init__(node)
        self.neighbour_nums = set(neighbour_nums)

    def __str__(self):
        return (super(PeekNeighboursAction, self).__str__() +
                f' neigbour nums {self.neighbour_nums}')


class MoveAction(Action):
    ACTION = 'MOVE'

    def __init__(self, node: int, to: int):
        super().__init__(node)
        self.to = to

    def __str__(self):
        return super(MoveAction, self).__str__() + f' to {self.to}'


class State:
    def __init__(self, true_graph: networkx.DiGraph):
        self.log = logging.getLogger('State')
        self.graph: networkx.DiGraph = true_graph
        self.actions: typing.List[typing.Tuple[typing.Type[Action],
                                               networkx.DiGraph,
                                               typing.List[str]]] = []

    def _add_action(self, action: typing.Type[Action], agent: Agent):
        self.actions.append((action, agent.map.copy(),
                             copy.deepcopy(agent.task_stack)))
        self.log.debug(str(action))

    def move_agent(self, agent: Agent, neighbour_node: int):
        if agent.pos == neighbour_node:
            return
        if neighbour_node not in self.graph[agent.pos]:
            raise ValueError(f'attempted to move from {agent.pos} to '
                             f'{neighbour_node} but it is not among neighbour '
                             f'nodes')
        peeked = False
        for a, _, _ in reversed(self.actions):
            if a.node == agent.pos:
                if a.ACTION == PeekNeighboursAction.ACTION:
                    peeked_neighbours = set()
                    for _, nb, order in self.graph.edges(agent.pos,
                                                         data='order'):
                        if (order in typing.cast(PeekNeighboursAction,
                                                 a).neighbour_nums):
                            peeked_neighbours.add(nb)
                    if neighbour_node in peeked_neighbours:
                        peeked = True
                    else:
                        break
            else:
                break
        if not peeked:
            raise ValueError(f'attempted to move from node {agent.pos} to '
                             f'{neighbour_node} without peeking it')

        # noinspection PyTypeChecker
        self._add_action(MoveAction(agent.pos, neighbour_node), agent)
        agent.pos = neighbour_node
        agent.peeked = False

    def peek_node_type(self, agent: Agent):
        agent.map.add_node(agent.pos, kind=self.graph.nodes[agent.pos]['kind'])
        # noinspection PyTypeChecker
        self._add_action(PeekNodeTypeAction(agent.pos), agent)

    def peek_all_neighbours(self, agent: Agent):
        pos_kind = self.graph.nodes[agent.pos]['kind']
        if pos_kind != NodeKind.PEEK_ALL:
            raise ValueError(f'attempted to peek all neighbours at node '
                             f'{agent.pos} which is of kind {pos_kind.name}')
        for a, _, _ in reversed(self.actions):
            if a.node == agent.pos:
                if a.ACTION == PeekNeighboursAction.ACTION:
                    raise ValueError(f'attempted multiple peeks at node '
                                     f'{agent.pos}')
            else:
                break

        ns = []
        for c, nb, o in self.graph.edges(agent.pos, data='order'):
            agent.map.add_edge(c, nb, order=o)
            ns.append(o)
        agent.peeked = True
        # noinspection PyTypeChecker
        self._add_action(PeekNeighboursAction(agent.pos, ns), agent)

    def peek_neighbour(self, agent: Agent, neighbour_no: int):
        pos_kind = self.graph.nodes[agent.pos]['kind']
        if pos_kind not in (NodeKind.PEEK_ALL, NodeKind.PEEK_ONE):
            raise ValueError(f'attempted to peek neighbour no {neighbour_no} '
                             f'at node {agent.pos} which is of kind '
                             f'{pos_kind.name}')
        for a, _ in reversed(self.actions):
            if a.node == agent.pos:
                if a.ACTION == PeekNeighboursAction.ACTION:
                    raise ValueError(f'attempted multiple peeks at node '
                                     f'{agent.pos}')
            else:
                break
        if not (0 <= neighbour_no < self.graph.out_degree(agent.pos)):
            raise ValueError(f'attempted to peek neighbour no {neighbour_no} '
                             f'at node {agent.pos} but it has '
                             f'{self.graph.out_degree(agent.pos)} neighbours')
        neighbour = None
        for _, v, o in self.graph.edges(agent.pos, data='order'):
            if o == neighbour_no:
                neighbour = v
                break
        agent.map.add_edge(agent.pos, neighbour, order=neighbour_no)
        agent.peeked = True
        # noinspection PyTypeChecker
        self._add_action(PeekNeighboursAction(agent.pos, [neighbour_no]), agent)

    def no_action(self, agent):
        # noinspection PyTypeChecker
        self._add_action(NoAction(agent.pos), agent)

    def generate_progress(self, outfile, final_seq: SolvedSeq, suptitle):
        pos = dict(self.graph.nodes('pos'))
        max_task_stack = len(max(self.actions, key=lambda x: len(x[2]))[2])
        with PdfPages(outfile) as p:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, gridspec_kw={
                'wspace': 0.05,
                'width_ratios': [2, 2, 1],
                'left': 0,
            })
            fig.set_size_inches(10, 5)
            for a, m, ts in self.actions:
                hnodes = {a.node}
                hedges = set()
                if a.ACTION == 'MOVE':
                    hnodes.add(a.to)
                    hedges.add((a.node, a.to))
                self.draw_graphs(self.graph, m, pos, ax1, ax2, hnodes, hedges,
                                 'r', 'r', 'c', 'k')
                ax3.clear()
                for i, t in enumerate(ts):
                    ax3.text(0, 0.2 + 0.2 * i, t)
                ax3.set_ylim((0, max_task_stack))
                ax3.tick_params(axis='x', which='both', bottom=False, top=False,
                                labelbottom=False)
                ax3.tick_params(axis='y', which='both', left=False, right=False,
                                labelleft=False)
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                ax3.spines['bottom'].set_visible(False)
                ax3.spines['left'].set_visible(False)

                ax1.set_xlabel('whole graph')
                ax2.set_xlabel('agent map')
                ax3.set_xlabel('task stack')
                fig.suptitle('{}\n{}'.format(suptitle, str(a)))
                p.savefig(fig)
            hnodes = set(final_seq)
            hedges = {e for e in zip(final_seq, final_seq[1:])}
            self.draw_graphs(self.graph, m, pos, ax1, ax2, hnodes, hedges,
                             'y', 'y', 'c', 'k')
            ax1.set_xlabel('whole graph')
            ax2.set_xlabel('agent map')
            ax3.set_xlabel('task stack')
            fig.suptitle('{}\nfinal path'.format(suptitle))
            p.savefig(fig)

    def draw_graphs(self, g1, g2, pos, ax1, ax2, hn, he, hnc, hec, ncd, ecd):
        nc1 = [hnc if n in hn else c
               for n, c in g1.nodes(data='___', default=ncd)]
        ec1 = [hec if (u, v) in he else c
               for u, v, c in g1.edges(data='___', default=ecd)]
        nc2 = [hnc if n in hn else c
               for n, c in g2.nodes(data='___', default=ncd)]
        ec2 = [hec if (u, v) in he else c
               for u, v, c in g2.edges(data='___', default=ecd)]
        ax1.clear()
        ax2.clear()
        networkx.draw_networkx(g1, pos, ax=ax1, node_color=nc1, edge_color=ec1)
        networkx.draw_networkx(g2, pos, ax=ax2, node_color=nc2, edge_color=ec2)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())


class Agent:
    def __init__(self, seq: Seq):
        self.log = logging.getLogger('Agent')
        assert seq[0] is not None
        self.seq = seq
        self.pos: int = seq[0]
        self.map: networkx.DiGraph = networkx.DiGraph()
        self.peeked: bool = False
        self.task_stack: typing.List[str] = []

    def __repr__(self):
        return f'pos: {self.pos}'

    def pop_task_stack(self, state: State):
        self.task_stack.pop()
        state.no_action(self)

    def push_task_stack(self, state: State, msg: str):
        self.task_stack.append(msg)
        state.no_action(self)

    def solve(self, state: State):
        self.log.info('solving master sequence %s', seq2str(self.seq))
        self.push_task_stack(state, f'solve {seq2str(self.seq)}')

        assert self.seq[0] == self.pos

        # always peek node type at the current position
        state.peek_node_type(self)

        # split sequence into subsequences that start and end with known nodes
        # and have only unknown nodes in between
        subseqs = split_seq(self.seq)
        solved_seq = []
        for subseq in subseqs:
            # solve each simple sequence
            solved_subseq = self.solve_simple(subseq, state)
            solved_seq = join(solved_seq, solved_subseq)

        return solved_seq

    def solve_simple(self, simple_seq: Seq, state: State) -> typing.List[int]:
        self.log.info('solving simple sequence %s', seq2str(simple_seq))
        self.push_task_stack(state, f'solve {seq2str(simple_seq)}')

        assert self.pos == simple_seq[0]

        # search for the path
        assert self.pos == simple_seq[0]
        path = self.search_path(simple_seq[0], simple_seq[-1], len(simple_seq),
                                len(simple_seq), state)

        # move to the end point
        # this may involve additional searching
        self.push_task_stack(state, f'move to {simple_seq[-1]}')
        move_stack = [[simple_seq[-1]]]
        while self.pos != move_stack[-1][0]:
            subpath = self.search_path(self.pos, simple_seq[-1], 0, 1024, state)
            move_stack.append(subpath)
        for subpath in reversed(move_stack[1:]):
            self.move_along_path(subpath, state)
        self.pop_task_stack(state)

        self.log.info('solved simple sequence %s: %s', simple_seq, path)
        self.pop_task_stack(state)
        return path

    def search_path_in_map(self, start: int, end: int, length: int) ->\
            typing.Optional[SolvedSeq]:
        self.log.info('searching for path from %s to %s of length %s in map',
                      start, end, length)

        path = length_n_path(self.map, start, end, length)
        if path is None:
            self.log.info('path from %s to %s of length %s NOT FOUND in map',
                          start, end, length)
        else:
            self.log.info('path from %s to %s of length %s FOUND in map',
                          start, end, length)
        return path

    def search_path(self, start: int, end: int, min_length: int,
                    max_length: int, state: State) -> SolvedSeq:
        # we have to be at start point
        assert self.pos == start

        # TODO must search for SIMPLE paths
        self.log.info('searching for path from %s to %s of length %s in world',
                      start, end, (min_length, max_length))

        # SEARCH for the path:
        #   1) is there a path of correct length in the map?
        #      yes -> return that path
        #   2) is current node the end node?
        #      yes -> is min_length > 0?
        #             yes -> BACKTRACK
        #             no  -> path found!
        #      no  -> is max_length <= 0?
        #             yes -> BACKTRACK
        #   3) pick the first neighbouring node that is not 'closed'
        #      decrease min_length and max_length
        #      add the neighbouring node to the path
        #      mark the neighbouring node as 'closed'
        #      MOVE to the neighbouring node
        #   4) SEARCH for the path

        # BACKTRACK - move one node back
        #   1) SEARCH for the path to previous node
        #   2) move along the path to previous node

        def search(_end: int, min_l: int, max_l: int, lvl: int,
                   task: typing.Optional[str]=None) ->\
                typing.Optional[SolvedSeq]:
            self.log.info('%s|%s -> %s| within (%s, %s)',
                          '\t' * (lvl + 1), self.pos, _end, min_l, max_l)
            self.push_task_stack(
                state, f'search {min_l} <= |{self.pos}->{_end}| <= {max_l}')
            if (self.pos not in self.map or
                    self.map.nodes.data('kind')[self.pos] is None):
                # peek node type if we don't know it
                state.peek_node_type(self)
            if (not self.peeked and
                    self.map.nodes.data('kind')[self.pos] == NodeKind.PEEK_ALL):
                # peek all neighbours if we can
                state.peek_all_neighbours(self)

            # if zero-length path is required, return no path immediately
            if min_l == max_l == 0:
                self.pop_task_stack(state)
                return None

            # try to find the shortest path within length bounds in the map
            # if there is one, that's it
            self.log.info('%s|%s -> %s| within (%s, %s) - searching in map',
                          '\t' * (lvl + 1), self.pos, _end, min_l, max_l)
            p = None
            if min_l == 0:
                p = shortest_path(self.map, self.pos, _end)
                if p is not None and len(p) - 1 <= max_l:
                    p = p[1:]
            else:
                for l in range(min_l, max_l + 1):
                    p = length_n_path(self.map, self.pos, _end, l + 1)
                    if p is not None:
                        p = p[1:]
                        break
            if p is not None:
                self.log.info('%s|%s -> %s| within (%s, %s) - found in map: %s',
                              '\t' * (lvl + 1), self.pos, _end, min_l, max_l,
                              p)
                self.pop_task_stack(state)
                return p

            self.log.info('%s|%s -> %s| within (%s, %s) - not found in map',
                          '\t' * (lvl + 1), self.pos, _end, min_l, max_l)
            p = []
            # are we at the end point?
            if self.pos == _end:
                # we are
                # are there any points left?
                if min_l > 0:
                    # there are -> failure
                    self.log.info(
                        '%s|%s -> %s| within (%s, %s) - at end under min len',
                        '\t' * (lvl + 1), self.pos, _end, min_l, max_l)
                    self.pop_task_stack(state)
                    return None
                else:
                    # there are not -> success, return path
                    self.log.info(
                        '%s|%s -> %s| within (%s, %s) - at end within bounds - '
                        'found', '\t' * (lvl + 1), self.pos, _end, min_l, max_l)
                    self.pop_task_stack(state)
                    return p
            else:
                # we are not
                # are we too long already?
                if max_l <= 0:
                    # we are -> failure
                    self.log.info(
                        '%s|%s -> %s| within (%s, %s) - not at end over max '
                        'len', '\t' * (lvl + 1), self.pos, _end, min_l, max_l)
                    self.pop_task_stack(state)
                    return None

            # save current position for backtracking purposes
            pos = self.pos

            # we are not at the end point and we are not too long
            # try one neighbour at a time and search the rest of the path from
            # there
            for _, neighbour, neighbour_no in self.map.edges(pos, data='order'):
                self.log.info('%s|%s -> %s| (at %s) within (%s, %s) - trying '
                              'neighbour no %s: %s',
                              '\t' * (lvl + 1), pos, _end, self.pos, min_l,
                              max_l, neighbour_no, neighbour)
                # we have to peek if we didn't already
                # if we did but the node type is not PEEK_ALL, something's wrong
                if (self.peeked and
                        self.map.nodes[self.pos]['kind'] != NodeKind.PEEK_ALL):
                    raise ValueError(f'node {self.pos} already peeked but is '
                                     f'not PEEK_ALL')
                if not self.peeked:
                    state.peek_neighbour(self, neighbour_no)
                    o2n = {o: nb for _, nb, o
                           in self.map.edges(self.pos, data='order')}
                    # the neighbour should have been None or had not changed
                    assert (neighbour is None or
                            neighbour == o2n[neighbour_no])
                    neighbour = o2n[neighbour_no]
                # move to the neighbour
                state.move_agent(self, neighbour)
                # add the neighbour to the path
                p.append(neighbour)
                # find the rest of the path from the neighbour
                subpath = search(_end, max(min_l - 1, 0), max_l - 1, lvl + 1)
                # did we manage to find the path?
                if subpath is None:
                    # we did not -> backtrack
                    self.log.info(
                        '%s|%s -> %s| (at %s) within (%s, %s) - subpath not '
                        'found, backtrack to %s',
                        '\t' * (lvl + 1), pos, _end, self.pos, min_l,
                        max_l, pos)
                    self.push_task_stack(state, f'backtrack to {pos}')

                    # find path from where we are currently back to the previous
                    # node as there might not be a direct edge
                    # we don't care how long the path is so min length is 0 and
                    # max length is some ridiculously big number
                    return_path = search(pos, 0, 2**10, lvl + 1)

                    if return_path is None:
                        raise ValueError('could not find return path')

                    # move along the return path
                    self.move_along_path(return_path, state)
                    self.log.info(
                        '%s|%s -> %s| (at %s) within (%s, %s) - backtracked to '
                        '%s',
                        '\t' * (lvl + 1), pos, _end, self.pos, min_l,
                        max_l, pos)
                    self.pop_task_stack(state)
                    p.pop()
                else:
                    # we found a subpath, return it
                    self.pop_task_stack(state)
                    return p + subpath
            # all neighbours were exhausted in searching for subpath -> failure
            self.log.info('%s|%s -> %s| (at %s) within (%s, %s) - exhausted '
                          'all neighbours',
                          '\t' * (lvl + 1), pos, _end, self.pos, min_l,
                          max_l)
            self.pop_task_stack(state)
            return None

        subpath = search(end, max(0, min_length - 1), max_length - 1, 0)
        if subpath is None:
            raise ValueError(f'there is no path from {start} to {end} of '
                             f'length {(min_length, max_length)}')
        return [start] + subpath

    def move_along_path(self, path: typing.List[int], state: State):
        self.push_task_stack(state, f'move along {path}')
        for n in path:
            state.peek_node_type(self)
            if not self.peeked:
                if self.map.nodes[self.pos]['kind'] == NodeKind.PEEK_ALL:
                    state.peek_all_neighbours(self)
                elif self.map.nodes[self.pos]['kind'] == NodeKind.PEEK_ONE:
                    for _, nn, i in self.map.edges(self.pos, data='order'):
                        if nn == n:
                            state.peek_neighbour(self, i)
                            break
            state.move_agent(self, n)
        self.pop_task_stack(state)


def main():
    graph = get_graph()
    for n, p in networkx.kamada_kawai_layout(graph).items():
        graph.nodes[n]['pos'] = p
    seq = get_seq(graph)
    agent = Agent(seq)
    state = State(graph)
    solved_seq = agent.solve(state)
    print(solved_seq)
    state.generate_progress('/tmp/progress.pdf', solved_seq, seq2str(seq))


if __name__ == '__main__':
    logging.basicConfig(
        format='%(name)s - %(levelname)s - %(message)s  '
               '(%(funcName)s:%(lineno)s)',
        level=logging.DEBUG
    )
    main()
