from __future__ import annotations

import os
import typing

import networkx as nx
import reportlab.graphics.renderPDF as renderPDF
import reportlab.lib.pagesizes as pagesizes
import reportlab.lib.units as units
import svglib.svglib as svglib
import matplotlib.pyplot as plt
from reportlab.pdfgen.canvas import Canvas

from utils import (Seq, SolvedSeq, NodeKind)


def fit_size(drawing, w, h):
    sx = w / drawing.width
    sy = h / drawing.height
    drawing.scale(sx, sy)
    return drawing


def save_design_graph(g: nx.DiGraph, savedir: str):
    plt.figure(0, figsize=(8, 8))
    nx.draw_networkx(g, pos=dict(g.nodes('pos')))
    plt.savefig(os.path.join(savedir, 'graph.pdf'))


def save_design_nodes(graph: nx.DiGraph, savedir: str, symbols_dir: str):
    pg_w = pagesizes.A5[0]
    pg_h = pagesizes.A5[1]
    dirs_btm_padding = 10 * units.mm
    symbol_w = 100 * units.mm
    symbol_h = 100 * units.mm
    dir_symbol_w = 20 * units.mm
    dir_symbol_h = 20 * units.mm
    meta_font_size = 5 * units.mm
    fontname = 'Helvetica'

    fn = os.path.join(savedir, 'uzly.pdf')
    canvas = Canvas(fn, pagesize=(pg_w, pg_h))
    canvas.setTitle('Uzly')
    for n in graph:
        kind = graph.nodes[n]['kind']

        hl_y = 2 * dirs_btm_padding + dir_symbol_h
        canvas.line(0, hl_y, pg_w, hl_y)

        symbol_fn = os.path.join(symbols_dir, '{:02d}.svg'.format(n))
        symbol = fit_size(svglib.svg2rlg(symbol_fn), symbol_w, symbol_h)
        symbol.getContents()[0].contents.pop()
        renderPDF.draw(symbol, canvas,
                       (pg_w - symbol_w) / 2, (pg_h + hl_y - symbol_h) / 2)

        if kind == NodeKind.PEEK_ALL:
            text = '*'
        elif kind == NodeKind.PEEK_ONE:
            text = '1'
        else:
            raise ValueError()
        canvas.setFont(fontname, meta_font_size)
        # canvas.drawRightString(pg_w - 5 * units.mm, pg_h - 9 * units.mm, text)
        nbnum = len(graph[n])
        for i in range(nbnum - 1):
            x = (i + 1) * pg_w / nbnum
            canvas.line(x, 0,
                        x, 2 * dirs_btm_padding + dir_symbol_h)
        for nb, nbd in graph[n].items():
            symbol_fn = os.path.join(symbols_dir, '{:02d}.svg'.format(nb))
            o = nbd['order']
            x = (o + 0.5) * pg_w / nbnum - dir_symbol_w / 2
            y = dirs_btm_padding
            symbol = fit_size(svglib.svg2rlg(symbol_fn),
                              dir_symbol_w, dir_symbol_h)
            symbol.getContents()[0].contents.pop()
            renderPDF.draw(symbol, canvas, x, y)
        canvas.showPage()
    canvas.save()


def save_design_cards(seqs: typing.List[Seq], savedir: str, symbols_dir: str,
                      file_basename: str, title: str):
    pg_w = pagesizes.A7[1]
    pg_h = pagesizes.A7[0] / 2
    lr_pad = 5 * units.mm
    dot_r = .5 * units.mm
    dot_hpad = 2 * units.mm
    dot_vpad = .5 * units.mm

    fn = os.path.join(savedir, f'{file_basename}.pdf')
    canvas = Canvas(fn, pagesize=(pg_w, pg_h))
    canvas.setTitle(title)
    for i, s in enumerate(seqs):
        i = i + 1
        sl = len(s)

        row = -1
        col = 0
        for k in range(i):
            if k % 5 == 0:
                row += 1
                col = 0

            rl = min(5, i - 5 * row)
            c = (rl - 1) * (2 * dot_r + dot_hpad) / 2
            x = pg_w / 2 - c + col * (2 * dot_r + dot_hpad)
            y = pg_h - dot_vpad - dot_r - row * (dot_vpad + 2 * dot_r)
            canvas.circle(x, y, dot_r, stroke=0, fill=1)
            col += 1

        size = (pg_w - 2 * lr_pad) / sl

        canvas.grid([lr_pad + k * size
                     for k in range(sl + 1)],
                    [(pg_h - size) / 2,
                     (pg_h + size) / 2])

        for j, e in enumerate(s):
            if e is None:
                continue
            symbol_fn = os.path.join(symbols_dir, '{:02d}.svg'.format(e))
            symbol = fit_size(svglib.svg2rlg(symbol_fn), size, size)
            symbol.getContents()[0].contents.pop()
            x = lr_pad + j * size
            y = (pg_h - size) / 2
            renderPDF.draw(symbol, canvas, x, y)
        canvas.showPage()
    canvas.save()


def save_design(graph: nx.DiGraph,
                seqs: typing.List[typing.Tuple[Seq, SolvedSeq]], savedir: str,
                symbols_dir: str):
    save_design_graph(graph, savedir)
    save_design_nodes(graph, savedir, symbols_dir)
    save_design_cards([x[0] for x in seqs], savedir, symbols_dir, 'zadani',
                      'Zadání')
    save_design_cards([x[1] for x in seqs], savedir, symbols_dir, 'reseni',
                      'Řešení')
