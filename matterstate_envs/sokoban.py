"""Sokoban: a self-contained, fully-observable, deterministic planning env.

Why this and not Sudoku: pushing a box into a corner is IRREVERSIBLE — a
reactive
policy can't recover, so solving REQUIRES lookahead. But unlike a static
puzzle the
dynamics are OBSERVED: every step you predict the next grid and then see it,
so a
world model grounds on reality for free (no labels, no solver). Levels come
from
DeepMind's boxoban set (pre-generated, guaranteed solvable).

State is a flat categorical grid (H*W,) over 7 cell types. The overlap
encoding
(box/player ON a target are their own classes) makes step() self-sufficient
— when a
mover leaves a cell, its class alone says whether to restore floor or
target.
"""

import numpy as np

WALL, FLOOR, TARGET, BOX, BOX_T, PLR, PLR_T = range(7)
_CHAR = {
    "#": WALL,
    " ": FLOOR,
    ".": TARGET,
    "$": BOX,
    "*": BOX_T,
    "@": PLR,
    "+": PLR_T,
}
DIRS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right
N_ACT = 4
H = W = 10


def load_levels(path, limit=None):
    """Parse a boxoban .txt (blocks of `; idx` + 10 grid lines) → (N, H*W) int8
    grids.
    """
    if limit is not None and limit <= 0:
        return np.empty((0, H * W), np.int8)
    grids, rows = [], []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(";"):
                if rows:
                    grids.append(
                        np.array(
                            [[_CHAR[c] for c in r] for r in rows], np.int8
                        ).ravel()
                    )
                    rows = []
                if limit is not None and len(grids) >= limit:
                    return np.stack(grids)
            elif line:
                rows.append(line)
    if rows:
        grids.append(
            np.array([[_CHAR[c] for c in r] for r in rows], np.int8).ravel()
        )
    if not grids:
        return np.empty((0, H * W), np.int8)
    return np.stack(grids)


def step(grid, action):
    """Apply one move to a flat grid (H*W,). Returns the next grid (a copy).
    Illegal
    moves (into a wall, or pushing a box into a wall/box) are no-ops — the
    grid is
    returned unchanged, which is itself a valid transition to learn.
    """
    if not (0 <= action < N_ACT):
        raise ValueError(f"invalid action: {action}")
    g = grid.reshape(H, W).copy()
    (pr, pc) = np.argwhere((g == PLR) | (g == PLR_T))[0]
    on_t = g[pr, pc] == PLR_T
    dr, dc = DIRS[action]
    nr, nc = pr + dr, pc + dc
    if not (0 <= nr < H and 0 <= nc < W) or g[nr, nc] == WALL:
        return g.ravel()
    if g[nr, nc] in (BOX, BOX_T):
        br, bc = nr + dr, nc + dc
        if not (0 <= br < H and 0 <= bc < W) or g[br, bc] in (WALL, BOX, BOX_T):
            return g.ravel()
        g[br, bc] = BOX_T if g[br, bc] == TARGET else BOX
        g[nr, nc] = PLR_T if g[nr, nc] == BOX_T else PLR
    else:
        g[nr, nc] = PLR_T if g[nr, nc] == TARGET else PLR
    g[pr, pc] = TARGET if on_t else FLOOR
    return g.ravel()


def make_solved(grid):
    """A solved board sharing this level's walls/targets: every target holds a
    box,
    the player sits on some free floor. (#boxes == #targets in boxoban, so
    this is a
    valid goal state.) Used as the seed for reverse play.
    """
    g = np.asarray(grid).reshape(H, W).copy()
    g[(g == BOX) | (g == PLR)] = FLOOR  # clear movers off non-target cells
    g[(g == BOX_T) | (g == PLR_T)] = TARGET
    g[g == TARGET] = BOX_T  # fill every target with a box
    free = np.argwhere(g == FLOOR)
    pr, pc = free[len(free) // 2]
    g[pr, pc] = PLR
    return g.ravel()


def reverse_predecessors(grid):
    """All (predecessor, action) with step(predecessor, action) == grid — i.e.
    the
    forward move `action` carries predecessor → grid. Inverts step()'s two
    cases (a
    plain move and a push). A reverse PULL (predecessor has a box where the
    player now
    is) is what drags boxes off targets to build a puzzle; `action` points
    back toward
    the solved state, so it's a ready-made policy label.
    """
    g = np.asarray(grid).reshape(H, W)
    (pr, pc) = np.argwhere((g == PLR) | (g == PLR_T))[0]
    p_under_t = g[pr, pc] == PLR_T
    out = []
    for a, (dr, dc) in enumerate(DIRS):
        qr, qc = pr - dr, pc - dc  # where the player came FROM
        if not (0 <= qr < H and 0 <= qc < W) or g[qr, qc] in (WALL, BOX, BOX_T):
            continue
        # plain move: predecessor = player at q, current player-cell freed
        gp = g.copy()
        gp[qr, qc] = PLR_T if gp[qr, qc] == TARGET else PLR
        gp[pr, pc] = TARGET if p_under_t else FLOOR
        out.append((gp.ravel(), a))
        # pull: a box sits ahead (pr+dr) and gets pulled back into the player's
        # cell
        fr, fc = pr + dr, pc + dc
        if 0 <= fr < H and 0 <= fc < W and g[fr, fc] in (BOX, BOX_T):
            gp = g.copy()
            gp[qr, qc] = PLR_T if gp[qr, qc] == TARGET else PLR
            gp[pr, pc] = BOX_T if p_under_t else BOX
            gp[fr, fc] = TARGET if gp[fr, fc] == BOX_T else FLOOR
            out.append((gp.ravel(), a))
    return out


def solved(grid):
    """True iff every box sits on a target (no bare BOX cells left)."""
    return not (np.asarray(grid) == BOX).any()


def n_on_target(grid):
    return int((np.asarray(grid) == BOX_T).sum())
