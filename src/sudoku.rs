//! Zero-dependency parallel Sudoku puzzle generator ported from `sudoku-rs`.

use std::thread;

use crate::rng::Rng;

const MAX_PUZZLE_TRIES: usize = 10_000;

fn box_dims(n: usize) -> (usize, usize) {
    let mut h = (n as f64).sqrt() as usize;
    while h > 0 && !n.is_multiple_of(h) {
        h -= 1;
    }
    (h, n / h)
}

fn shuffle<T>(slice: &mut [T], rng: &mut Rng) {
    let len = slice.len();
    for i in (1..len).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        slice.swap(i, j);
    }
}

fn ok(g: &[i8], n: usize, bh: usize, bw: usize, i: usize, v: i8) -> bool {
    let (r, c) = (i / n, i % n);
    for cc in 0..n {
        if g[r * n + cc] == v {
            return false;
        }
    }
    for rr in 0..n {
        if g[rr * n + c] == v {
            return false;
        }
    }
    let (br, bc) = ((r / bh) * bh, (c / bw) * bw);
    for rr in br..br + bh {
        for cc in bc..bc + bw {
            if g[rr * n + cc] == v {
                return false;
            }
        }
    }
    true
}

fn fill(g: &mut [i8], n: usize, bh: usize, bw: usize, i: usize, rng: &mut Rng) -> bool {
    if i == n * n {
        return true;
    }
    let mut order: Vec<i8> = (1..=n as i8).collect();
    shuffle(&mut order, rng);
    for v in order {
        if ok(g, n, bh, bw, i, v) {
            g[i] = v;
            if fill(g, n, bh, bw, i + 1, rng) {
                return true;
            }
            g[i] = 0;
        }
    }
    false
}

fn count_solutions(
    g: &mut [i8],
    n: usize,
    bh: usize,
    bw: usize,
    pos: usize,
    cnt: &mut usize,
    cap: usize,
) {
    if *cnt >= cap {
        return;
    }
    if pos == n * n {
        *cnt += 1;
        return;
    }
    if g[pos] != 0 {
        count_solutions(g, n, bh, bw, pos + 1, cnt, cap);
        return;
    }
    for v in 1..=n as i8 {
        if ok(g, n, bh, bw, pos, v) {
            g[pos] = v;
            count_solutions(g, n, bh, bw, pos + 1, cnt, cap);
            g[pos] = 0;
        }
    }
}

fn one_puzzle(n: usize, clue_frac: f64, unique_only: bool, rng: &mut Rng) -> (Vec<i8>, Vec<i8>) {
    let (bh, bw) = box_dims(n);
    let cells = n * n;
    let holes = ((1.0 - clue_frac) * cells as f64).round() as usize;
    for _ in 0..MAX_PUZZLE_TRIES {
        let mut sol = vec![0i8; cells];
        fill(&mut sol, n, bh, bw, 0, rng);
        let mut puz = sol.clone();
        let mut pos: Vec<usize> = (0..cells).collect();
        shuffle(&mut pos, rng);
        for &p in pos.iter().take(holes) {
            puz[p] = 0;
        }
        if !unique_only {
            return (puz, sol);
        }
        let mut g = puz.clone();
        let mut cnt = 0;
        count_solutions(&mut g, n, bh, bw, 0, &mut cnt, 2);
        if cnt == 1 {
            return (puz, sol);
        }
    }
    panic!("sudoku::one_puzzle exceeded MAX_PUZZLE_TRIES");
}

/// Generate `num` `(puzzle, solution)` flat `i8` grids of size `n * n`.
/// Deterministic per `(seed, index)` and parallelized across available CPU cores.
pub fn generate(
    n: usize,
    num: usize,
    clue_frac: f64,
    seed: u64,
    unique_only: bool,
) -> (Vec<i8>, Vec<i8>) {
    let cells = n * n;
    let mut puzzles = vec![0i8; num * cells];
    let mut solutions = vec![0i8; num * cells];
    if num == 0 {
        return (puzzles, solutions);
    }

    let threads = thread::available_parallelism()
        .map_or(1, |p| p.get())
        .min(num);
    let chunk = num.div_ceil(threads);

    thread::scope(|s| {
        for (t_idx, (p_chunk, s_chunk)) in puzzles
            .chunks_mut(chunk * cells)
            .zip(solutions.chunks_mut(chunk * cells))
            .enumerate()
        {
            let base = t_idx * chunk;
            s.spawn(move || {
                for (local, (p_dst, s_dst)) in p_chunk
                    .chunks_exact_mut(cells)
                    .zip(s_chunk.chunks_exact_mut(cells))
                    .enumerate()
                {
                    let idx = base + local;
                    let item_seed = seed
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(idx as u64);
                    let mut rng = Rng::seed_from_u64(item_seed);
                    let (puz, sol) = one_puzzle(n, clue_frac, unique_only, &mut rng);
                    p_dst.copy_from_slice(&puz);
                    s_dst.copy_from_slice(&sol);
                }
            });
        }
    });

    (puzzles, solutions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_valid_unique_4x4_puzzles() {
        let (puz, sol) = generate(4, 8, 0.5, 42, true);
        assert_eq!(puz.len(), 8 * 16);
        assert_eq!(sol.len(), 8 * 16);
        for i in 0..8 {
            let p = &puz[i * 16..(i + 1) * 16];
            let s = &sol[i * 16..(i + 1) * 16];
            for (&pv, &sv) in p.iter().zip(s.iter()) {
                assert!((1..=4).contains(&sv));
                if pv != 0 {
                    assert_eq!(pv, sv);
                }
            }
        }
    }
}
