use hegel::TestCase;
use hegel::generators as gs;
use matterstate_envs::{pointmaze, pointpush, reacher, sokoban};

#[hegel::test]
fn pointmaze_step_stays_in_bounds_and_outside_walls(tc: TestCase) {
    let x: f32 = tc.draw(gs::floats().min_value(0.0).max_value(1.0));
    let y: f32 = tc.draw(gs::floats().min_value(0.0).max_value(1.0));
    let ax: f32 = tc.draw(gs::floats().min_value(-5.0).max_value(5.0));
    let ay: f32 = tc.draw(gs::floats().min_value(-5.0).max_value(5.0));

    let mut xs = [x];
    let mut ys = [y];
    let was_free = !pointmaze::in_wall(x, y);
    pointmaze::step_batch_in_place(&mut xs, &mut ys, &[ax], &[ay]);

    assert!(xs[0].is_finite() && ys[0].is_finite());
    assert!((0.0..=1.0).contains(&xs[0]) && (0.0..=1.0).contains(&ys[0]));
    if was_free {
        assert!(!pointmaze::in_wall(xs[0], ys[0]));
    }
}

#[hegel::test]
fn pointpush_step_preserves_bounds(tc: TestCase) {
    let ax: f32 = tc.draw(gs::floats().min_value(0.0).max_value(1.0));
    let ay: f32 = tc.draw(gs::floats().min_value(0.0).max_value(1.0));
    let ox: f32 = tc.draw(gs::floats().min_value(0.0).max_value(1.0));
    let oy: f32 = tc.draw(gs::floats().min_value(0.0).max_value(1.0));
    let ux: f32 = tc.draw(gs::floats().min_value(-5.0).max_value(5.0));
    let uy: f32 = tc.draw(gs::floats().min_value(-5.0).max_value(5.0));

    let mut axs = [ax];
    let mut ays = [ay];
    let mut oxs = [ox];
    let mut oys = [oy];
    pointpush::step_batch_in_place(&mut axs, &mut ays, &mut oxs, &mut oys, &[ux], &[uy]);

    assert!((0.0..=1.0).contains(&axs[0]) && (0.0..=1.0).contains(&ays[0]));
    assert!((0.0..=1.0).contains(&oxs[0]) && (0.0..=1.0).contains(&oys[0]));
}

#[hegel::test]
fn reacher_step_wraps_angles_to_pi(tc: TestCase) {
    let t1: f32 = tc.draw(
        gs::floats()
            .min_value(-std::f32::consts::PI)
            .max_value(std::f32::consts::PI),
    );
    let t2: f32 = tc.draw(
        gs::floats()
            .min_value(-std::f32::consts::PI)
            .max_value(std::f32::consts::PI),
    );
    let a1: f32 = tc.draw(gs::floats().min_value(-5.0).max_value(5.0));
    let a2: f32 = tc.draw(gs::floats().min_value(-5.0).max_value(5.0));

    let mut t1s = [t1];
    let mut t2s = [t2];
    reacher::step_batch_in_place(&mut t1s, &mut t2s, &[a1], &[a2]);

    let limit = std::f32::consts::PI + 1e-5;
    assert!(t1s[0].abs() <= limit && t2s[0].abs() <= limit);
}

#[hegel::test]
fn sokoban_step_preserves_player_count(tc: TestCase) {
    let action_idx: u8 = tc.draw(gs::integers().min_value(0).max_value(3));
    let pr: usize = tc.draw(gs::integers().min_value(1).max_value(8));
    let pc: usize = tc.draw(gs::integers().min_value(1).max_value(8));
    let mut cells = [sokoban::FLOOR; sokoban::CELLS];
    for i in 0..10 {
        cells[i] = sokoban::WALL;
        cells[90 + i] = sokoban::WALL;
        cells[i * 10] = sokoban::WALL;
        cells[i * 10 + 9] = sokoban::WALL;
    }
    cells[pr * 10 + pc] = sokoban::PLR;

    let board = sokoban::Board::from_cells(cells).expect("valid board");
    let action = sokoban::Action::from_u8(action_idx).expect("action in 0..=3");
    let next = board.step(action);

    let players = next
        .cells()
        .iter()
        .filter(|&&c| c == sokoban::PLR || c == sokoban::PLR_T)
        .count();
    assert_eq!(players, 1);
}
