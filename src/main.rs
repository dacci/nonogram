use anyhow::Result;
use clap::Parser;
use nonogram::{Cell, Puzzle, Solution, Solver};
use std::fs::File;
use std::path::PathBuf;

#[derive(Debug, clap::Parser)]
#[command(version, about)]
struct Args {
    /// Disable depth-first search.
    #[arg(short = 'D', long)]
    disable_dfs: bool,

    /// Path to a file that contains clues.
    path: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let puzzle = Puzzle::from_reader(File::open(&args.path)?)?;
    puzzle.validate()?;

    let mut solver = Solver::new();
    if args.disable_dfs {
        solver.set_dfs(false);
    }

    let sol = solver.solve(&puzzle)?;
    print_solution(&sol, puzzle.width());

    Ok(())
}

fn print_solution(sol: &Solution, width: usize) {
    let line = {
        let mut buf = String::with_capacity(width);
        for j in 0..width {
            if j % 5 == 0 {
                buf.push('+');
            }
            buf.push('-');
        }
        buf.push('+');
        buf
    };

    for (i, row) in sol.rows().enumerate() {
        if i % 5 == 0 {
            println!("{line}");
        }

        for (j, cell) in row.iter().enumerate() {
            if j % 5 == 0 {
                print!("|");
            }
            let c = match cell {
                Cell::Unknown => '?',
                Cell::Empty => ' ',
                Cell::Filled => 'X',
            };
            print!("{c}");
        }
        println!("|");
    }

    println!("{line}");
}
