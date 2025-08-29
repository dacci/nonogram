use std::io;
use std::iter::StepBy;
use std::num::ParseIntError;
use std::ops::{Deref, DerefMut, Range};

#[derive(Debug, thiserror::Error)]
pub enum PuzzleError {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    ParseInt(#[from] ParseIntError),

    #[error("Row {0} exceeds width")]
    Row(usize),

    #[error("Column {0} exceeds height")]
    Column(usize),
}

#[derive(Debug, Clone)]
pub struct Puzzle {
    pub rows: Vec<Vec<usize>>,
    pub cols: Vec<Vec<usize>>,
}

impl Puzzle {
    pub fn from_reader(r: impl io::Read) -> Result<Self, PuzzleError> {
        use std::io::BufRead;

        let mut rows = Vec::new();
        let mut cols = Vec::new();

        let lines = io::BufReader::new(r).lines();
        let mut row = true;
        for i in lines {
            let line = i?;
            if line.starts_with('*') {
                continue;
            }
            if line.starts_with('&') {
                row = false;
                continue;
            }

            let constraints = line
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<_, _>>()?;
            if row {
                rows.push(constraints);
            } else {
                cols.push(constraints);
            }
        }

        Ok(Self { rows, cols })
    }

    pub fn validate(&self) -> Result<(), PuzzleError> {
        for (pos, row) in self.rows.iter().enumerate() {
            if self.width() < row.iter().sum::<usize>() + row.len() - 1 {
                return Err(PuzzleError::Row(pos + 1));
            }
        }

        for (pos, col) in self.cols.iter().enumerate() {
            if self.height() < col.iter().sum::<usize>() + col.len() - 1 {
                return Err(PuzzleError::Column(pos + 1));
            }
        }

        Ok(())
    }

    pub fn width(&self) -> usize {
        self.cols.len()
    }

    pub fn height(&self) -> usize {
        self.rows.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    #[error("Conflict detected")]
    Conflict,

    #[error("No solution")]
    NoSolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Unknown,
    Empty,
    Filled,
}

#[derive(Debug, Clone)]
pub struct Solution {
    width: usize,
    height: usize,
    h_cells: Vec<Cell>,
    v_cells: Vec<Cell>,
}

impl Solution {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            h_cells: vec![Cell::Unknown; width * height],
            v_cells: vec![Cell::Unknown; height * width],
        }
    }

    fn set(&mut self, row: usize, col: usize, cell: Cell) -> Result<bool, SolverError> {
        if self.height <= row || self.width <= col {
            return Ok(false);
        }

        let index = row * self.width + col;
        let data = &mut self.h_cells[index];
        if *data == Cell::Unknown {
            *data = cell;
            self.v_cells[col * self.height + row] = cell;
            Ok(true)
        } else if *data == cell {
            Ok(false)
        } else {
            Err(SolverError::Conflict)
        }
    }

    fn filled(&self) -> bool {
        self.h_cells.iter().all(|&c| c != Cell::Unknown)
    }

    pub fn check(&self, puz: &Puzzle) -> bool {
        // Check row constraints
        for row in 0..self.height {
            let mut col = 0;
            for i in 0..puz.rows[row].len() {
                // Ignore leading empty spaces
                while col < self.width && self.h_cells[row * self.width + col] == Cell::Empty {
                    col += 1;
                }

                for _ in 0..puz.rows[row][i] {
                    // Out of bounds
                    if self.width <= col {
                        return false;
                    }
                    if self.h_cells[row * self.width + col] != Cell::Filled {
                        return false;
                    }
                    col += 1;
                }
            }

            // Check trailing empty spaces
            while col < self.width {
                if self.h_cells[row * self.width + col] != Cell::Empty {
                    return false;
                }
                col += 1;
            }
        }

        // Check column constraints
        for col in 0..self.width {
            let mut row = 0usize;
            for i in 0..puz.cols[col].len() {
                // Ignore leading empty spaces
                while row < self.height && self.h_cells[row * self.width + col] == Cell::Empty {
                    row += 1;
                }

                for _ in 0..puz.cols[col][i] {
                    // Out of bounds
                    if self.height <= row {
                        return false;
                    }
                    if self.h_cells[row * self.width + col] != Cell::Filled {
                        return false;
                    }
                    row += 1;
                }
            }

            // Check trailing empty spaces
            while row < self.height {
                if self.h_cells[row * self.width + col] != Cell::Empty {
                    return false;
                }
                row += 1;
            }
        }

        true
    }

    pub fn rows(&self) -> impl Iterator<Item = &[Cell]> {
        SliceIterator::new(self.h_cells.as_slice(), self.width)
    }

    pub fn columns(&self) -> impl Iterator<Item = &[Cell]> {
        SliceIterator::new(self.v_cells.as_slice(), self.height)
    }
}

pub struct SliceIterator<'a> {
    slice: &'a [Cell],
    step: usize,
    pos: StepBy<Range<usize>>,
}

impl<'a> SliceIterator<'a> {
    fn new(slice: &'a [Cell], step: usize) -> Self {
        Self {
            slice,
            step,
            pos: (0..slice.len()).step_by(step),
        }
    }
}

impl<'a> Iterator for SliceIterator<'a> {
    type Item = &'a [Cell];

    fn next(&mut self) -> Option<Self::Item> {
        self.pos.next().map(|pos| &self.slice[pos..pos + self.step])
    }
}

#[derive(Debug, Clone, Default)]
struct Run {
    start: usize,
    end: usize,
    len: usize,
}

impl Run {
    fn start(&mut self, start: usize) -> Result<bool, SolverError> {
        if self.end + 1 < start + self.len {
            Err(SolverError::Conflict)
        } else if self.start < start {
            self.start = start;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn end(&mut self, end: usize) -> Result<bool, SolverError> {
        if end + 1 < self.start + self.len {
            Err(SolverError::Conflict)
        } else if end < self.end {
            self.end = end;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[repr(transparent)]
struct Cells(Vec<Cell>);

impl Cells {
    fn set(&mut self, index: usize, cell: Cell) -> Result<bool, SolverError> {
        if self.0.len() <= index {
            return Ok(false);
        }

        let data = &mut self.0[index];
        if *data == Cell::Unknown {
            *data = cell;
            Ok(true)
        } else if *data == cell {
            Ok(false)
        } else {
            Err(SolverError::Conflict)
        }
    }
}

impl Deref for Cells {
    type Target = [Cell];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Cells {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone)]
struct Line {
    index: usize,
    runs: Vec<Run>,
    solved: bool,
}

impl Line {
    fn new(index: usize, width: usize, clues: &[usize]) -> Self {
        let size = clues.len();
        let mut runs = vec![Run::default(); size];

        runs[0].start = 0;
        runs[size - 1].end = width - 1;

        // Initialize starting point for each run
        let mut start = clues[0] + 1;
        for j in 1..size {
            runs[j].start = start;
            start += clues[j] + 1;
        }

        // Initialize ending point for each run
        let mut end = if clues[size - 1] < width - 1 {
            width - clues[size - 1] - 2
        } else {
            width
        };
        for j in (0..size - 1).rev() {
            runs[j].end = end;
            end -= clues[j] - 1;
        }

        for j in 0..size {
            runs[j].len = clues[j];
        }

        Self {
            index,
            runs,
            solved: false,
        }
    }

    fn is_solved(&self, cells: &[Cell]) -> bool {
        let mut pos = 0;
        let len = cells.len();

        for run in &self.runs {
            // Ignore leading empty spaces
            while pos < len && cells[pos] == Cell::Empty {
                pos += 1;
            }

            for _ in 0..run.len {
                // Out of bounds
                if len <= pos {
                    return false;
                }

                if cells[pos] != Cell::Filled {
                    return false;
                }

                pos += 1;
            }
        }

        // Check trailing empty spaces
        while pos < len {
            if cells[pos] != Cell::Empty {
                return false;
            }

            pos += 1;
        }

        true
    }

    fn solve(&mut self, cells: &mut Cells) -> Result<bool, SolverError> {
        let mut progress = false;
        let line_len = cells.len();
        let run_len = self.runs.len();

        // ---- PART 1 ----
        // Rule 1.1
        for &Run { start, end, len } in self.runs.iter() {
            for k in end + 1 - len..start + len {
                if cells.set(k, Cell::Filled)? {
                    progress = true;
                }
            }
        }

        // Rule 1.2
        let first_start = self.runs[0].start;
        let last_end = self.runs[run_len - 1].end;
        for j in 0..line_len {
            if (j < first_start || last_end < j) && cells.set(j, Cell::Empty)? {
                progress = true;
            }
        }
        for j in 0..run_len - 1 {
            let current_end = self.runs[j].end;
            let next_start = self.runs[j + 1].start;
            for k in current_end + 1..next_start {
                if cells.set(k, Cell::Empty)? {
                    progress = true;
                }
            }
        }

        // Rule 1.3
        for (
            j,
            &Run {
                start: cur_start,
                end: cur_end,
                ..
            },
        ) in self.runs.iter().enumerate()
        {
            if 1 <= cur_start && cells[cur_start] == Cell::Filled {
                let length1 = self.runs.iter().take(j).all(|&Run { start, end, len }| {
                    cur_start < start || end < cur_start || len == 1
                });
                if length1 && cells.set(cur_start - 1, Cell::Empty)? {
                    progress = true;
                }
            }

            if cur_end + 1 < line_len && cells[cur_end] == Cell::Filled {
                let length1 = self
                    .runs
                    .iter()
                    .skip(j + 1)
                    .all(|&Run { start, end, len }| cur_end < start || end < cur_end || len == 1);
                if length1 && cells.set(cur_end + 1, Cell::Empty)? {
                    progress = true;
                }
            }
        }

        // Rule 1.4
        for j in 1..line_len - 1 {
            if cells[j - 1] == Cell::Filled
                && cells[j] == Cell::Unknown
                && cells[j + 1] == Cell::Filled
            {
                let mut new_len = 1;
                for &cell in cells.iter().take(j).rev() {
                    if cell != Cell::Filled {
                        break;
                    }
                    new_len += 1;
                }
                for &cell in cells.iter().take(line_len).skip(j + 1) {
                    if cell != Cell::Filled {
                        break;
                    }
                    new_len += 1;
                }

                let mut max_len = 0;
                for &Run { start, end, len } in self.runs.iter() {
                    if start < j && j < end && max_len < len {
                        max_len = len;
                    }
                }

                if max_len < new_len && cells.set(j, Cell::Empty)? {
                    progress = true;
                }
            }
        }

        // Rule 1.5
        for j in 1..line_len {
            if cells[j - 1] != Cell::Filled && cells[j] == Cell::Filled {
                let mut min_len = line_len + 1;
                for &Run { start, end, len } in self.runs.iter() {
                    if start <= j && j <= end && len < min_len {
                        min_len = len;
                    }
                }

                if min_len <= line_len {
                    let mut empty = j;
                    while j < empty + min_len && 0 < empty && cells[empty] != Cell::Empty {
                        empty -= 1;
                    }
                    if j < empty + min_len {
                        for k in j + 1..empty + min_len {
                            if cells.set(k, Cell::Filled)? {
                                progress = true;
                            }
                        }
                    }

                    let mut empty = j + 1;
                    while empty <= j + min_len && empty < line_len && cells[empty] != Cell::Empty {
                        empty += 1;
                    }
                    if empty < j + min_len {
                        for k in empty - min_len..j {
                            if cells.set(k, Cell::Filled)? {
                                progress = true;
                            }
                        }
                    }
                }

                let mut new_len = 0;
                let mut new_start = j;
                let mut new_end = j;
                while 0 < new_start && cells[new_start] == Cell::Filled {
                    new_len += 1;
                    new_start -= 1;
                }
                while new_end < line_len && cells[new_end] == Cell::Filled {
                    new_len += 1;
                    new_end += 1;
                }

                let same_len = self
                    .runs
                    .iter()
                    .all(|&Run { start, end, len }| j < start || end < j || len == new_len - 1);
                if same_len {
                    if cells.set(new_start, Cell::Empty)? {
                        progress = true;
                    }
                    if cells.set(new_end, Cell::Empty)? {
                        progress = true;
                    }
                }
            }
        }

        // ---- PART 2 ----
        // Rule 2.1
        for j in 1..run_len {
            let Run { start, len, .. } = self.runs[j - 1];
            let current = &mut self.runs[j];
            if current.start <= start && current.start(start + len + 1)? {
                progress = true;
            }
        }
        for j in 0..run_len - 1 {
            let Run { end, len, .. } = self.runs[j + 1];
            let current = &mut self.runs[j];
            if end <= current.end && current.end(end - len - 1)? {
                progress = true;
            }
        }

        // Rule 2.2
        for run in self.runs.iter_mut() {
            if 0 < run.start && cells[run.start - 1] == Cell::Filled && run.start(run.start + 1)? {
                progress = true;
            }
            if run.end + 1 < line_len
                && cells[run.end + 1] == Cell::Filled
                && run.end(run.end - 1)?
            {
                progress = true;
            }
        }

        // Rule 2.3
        for j in 1..run_len - 1 {
            let Run { end: prev_end, .. } = self.runs[j - 1];
            let Run {
                start: next_start, ..
            } = self.runs[j + 1];
            let run = &mut self.runs[j];
            let mut seg_start = run.start;
            let mut seg_end = seg_start - 1;

            for (k, &cell) in cells.iter().enumerate().take(run.end + 1).skip(run.start) {
                if cell == Cell::Filled {
                    seg_end = k;
                } else {
                    if seg_start + run.len < seg_end + 1 {
                        if seg_end <= prev_end
                            && seg_start < next_start
                            && run.start(seg_end + 2)?
                        {
                            progress = true;
                        }

                        if next_start <= seg_start
                            && prev_end < seg_end
                            && run.end(seg_start - 2)?
                        {
                            progress = true;
                        }
                    }

                    seg_start = k + 1;
                    seg_end = seg_start - 1;
                }
            }
        }

        // ---- PART 3 ----
        // Rule 3.1
        for j in 0..run_len {
            let prev_end = if j == 0 { 0 } else { self.runs[j - 1].end + 1 };
            let next_start = if j == run_len - 1 {
                line_len
            } else {
                self.runs[j + 1].start
            };

            let mut start_cell = prev_end;
            while start_cell < next_start && cells[start_cell] != Cell::Filled {
                start_cell += 1;
            }

            let mut end_cell = next_start - 1;
            while prev_end < end_cell && cells[end_cell] != Cell::Filled {
                end_cell -= 1;
            }

            let run = &mut self.runs[j];
            let Run {
                start, end, len, ..
            } = *run;
            if start_cell <= end_cell && end_cell < start_cell + len {
                let u = start_cell + len - end_cell - 1;
                for k in start_cell + 1..end_cell {
                    if cells.set(k, Cell::Filled)? {
                        progress = true;
                    }
                }

                if u + start < start_cell && run.start(start_cell - u)? {
                    progress = true;
                }
                if end_cell + u < end && run.end(end_cell + u)? {
                    progress = true;
                }
            }
        }

        // Rule 3.2
        for run in self.runs.iter_mut() {
            let mut seg_len = 0;
            let mut index = run.start;
            for (k, &cell) in cells.iter().enumerate().take(run.end + 1).skip(run.start) {
                if cell != Cell::Empty {
                    seg_len += 1;
                }
                if cell == Cell::Empty || k == run.end {
                    if run.len <= seg_len {
                        if run.start(index)? {
                            progress = true;
                        }
                    } else {
                        seg_len = 0;
                        index = k + 1;
                    }
                }
            }

            let mut seg_len = 0;
            let mut index = run.end;
            for k in (run.start..=run.end).rev() {
                if cells[k] != Cell::Empty {
                    seg_len += 1;
                }
                if cells[k] == Cell::Empty || k == run.start {
                    if run.len <= seg_len {
                        if run.end(index)? {
                            progress = true;
                        }
                    } else if k == 0 {
                        break;
                    } else {
                        seg_len = 0;
                        index = k - 1;
                    }
                }
            }
        }

        // Rule 3.3-1
        for j in 0..run_len {
            let Run { start, len, .. } = self.runs[j];
            if cells[start] == Cell::Filled && (j == 0 || self.runs[j - 1].end < start) {
                for k in start + 1..start + len {
                    if cells.set(k, Cell::Filled)? {
                        progress = true;
                    }
                }
                if 1 <= start && cells.set(start - 1, Cell::Empty)? {
                    progress = true;
                }
                if start + len < line_len && cells.set(start + len, Cell::Empty)? {
                    progress = true;
                }

                self.runs[j].end = start + len - 1;

                if j < run_len - 1 {
                    let next_run = &mut self.runs[j + 1];
                    if next_run.start < start + len && next_run.start(start + len + 1)? {
                        progress = true;
                    }
                }

                if 0 < j {
                    let prev_run = &mut self.runs[j - 1];
                    if start < prev_run.end + 2 && prev_run.end(start - 2)? {
                        progress = true;
                    }
                }
            }
        }

        // Rule 3.3-2
        for j in 0..run_len {
            let Run { start, end, .. } = self.runs[j];
            let mut filled = start;
            while filled < end && cells[filled] != Cell::Filled {
                filled += 1;
            }

            let mut empty = filled;
            while empty <= end && cells[empty] != Cell::Empty {
                empty += 1;
            }

            if (j == 0 || self.runs[j - 1].end < start)
                && empty < end
                && filled < empty
                && self.runs[j].end(empty - 1)?
            {
                progress = true;
            }
        }

        // Rule 3.3-3
        for j in 0..run_len {
            let Run { start, end, len } = self.runs[j];

            if j == 0 || self.runs[j - 1].end < start {
                let mut filled = start;
                while filled < end && cells[filled] != Cell::Filled {
                    filled += 1;
                }

                let mut index = filled;
                while index <= end && cells[index] == Cell::Filled {
                    index += 1;
                }

                index += 1;
                let mut k = index;
                while k <= end {
                    if cells[k] != Cell::Filled || k == end {
                        if filled + len < k {
                            if self.runs[j].end(index - 2)? {
                                progress = true;
                            }
                            k = end + 1;
                        }
                        index = k + 1;
                    }
                    k += 1;
                }
            }
        }

        Ok(progress)
    }
}

#[derive(Debug, Clone)]
pub struct Solver {
    rows: Vec<Line>,
    cols: Vec<Line>,
    row: usize,
    index: Option<usize>,
    dfs: bool,
}

impl Solver {
    pub fn new(puz: &Puzzle) -> Self {
        Self {
            rows: puz
                .rows
                .iter()
                .enumerate()
                .map(|(i, clues)| Line::new(i, puz.cols.len(), clues))
                .collect(),
            cols: puz
                .cols
                .iter()
                .enumerate()
                .map(|(i, clues)| Line::new(i, puz.rows.len(), clues))
                .collect(),
            row: 0,
            index: None,
            dfs: true,
        }
    }

    pub fn set_dfs(&mut self, dfs: bool) {
        self.dfs = dfs;
    }

    pub fn solve(&mut self) -> Result<Solution, SolverError> {
        self.solve_internal(Solution::new(self.cols.len(), self.rows.len()))
    }

    fn solve_internal(&mut self, mut sol: Solution) -> Result<Solution, SolverError> {
        for Line {
            index: i,
            runs,
            solved,
        } in &mut self.rows
        {
            if 1 < runs.len() || runs.len() == 1 && runs[0].len != 0 {
                continue;
            }

            for j in 0..sol.width {
                sol.set(*i, j, Cell::Empty)?;
            }
            *solved = true;
        }
        for Line {
            index: j,
            runs,
            solved,
        } in &mut self.cols
        {
            if 1 < runs.len() || runs.len() == 1 && runs[0].len != 0 {
                continue;
            }

            for i in 0..sol.height {
                sol.set(i, *j, Cell::Empty)?;
            }
            *solved = true;
        }

        loop {
            let pair = self.solve_step(sol)?;
            sol = pair.1;
            if !pair.0 {
                break;
            }
        }
        if sol.filled() || !self.dfs {
            return Ok(sol);
        }

        let mut row = self.row;
        let mut index = if let Some(index) = self.index {
            index + 1
        } else {
            0
        };
        if self.rows[row].runs.len() <= index {
            row += 1;
            index = 0;
        }
        if sol.height <= row {
            return Err(SolverError::NoSolution);
        }

        let Run { start, end, len } = self.rows[row].runs[index];
        'outer: for run_start in start..=end + 1 - len {
            let mut solver = self.clone();
            solver.rows.iter_mut().for_each(|line| line.solved = false);
            solver.cols.iter_mut().for_each(|line| line.solved = false);
            solver.row = row;
            solver.index = Some(index);

            let run_end = if 0 < len { run_start + len - 1 } else { end };

            // Set run at a particular location
            let new_run = &mut solver.rows[row].runs[index];
            new_run.start = run_start;
            new_run.end = run_end;

            sol = Solution::new(solver.cols.len(), solver.rows.len());

            // Update all cells in the run region
            for i in run_start..=run_end {
                if sol.set(row, i, Cell::Filled).is_err() {
                    continue 'outer;
                }
            }
            if 0 < run_start && sol.set(row, run_end + 1, Cell::Empty).is_err() {
                continue;
            }
            if run_end <= sol.width && sol.set(row, run_end + 1, Cell::Empty).is_err() {
                continue;
            }

            let res = solver.solve_internal(sol);
            if res.is_ok() {
                return res;
            }
        }

        Err(SolverError::NoSolution)
    }

    fn solve_step(&mut self, mut sol: Solution) -> Result<(bool, Solution), SolverError> {
        let mut progress = false;

        // ======================
        // ======== ROWS ========
        // ======================
        let rows = self
            .rows
            .iter_mut()
            .zip(sol.rows())
            .filter_map(|(line, cells)| {
                if line.solved {
                    return None;
                }

                let mut cells = Cells(cells.to_vec());
                match line.solve(&mut cells) {
                    Ok(false) => None,
                    Ok(true) => {
                        line.solved = line.is_solved(&cells);
                        Some(Ok((line, cells)))
                    }
                    Err(e) => Some(Err(e)),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (Line { index, .. }, cells) in rows {
            for (j, cell) in cells.iter().enumerate() {
                if sol.set(*index, j, *cell)? {
                    progress = true;
                }
            }
        }

        // =========================
        // ======== COLUMNS ========
        // =========================
        let cols = self
            .cols
            .iter_mut()
            .zip(sol.columns())
            .filter_map(|(line, cells)| {
                if line.solved {
                    return None;
                }

                let mut cells = Cells(cells.to_vec());
                match line.solve(&mut cells) {
                    Ok(false) => None,
                    Ok(true) => {
                        line.solved = line.is_solved(&cells);
                        Some(Ok((line, cells)))
                    }
                    Err(e) => Some(Err(e)),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (Line { index, .. }, cells) in cols {
            for (i, cell) in cells.iter().enumerate() {
                if sol.set(i, *index, *cell)? {
                    progress = true;
                }
            }
        }

        Ok((progress, sol))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_puzzle_validate() {
        assert!(
            Puzzle {
                rows: vec![],
                cols: vec![],
            }
            .validate()
            .is_ok()
        );

        assert!(
            Puzzle {
                rows: vec![vec![0]],
                cols: vec![vec![0]],
            }
            .validate()
            .is_ok()
        );

        assert!(matches!(
            Puzzle {
                rows: vec![vec![2]],
                cols: vec![vec![0]],
            }
            .validate(),
            Err(PuzzleError::Row(1))
        ));

        assert!(matches!(
            Puzzle {
                rows: vec![vec![0]],
                cols: vec![vec![2]],
            }
            .validate(),
            Err(PuzzleError::Column(1))
        ));
    }
}
