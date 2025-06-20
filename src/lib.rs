use std::io;
use std::iter::StepBy;
use std::num::ParseIntError;
use std::ops::RangeInclusive;

#[derive(Debug, thiserror::Error)]
pub enum PuzzleError {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    ParseInt(#[from] ParseIntError),
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
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Cell>,
}

impl Solution {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![Cell::Unknown; width * height],
        }
    }

    fn set(&mut self, row: usize, col: usize, cell: Cell) -> Result<bool, SolverError> {
        if self.height <= row || self.width <= col {
            return Ok(false);
        }

        let index = row * self.width + col;
        let data = &mut self.cells[index];
        if *data == Cell::Unknown {
            *data = cell;
            Ok(true)
        } else if *data == cell {
            Ok(false)
        } else {
            Err(SolverError::Conflict)
        }
    }

    fn filled(&self) -> bool {
        self.cells.iter().all(|&c| c != Cell::Unknown)
    }

    pub fn check(&self, puz: &Puzzle) -> bool {
        // Check row constraints
        for row in 0..self.height {
            let mut col = 0;
            for i in 0..puz.rows[row].len() {
                // Ignore leading empty spaces
                while col < self.width && self.cells[row * self.width + col] == Cell::Empty {
                    col += 1;
                }

                for _ in 0..puz.rows[row][i] {
                    // Out of bounds
                    if self.width <= col {
                        return false;
                    }
                    if self.cells[row * self.width + col] != Cell::Filled {
                        return false;
                    }
                    col += 1;
                }
            }

            // Check trailing empty spaces
            while col < self.width {
                if self.cells[row * self.width + col] != Cell::Empty {
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
                while row < self.height && self.cells[row * self.width + col] == Cell::Empty {
                    row += 1;
                }

                for _ in 0..puz.cols[col][i] {
                    // Out of bounds
                    if self.height <= row {
                        return false;
                    }
                    if self.cells[row * self.width + col] != Cell::Filled {
                        return false;
                    }
                    row += 1;
                }
            }

            // Check trailing empty spaces
            while row < self.height {
                if self.cells[row * self.width + col] != Cell::Empty {
                    return false;
                }
                row += 1;
            }
        }

        true
    }

    pub fn rows(&self) -> impl Iterator<Item = &[Cell]> {
        RowIterator {
            slice: self.cells.as_slice(),
            lower: 0,
            step: (self.width..=self.width * self.height).step_by(self.width),
        }
    }
}

struct RowIterator<'a> {
    slice: &'a [Cell],
    lower: usize,
    step: StepBy<RangeInclusive<usize>>,
}

impl<'a> Iterator for RowIterator<'a> {
    type Item = &'a [Cell];

    fn next(&mut self) -> Option<Self::Item> {
        match self.step.next() {
            None => None,
            Some(upper) => {
                let lower = self.lower;
                self.lower = upper;
                Some(&self.slice[lower..upper])
            }
        }
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

#[derive(Debug, Clone)]
pub struct Solver {
    row_runs: Vec<Vec<Run>>,
    col_runs: Vec<Vec<Run>>,
    row: usize,
    index: Option<usize>,
    dfs: bool,
}

impl Solver {
    pub fn new(puz: &Puzzle) -> Self {
        fn mapper(width: usize, clues: &[usize]) -> Vec<Run> {
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

            runs
        }

        Self {
            row_runs: puz
                .rows
                .iter()
                .map(|clues| mapper(puz.cols.len(), clues))
                .collect(),
            col_runs: puz
                .cols
                .iter()
                .map(|clues| mapper(puz.rows.len(), clues))
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
        self.solve_internal(Solution::new(self.col_runs.len(), self.row_runs.len()))
    }

    fn solve_internal(&mut self, mut sol: Solution) -> Result<Solution, SolverError> {
        loop {
            if !self.solve_step(&mut sol)? {
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
        if self.row_runs[row].len() <= index {
            row += 1;
            index = 0;
        }
        if sol.height <= row {
            return Err(SolverError::NoSolution);
        }

        let Run { start, end, len } = self.row_runs[row][index];
        'outer: for run_start in start..=end + 1 - len {
            let mut solver = self.clone();
            solver.row = row;
            solver.index = Some(index);

            let new_run = &mut solver.row_runs[row][index];
            let run_end = run_start + len - 1;

            // Set run at particular location
            new_run.start = run_start;
            new_run.end = run_end;

            sol = Solution::new(self.col_runs.len(), self.row_runs.len());

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

    fn solve_step(&mut self, sol: &mut Solution) -> Result<bool, SolverError> {
        let mut progress = false;
        let height = self.row_runs.len();
        let width = self.col_runs.len();

        // ======================
        // ======== ROWS ========
        // ======================
        for i in 0..height {
            let runs = &mut self.row_runs[i];
            let size = runs.len();

            // ---- PART 1 ----
            // Rule 1.1
            for Run { start, end, len } in runs.iter() {
                for k in end + 1 - len..start + len {
                    if sol.set(i, k, Cell::Filled)? {
                        progress = true;
                    }
                }
            }

            // Rule 1.2
            let first_start = runs[0].start;
            let last_end = runs[size - 1].end;
            for j in 0..width {
                if (j < first_start || last_end < j) && sol.set(i, j, Cell::Empty)? {
                    progress = true;
                }
            }
            for j in 0..size - 1 {
                let current_end = runs[j].end;
                let next_start = runs[j + 1].start;
                for k in current_end + 1..next_start {
                    if sol.set(i, k, Cell::Empty)? {
                        progress = true;
                    }
                }
            }

            // Rule 1.3
            for j in 0..size {
                let Run {
                    start: cur_start,
                    end: cur_end,
                    ..
                } = runs[j];

                if 1 <= cur_start && sol.cells[i * width + cur_start] == Cell::Filled {
                    let mut length1 = true;
                    for &Run { start, end, len } in &runs[0..j] {
                        if start <= cur_start && cur_start <= end && len != 1 {
                            length1 = false;
                            break;
                        }
                    }

                    if length1 && sol.set(i, cur_start - 1, Cell::Empty)? {
                        progress = true;
                    }
                }

                if cur_end + 1 < width && sol.cells[i * width + cur_end] == Cell::Filled {
                    let mut length1 = true;
                    for &Run { start, end, len } in &runs[j + 1..size] {
                        if start <= cur_end && cur_end <= end && len != 1 {
                            length1 = false;
                            break;
                        }
                    }

                    if length1 && sol.set(i, cur_end + 1, Cell::Empty)? {
                        progress = true;
                    }
                }
            }

            // Rule 1.4
            for j in 1..width - 1 {
                if sol.cells[i * width + (j - 1)] == Cell::Filled
                    && sol.cells[i * width + j] == Cell::Unknown
                    && sol.cells[i * width + (j + 1)] == Cell::Filled
                {
                    let mut new_len = 1;
                    for k in (0..j).rev() {
                        if sol.cells[i * width + k] != Cell::Filled {
                            break;
                        }
                        new_len += 1;
                    }
                    for k in j + 1..width {
                        if sol.cells[i * width + k] != Cell::Filled {
                            break;
                        }
                        new_len += 1;
                    }

                    let mut max_len = 0;
                    for &Run { start, end, len } in &runs[0..size] {
                        if start < j && j < end && max_len < len {
                            max_len = len;
                        }
                    }

                    if max_len < new_len && sol.set(i, j, Cell::Empty)? {
                        progress = true;
                    }
                }
            }

            // Rule 1.5
            for j in 1..width {
                if (sol.cells[i * width + (j - 1)] == Cell::Empty
                    || sol.cells[i * width + (j - 1)] == Cell::Unknown)
                    && sol.cells[i * width + j] == Cell::Filled
                {
                    let mut min_len = width + 1;
                    for &Run { start, end, len } in &runs[0..size] {
                        if start <= j && j <= end && len < min_len {
                            min_len = len;
                        }
                    }

                    if min_len <= width {
                        let mut empty = j;
                        while j < empty + min_len
                            && 0 < empty
                            && sol.cells[i * width + empty] != Cell::Empty
                        {
                            empty -= 1;
                        }
                        if j < empty + min_len {
                            for k in j + 1..empty + min_len {
                                if sol.set(i, k, Cell::Filled)? {
                                    progress = true;
                                }
                            }
                        }

                        let mut empty = j + 1;
                        while empty <= j + min_len
                            && empty < width
                            && sol.cells[i * width + empty] != Cell::Empty
                        {
                            empty += 1;
                        }
                        if empty < j + min_len {
                            for k in empty - min_len..j {
                                if sol.set(i, k, Cell::Filled)? {
                                    progress = true;
                                }
                            }
                        }
                    }

                    let mut new_len = 0;
                    let mut new_start = j;
                    let mut new_end = j;
                    while 0 < new_start && sol.cells[i * width + new_start] == Cell::Filled {
                        new_len += 1;
                        new_start -= 1;
                    }
                    while new_end < width && sol.cells[i * width + new_end] == Cell::Filled {
                        new_len += 1;
                        new_end += 1;
                    }

                    let mut same_len = true;
                    for &Run { start, end, len } in &runs[0..size] {
                        if start <= j && j <= end && len != new_len - 1 {
                            same_len = false;
                            break;
                        }
                    }

                    if same_len {
                        if sol.set(i, new_start, Cell::Empty)? {
                            progress = true;
                        }
                        if sol.set(i, new_end, Cell::Empty)? {
                            progress = true;
                        }
                    }
                }
            }

            // ---- PART 2 ----
            // Rule 2.1
            for j in 1..size {
                let Run { start, len, .. } = runs[j - 1];
                let current = &mut runs[j];
                if current.start <= start {
                    let start = start + len + 1;
                    if current.start(start)? {
                        progress = true;
                    }
                }
            }
            for j in 0..size - 1 {
                let Run { end, len, .. } = runs[j + 1];
                let current = &mut runs[j];
                if end <= current.end && runs[j].end(end - len - 1)? {
                    progress = true;
                }
            }

            // Rule 2.2
            for run in runs.iter_mut() {
                if 0 < run.start {
                    let prev_cell = sol.cells[i * sol.width + run.start - 1];
                    if prev_cell == Cell::Filled && run.start(run.start + 1)? {
                        progress = true;
                    }
                }
                if run.end + 1 < sol.width {
                    let next_cell = sol.cells[i * sol.width + run.end + 1];
                    if next_cell == Cell::Filled && run.end(run.end - 1)? {
                        progress = true;
                    }
                }
            }

            // Rule 2.3
            for j in 1..size - 1 {
                let Run { end: prev_end, .. } = runs[j - 1];
                let Run {
                    start: next_start, ..
                } = runs[j + 1];
                let run = &mut runs[j];
                let mut seg_start = run.start;
                let mut seg_end = seg_start - 1;
                for k in run.start..=run.end {
                    if sol.cells[i * sol.width + k] == Cell::Filled {
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
            for j in 0..size {
                let prev_end = if j == 0 { -1 } else { runs[j - 1].end as isize };
                let next_start = if j == size - 1 {
                    width as isize
                } else {
                    runs[j + 1].start as isize
                };
                let mut start_cell = prev_end + 1;
                while start_cell < next_start
                    && sol.cells[i * sol.width + start_cell as usize] != Cell::Filled
                {
                    start_cell += 1;
                }
                let mut end_cell = next_start - 1;
                while prev_end < end_cell
                    && sol.cells[i * sol.width + end_cell as usize] != Cell::Filled
                {
                    end_cell -= 1;
                }

                let run = &mut runs[j];
                if start_cell <= end_cell && end_cell < start_cell + run.len as isize {
                    let u = start_cell + run.len as isize - end_cell - 1;
                    for k in start_cell + 1..end_cell {
                        if sol.set(i, k as usize, Cell::Filled)? {
                            progress = true;
                        }
                    }

                    if u + (run.start as isize) < start_cell
                        && run.start((start_cell - u) as usize)?
                    {
                        progress = true;
                    }
                    if end_cell + u < run.end as isize && run.end((end_cell + u) as usize)? {
                        progress = true;
                    }
                }
            }

            // Rule 3.2
            for run in runs.iter_mut() {
                let mut seg_len = 0;
                let mut index = run.start;
                for k in run.start..=run.end {
                    if sol.cells[i * sol.width + k] != Cell::Empty {
                        seg_len += 1;
                    }
                    if sol.cells[i * sol.width + k] == Cell::Empty || k == run.end {
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
                seg_len = 0;
                index = run.end;
                for k in (run.start..=run.end).rev() {
                    if sol.cells[i * sol.width + k] != Cell::Empty {
                        seg_len += 1;
                    }
                    if sol.cells[i * sol.width + k] == Cell::Empty || k == run.start {
                        if run.len <= seg_len {
                            if run.end(index)? {
                                progress = true;
                            }
                        } else {
                            seg_len = 0;
                            index = k - 1;
                        }
                    }
                }
            }

            // Rule 3.3-1
            for j in 0..size {
                let Run { start, len, .. } = runs[j];
                if sol.cells[i * sol.width + start] == Cell::Filled
                    && (j == 0 || runs[j - 1].end < start)
                {
                    for k in start + 1..start + len {
                        if sol.set(i, k, Cell::Filled)? {
                            progress = true;
                        }
                    }
                    if 1 <= start && sol.set(i, start - 1, Cell::Empty)? {
                        progress = true;
                    }
                    if start + len < width && sol.set(i, start + len, Cell::Empty)? {
                        progress = true;
                    }
                    runs[j].end = start + len - 1;
                    if j < size - 1
                        && runs[j + 1].start < start + len
                        && runs[j + 1].start(start + len + 1)?
                    {
                        progress = true;
                    }
                    if 0 < j && start < runs[j - 1].end + 2 && runs[j - 1].end(start - 2)? {
                        progress = true;
                    }
                }
            }

            // Rule 3.3-2
            for j in 0..size {
                let Run { start, end, .. } = runs[j];
                let mut filled = start;
                while filled < end && sol.cells[i * sol.width + filled] != Cell::Filled {
                    filled += 1;
                }
                let mut empty = filled;
                while empty <= end && sol.cells[i * sol.width + empty] != Cell::Empty {
                    empty += 1;
                }
                if (j == 0 || runs[j - 1].end < start)
                    && empty < end
                    && filled < empty
                    && runs[j].end(empty - 1)?
                {
                    progress = true;
                }
            }

            // Rule 3.3-3
            for j in 0..size {
                let Run { start, end, len } = runs[j];

                if j == 0 || runs[j - 1].end < start {
                    let mut filled = start;
                    while filled < end && sol.cells[i * sol.width + filled] != Cell::Filled {
                        filled += 1;
                    }

                    let mut index = filled;
                    while index <= end && sol.cells[i * sol.width + index] == Cell::Filled {
                        index += 1;
                    }

                    index += 1;
                    let mut k = index;
                    while k <= end {
                        if sol.cells[i * sol.width + k] != Cell::Filled || k == end {
                            if filled + len < k {
                                if runs[j].end(index - 2)? {
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
        }

        // =========================
        // ======== COLUMNS ========
        // =========================
        for i in 0..width {
            let runs = &mut self.col_runs[i];
            let size = runs.len();

            // ---- PART 1 ----
            // Rule 1.1
            for Run { start, end, len } in runs.iter() {
                for k in end + 1 - len..start + len {
                    if sol.set(k, i, Cell::Filled)? {
                        progress = true;
                    }
                }
            }

            // Rule 1.2
            let first_start = runs[0].start;
            let last_end = runs[size - 1].end;
            for j in 0..height {
                if (j < first_start || last_end < j) && sol.set(j, i, Cell::Empty)? {
                    progress = true;
                }
            }
            for j in 0..size - 1 {
                let current_end = runs[j].end;
                let next_start = runs[j + 1].start;
                for k in current_end + 1..next_start {
                    if sol.set(k, i, Cell::Empty)? {
                        progress = true;
                    }
                }
            }

            // Rule 1.3
            for j in 0..size {
                let Run {
                    start: cur_start,
                    end: cur_end,
                    ..
                } = runs[j];

                if 1 <= cur_start && sol.cells[cur_start * width + i] == Cell::Filled {
                    let mut length1 = true;
                    for &Run { start, end, len } in &runs[0..j] {
                        if start <= cur_start && cur_start <= end && len != 1 {
                            length1 = false;
                            break;
                        }
                    }

                    if length1 && sol.set(cur_start - 1, i, Cell::Empty)? {
                        progress = true;
                    }
                }

                if cur_end + 1 < height && sol.cells[cur_end * width + i] == Cell::Filled {
                    let mut length1 = true;
                    for &Run { start, end, len } in &runs[j + 1..size] {
                        if start <= cur_end && cur_end <= end && len != 1 {
                            length1 = false;
                            break;
                        }
                    }

                    if length1 && sol.set(cur_end + 1, i, Cell::Empty)? {
                        progress = true;
                    }
                }
            }

            // Rule 1.4
            for j in 1..height - 1 {
                if sol.cells[(j - 1) * width + i] == Cell::Filled
                    && sol.cells[j * width + i] == Cell::Unknown
                    && sol.cells[(j + 1) * width + i] == Cell::Filled
                {
                    let mut new_len = 1;
                    for k in (0..j).rev() {
                        if sol.cells[k * width + i] != Cell::Filled {
                            break;
                        }
                        new_len += 1;
                    }
                    for k in j + 1..height {
                        if sol.cells[k * width + i] != Cell::Filled {
                            break;
                        }
                        new_len += 1;
                    }

                    let mut max_len = 0;
                    for &Run { start, end, len } in &runs[0..size] {
                        if start < j && j < end && max_len < len {
                            max_len = len;
                        }
                    }

                    if max_len < new_len && sol.set(j, i, Cell::Empty)? {
                        progress = true;
                    }
                }
            }

            // Rule 1.5
            for j in 1..height {
                if (sol.cells[(j - 1) * width + i] == Cell::Empty
                    || sol.cells[(j - 1) * width + i] == Cell::Unknown)
                    && sol.cells[j * width + i] == Cell::Filled
                {
                    let mut min_len = height + 1;
                    for &Run { start, end, len } in &runs[0..size] {
                        if start <= j && j <= end && len < min_len {
                            min_len = len;
                        }
                    }

                    if min_len <= height {
                        let mut empty = j;
                        while j < empty + min_len
                            && 0 < empty
                            && sol.cells[empty * width + i] != Cell::Empty
                        {
                            empty -= 1;
                        }
                        if j < empty + min_len {
                            for k in j + 1..empty + min_len {
                                if sol.set(k, i, Cell::Filled)? {
                                    progress = true;
                                }
                            }
                        }

                        let mut empty = j + 1;
                        while empty <= j + min_len
                            && empty < height
                            && sol.cells[empty * width + i] != Cell::Empty
                        {
                            empty += 1;
                        }
                        if empty < j + min_len {
                            for k in empty - min_len..j {
                                if sol.set(k, i, Cell::Filled)? {
                                    progress = true;
                                }
                            }
                        }
                    }

                    let mut new_len = 0;
                    let mut new_start = j;
                    let mut new_end = j;
                    while 0 < new_start && sol.cells[new_start * width + i] == Cell::Filled {
                        new_len += 1;
                        new_start -= 1;
                    }
                    while new_end < height && sol.cells[new_end * width + i] == Cell::Filled {
                        new_len += 1;
                        new_end += 1;
                    }

                    let mut same_len = true;
                    for &Run { start, end, len } in &runs[0..size] {
                        if start <= j && j <= end && len != new_len - 1 {
                            same_len = false;
                            break;
                        }
                    }

                    if same_len {
                        if sol.set(new_start, i, Cell::Empty)? {
                            progress = true;
                        }
                        if sol.set(new_end, i, Cell::Empty)? {
                            progress = true;
                        }
                    }
                }
            }

            // ---- PART 2 ----
            // Rule 2.1
            for j in 1..size {
                let Run { start, len, .. } = runs[j - 1];
                let current = &mut runs[j];
                if current.start <= start {
                    let start = start + len + 1;
                    if current.start(start)? {
                        progress = true;
                    }
                }
            }
            for j in 0..size - 1 {
                let Run { end, len, .. } = runs[j + 1];
                let current = &mut runs[j];
                if end <= current.end && runs[j].end(end - len - 1)? {
                    progress = true;
                }
            }

            // Rule 2.2
            for run in runs.iter_mut() {
                if 0 < run.start {
                    let prev_cell = sol.cells[(run.start - 1) * sol.width + i];
                    if prev_cell == Cell::Filled && run.start(run.start + 1)? {
                        progress = true;
                    }
                }
                if run.end + 1 < sol.height {
                    let next_cell = sol.cells[(run.end + 1) * sol.width + i];
                    if next_cell == Cell::Filled && run.end(run.end - 1)? {
                        progress = true;
                    }
                }
            }

            // Rule 2.3
            for j in 1..size - 1 {
                let Run { end: prev_end, .. } = runs[j - 1];
                let Run {
                    start: next_start, ..
                } = runs[j + 1];
                let run = &mut runs[j];
                let mut seg_start = run.start;
                let mut seg_end = seg_start - 1;
                for k in run.start..=run.end {
                    if sol.cells[k * sol.width + i] == Cell::Filled {
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
            for j in 0..size {
                let prev_end = if j == 0 { -1 } else { runs[j - 1].end as isize };
                let next_start = if j == size - 1 {
                    height as isize
                } else {
                    runs[j + 1].start as isize
                };
                let mut start_cell = prev_end + 1;
                while start_cell < next_start
                    && sol.cells[start_cell as usize * sol.width + i] != Cell::Filled
                {
                    start_cell += 1;
                }
                let mut end_cell = next_start - 1;
                while prev_end < end_cell
                    && sol.cells[end_cell as usize * sol.width + i] != Cell::Filled
                {
                    end_cell -= 1;
                }

                let run = &mut runs[j];
                if start_cell <= end_cell && end_cell < start_cell + run.len as isize {
                    let u = start_cell + run.len as isize - end_cell - 1;
                    for k in start_cell + 1..end_cell {
                        if sol.set(k as usize, i, Cell::Filled)? {
                            progress = true;
                        }
                    }

                    if u + (run.start as isize) < start_cell
                        && run.start((start_cell - u) as usize)?
                    {
                        progress = true;
                    }
                    if end_cell + u < run.end as isize && run.end((end_cell + u) as usize)? {
                        progress = true;
                    }
                }
            }

            // Rule 3.2
            for run in runs.iter_mut() {
                let mut seg_len = 0;
                let mut index = run.start;
                for k in run.start..=run.end {
                    if sol.cells[k * sol.width + i] != Cell::Empty {
                        seg_len += 1;
                    }
                    if sol.cells[k * sol.width + i] == Cell::Empty || k == run.end {
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
                seg_len = 0;
                index = run.end;
                for k in (run.start..=run.end).rev() {
                    if sol.cells[k * sol.width + i] != Cell::Empty {
                        seg_len += 1;
                    }
                    if sol.cells[k * sol.width + i] == Cell::Empty || k == run.start {
                        if run.len <= seg_len {
                            if run.end(index)? {
                                progress = true;
                            }
                        } else {
                            seg_len = 0;
                            index = k - 1;
                        }
                    }
                }
            }

            // Rule 3.3-1
            for j in 0..size {
                let Run { start, len, .. } = runs[j];
                if sol.cells[start * sol.width + i] == Cell::Filled
                    && (j == 0 || runs[j - 1].end < start)
                {
                    for k in start + 1..start + len {
                        if sol.set(k, i, Cell::Filled)? {
                            progress = true;
                        }
                    }
                    if 1 <= start && sol.set(start - 1, i, Cell::Empty)? {
                        progress = true;
                    }
                    if start + len < height && sol.set(start + len, i, Cell::Empty)? {
                        progress = true;
                    }
                    runs[j].end = start + len - 1;
                    if j < size - 1
                        && runs[j + 1].start < start + len
                        && runs[j + 1].start(start + len + 1)?
                    {
                        progress = true;
                    }
                    if 0 < j && start < runs[j - 1].end + 2 && runs[j - 1].end(start - 2)? {
                        progress = true;
                    }
                }
            }

            // Rule 3.3-2
            for j in 0..size {
                let Run { start, end, .. } = runs[j];
                let mut filled = start;
                while filled < end && sol.cells[filled * sol.width + i] != Cell::Filled {
                    filled += 1;
                }
                let mut empty = filled;
                while empty <= end && sol.cells[empty * sol.width + i] != Cell::Empty {
                    empty += 1;
                }
                if (j == 0 || runs[j - 1].end < start)
                    && empty < end
                    && filled < empty
                    && runs[j].end(empty - 1)?
                {
                    progress = true;
                }
            }

            // Rule 3.3-3
            for j in 0..size {
                let Run { start, end, len } = runs[j];

                if j == 0 || runs[j - 1].end < start {
                    let mut filled = start;
                    while filled < end && sol.cells[filled * sol.width + i] != Cell::Filled {
                        filled += 1;
                    }

                    let mut index = filled;
                    while index <= end && sol.cells[index * sol.width + i] == Cell::Filled {
                        index += 1;
                    }

                    index += 1;
                    let mut k = index;
                    while k <= end {
                        if sol.cells[k * sol.width + i] != Cell::Filled || k == end {
                            if filled + len < k {
                                if runs[j].end(index - 2)? {
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
        }

        Ok(progress)
    }
}
