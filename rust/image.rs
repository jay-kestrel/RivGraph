// Modified RivGraph image helpers; copyright 2019 Triad National Security, LLC.
// LANL contract 89233218CNA000001. Full notice and BSD-3 terms: ../LICENSE.txt.
#[derive(Clone)]
pub struct Image {
    pub data: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}
impl Image {
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0; rows * cols],
            rows,
            cols,
        }
    }
    pub fn edge(&self, i: usize) -> bool {
        i < self.cols
            || i >= self.data.len() - self.cols
            || i.is_multiple_of(self.cols)
            || i % self.cols == self.cols - 1
    }
    pub fn neighbors(&self, i: usize) -> Vec<usize> {
        let (r, c) = (i / self.cols, i % self.cols);
        let mut out = Vec::with_capacity(8);
        for dr in -1..=1isize {
            for dc in -1..=1isize {
                let (y, x) = (r as isize + dr, c as isize + dc);
                if (dr != 0 || dc != 0)
                    && y >= 0
                    && x >= 0
                    && y < self.rows as isize
                    && x < self.cols as isize
                {
                    let n = y as usize * self.cols + x as usize;
                    if self.data[n] != 0 {
                        out.push(n);
                    }
                }
            }
        }
        out
    }
    pub fn four(&self, i: usize) -> Vec<usize> {
        self.neighbors(i)
            .into_iter()
            .filter(|&n| self.is_four(i, n))
            .collect()
    }
    pub fn is_four(&self, i: usize, n: usize) -> bool {
        let d = i.abs_diff(n);
        d == 1 || d == self.cols
    }
    pub fn components(&self, four: bool) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.data.len()];
        let mut result = Vec::new();
        for i in 0..self.data.len() {
            if seen[i] || self.data[i] == 0 {
                continue;
            }
            seen[i] = true;
            let mut pixels = vec![i];
            let mut at = 0;
            while at < pixels.len() {
                let p = pixels[at];
                for n in self.neighbors(p) {
                    if !seen[n] && (!four || self.is_four(p, n)) {
                        seen[n] = true;
                        pixels.push(n);
                    }
                }
                at += 1;
            }
            pixels.sort_unstable();
            result.push(pixels);
        }
        result
    }
    pub fn connectivity(&self) -> Self {
        let mut out = Self::zero(self.rows, self.cols);
        for i in 0..self.data.len() {
            if self.data[i] == 0 {
                continue;
            }
            let (r, c) = (i / self.cols, i % self.cols);
            let mirror = |x: isize, size: usize| -> usize {
                if size == 1 {
                    0
                } else if x < 0 {
                    1
                } else if x >= size as isize {
                    size - 2
                } else {
                    x as usize
                }
            };
            for dr in -1..=1isize {
                for dc in -1..=1isize {
                    if dr != 0 || dc != 0 {
                        let n = mirror(r as isize + dr, self.rows) * self.cols
                            + mirror(c as isize + dc, self.cols);
                        out.data[i] += u8::from(self.data[n] != 0);
                    }
                }
            }
        }
        out
    }
    pub fn axes(&self, i: usize) -> u8 {
        if self.edge(i) || self.data[i] != 1 {
            return 0;
        }
        [1, self.cols, self.cols + 1, self.cols - 1]
            .iter()
            .map(|&d| u8::from(self.data[i + d] == 1 || self.data[i - d] == 1))
            .sum()
    }
    pub fn edges(&self) -> Vec<usize> {
        // Preserve edge_coords' insertion sequence (including duplicates).
        (0..self.rows)
            .map(|r| r * self.cols)
            .chain((0..self.cols).map(|c| (self.rows - 1) * self.cols + c))
            .chain((0..self.rows).rev().map(|r| r * self.cols + self.cols - 1))
            .chain((0..self.cols).rev())
            .collect()
    }
}
