# Nonogram solver

## Usage

```console
$ nonogram --help
Nonogram solver

Usage: nonogram [OPTIONS] <PATH>

Arguments:
  <PATH>  Path to a file that contains clues

Options:
  -D, --disable-dfs  Disable depth-first search
  -h, --help         Print help
  -V, --version      Print version
```

```console
$ nonogram example.txt
+-----+
|XX XX|
|     |
| X X |
|X   X|
| XXX |
+-----+
```

## File format

This program accepts file format for Andrew Makhorin's pbnsol nonogram solver.

```
* this is a comment
* row clues, top to bottom, left to right in each row
2 2
0
1 1
1 1
3
&
* `&` indicates end of row clues
* column clues, left to right, top to bottom in each column
1 1
1 1 1
1
1 1 1
1 1
```
