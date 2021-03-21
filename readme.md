## What is this?

This is a repo containing research code for a parallelized scattered neighborhood search designed to solve Nurikabe puzzles

### What does that mean?

If you stumbled upon this repo, the above probably doesn't mean anything to you - read ["Sequential and Parallel Scattered Variable Neighborhood Search for Solving Nurikabe"](https://www.springer.com/gp/book/9783030696245) first

## Prerequisites

This program _should_ run out-of-the box on most any machine with python 3.7+ installed. I say "should" because there are some idiosyncrasies - if you're using the anaconda distribution of python, you may run into problems importi ng matplotlib (I, to be honest, don't know why). If you're using a "normal" distribution you should be good to go, though you might need to import some common packages (like numpy, pandas, and matplotlib) using pip.

## What do I need to know before I run this?

You should be able to run this using `python3 nurikabeTest.py` and following the on-screen instructions. However, if you want to get a better sense of what's going on "under the hood", there are two booleans (`perfDebug` and `debug`) near the top of `nurukabeTest.py`, and one boolean (`debug`) near the top of `board.py`, all of which you should set to `True`. These enable a kind of "verbose" mode that will add performance and behavior information to the output.

When running the program, you will notice some `.png` files with weird names like `0 0.png` being generated - there are solutions! Some additional `png`s with names like `converg0 0 87278.png` and `8initial Board.png` will also appear - feel free to ignore these.

If you're really into debug data, a file named `boardData.csv` is created for each run. Pro tip: open it in Excel, it makes parsing it way easier.

An important note: this program is incredibly memory intensive. I would recommend, if you're planning on running a full test suite, running it on a machine with _at least_ 64 gb of RAM. If you run it on a machine with <= 8 gb of ram, the swap usage will absolutely thrash your hard disk - you've been warned!

This program's performance is extremely dependent on your machine's core count - it absolutely loves high core counts, and doesn't _really_ care that much about clock speed. If you want high performance, run it on a high-core count Xeon or Epyc CPU, not an overclocked I7.

## How are boards encoded? Can I add my own custom tests?

Boards are encoded using a (fairly confusing) custom format. Consider this board from `nurikabeFull.csv`:

`101,3,0,0,1,2,1,5`

The first value is the board ID (which is used to identify the board in `boardData.csv`), while the second tells us the dimensions of the board. This board, as an example, has id `101` and is `3x3` in size. After the first two values, subsequent entries need to be considered in groups of three. This board has two groups of three: `0,0,1` and `2,1,5`. Each of these triplets consists of coordinates and the value at those coordinates - the first two values in a triplet are the x and y dimensions respectively, while the third value is the numbered white square that can be found at that location. So, in sum, the board denoted by `101,3,0,0,1,2,1,5` is a 3x3 board with ID 101, with a white 1 square at location `0,0` and a white 5 square at location `2,1`.

Any board encoded using this format (once placed into `bigBoards.csv`, `nurikabeFull.csv`, or `nurikabeRep.csv` on a new line) can be processed and (hopefully!) solved by our program.
