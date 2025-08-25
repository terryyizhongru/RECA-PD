## split-mono Scripts

> **Note:** An expanded explanation of these scripts will be added later.

First, split the wav files of Monologue to 10 wav segments, and extract features the same way as the PC-GITA dataset


### `split.py`
Expands each non-header row in a CSV file into **10 rows**, replacing  
`MONOLOGO-NR` with `MONOLOGO-NR_part1` through `MONOLOGO-NR_part10`.

### `rm.py`
Recursively processes all CSV files in the specified directory and its subdirectories,  
keeping only the header row and rows whose **third column** equals **`MONOLOGUE`**.

### `clean.py`
Removes or fixes rows that are incorrectly formatted.
