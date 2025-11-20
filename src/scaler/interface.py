#!/usr/bin/env python3
"""
bits_split.py

Read a text file whose lines are 0-1 bit-strings.  For every line
    abc…xyz
all but the final bit ("abc…y") become one row of the **state matrix**,
and the final bit ("z") becomes one row of the **observable vector**.

Example
-------
Input file lines:
    000000000
    010101011
    111111110

Output (JSON):
[
  [[false,false,false,false,false,false,false,false],
   [false,false,false,false,false,false,false,false],
   [true,true,true,true,true,true,true,true]],
  [[false],[true],[false]]
]
"""

import sys
import json
from pathlib import Path


def parse_line(line: str):
    """Return (state_row, observable_row) or None for blank lines."""
    bits = line.strip()
    if not bits:                           # skip empty / whitespace-only lines
        return None
    *state_chars, obs_char = bits          # unpack: last char is observable
    state_row = [c == "1" for c in state_chars]
    observable_row = [obs_char == "1"]
    return state_row, observable_row


def read_file(path: Path):
    """Produce two nested-list structures: states and observables."""
    states, observables = [], []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            parsed = parse_line(ln)
            if parsed is None:
                continue
            s, o = parsed
            states.append(s)
            observables.append(o)
    return states, observables


def main():
    if len(sys.argv) != 2:
        print(
            f"Usage: {Path(sys.argv[0]).name} <bitstring_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    infile = Path(sys.argv[1])
    if not infile.is_file():
        print(f"Error: '{infile}' is not a readable file.", file=sys.stderr)
        sys.exit(1)

    states, observables = read_file(infile)

    # Serialize as JSON so it’s valid Python *and* easy to reuse elsewhere
    print(json.dumps([states, observables]))


if __name__ == "__main__":
    main()
