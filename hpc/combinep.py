#!/usr/bin/env python3
"""
Concatenate part_*_of_*.csv files written by the array jobs.
The output is  combined_results.csv  in the same folder.
"""
import sys, glob, os, pandas as pd

if len(sys.argv) != 2:
    sys.exit("usage: combine_csv.py <folder>")

folder = sys.argv[1]
parts  = sorted(glob.glob(os.path.join(folder, "part_*_of_*.csv")))
if not parts:
    sys.exit(f"no part_*.csv files found in {folder}")

df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
out = os.path.join(folder, "combined_results.csv")
df.to_csv(out, index=False)
print(f"[combine_csv] wrote {out} with {len(df)} rows from {len(parts)} parts")

