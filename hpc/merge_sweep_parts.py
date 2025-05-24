#!/usr/bin/env python


import argparse, glob, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def main(parts_dir: str, sweep_name: str, out_dir: str):
    parts_dir = Path(parts_dir)
    out_dir   = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(parts_dir.glob("part_*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSVs found in {parts_dir}")

    df = pd.concat((pd.read_csv(f) for f in csvs), ignore_index=True)
    df = df.sort_values(sweep_name).reset_index(drop=True)

    merged = out_dir / "data.csv"
    df.to_csv(merged, index=False)
    print(f"[merge] combined file → {merged}")

    
    x = df[sweep_name]

    plt.figure(figsize=(8,5))
    plt.plot(x, df["lower_cube_flow_out"], marker='o')
    plt.xscale('log'); plt.grid(True, which='both', linestyle='--', lw=0.2)
    plt.xlabel(sweep_name); plt.ylabel("Lower cube flow out")
    plt.title(f"Lower cube flux vs {sweep_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "lower_cube_plot.pdf"); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(x, df["upper_cube_flow_in"],  marker='s', ls='--', label="Flux in")
    plt.plot(x, df["upper_cube_flow_out"], marker='^', ls='-.', label="Flux out")
    plt.plot(x, df["upper_cube_flow"],     marker='d', ls=':',  label="Total flux")
    plt.xscale('log'); plt.grid(True, which='both', linestyle='--', lw=0.2)
    plt.xlabel(sweep_name); plt.ylabel("Upper cube flux [units]")
    plt.title(f"Upper cube fluxes vs {sweep_name}"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "upper_cube_plot.pdf"); plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parts_dir", required=True)
    p.add_argument("--sweep_name", required=True)
    p.add_argument("--out_dir",   required=True)
    main(**vars(p.parse_args()))

