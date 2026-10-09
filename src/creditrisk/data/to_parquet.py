"""raw Home Credit CSV -> small fast parquet files"""

import argparse
import gc
import time

import pandas as pd

from creditrisk.config import get_path, load_config


def shrink(df : pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df[col] = s.astype("float32")
        elif not pd.api.types.is_numeric_dtype(s):
            df[col] = s.astype("category")

    return df

def memory_mb(df: pd.DataFrame) -> float:
    """Real memory used by a table, in MB (deep=True also counts the text)."""
    return df.memory_usage(deep=True).sum() / 1e6
 
 
def convert_all(force: bool = False) -> None:
    """Convert every table listed in config.yaml from CSV to Parquet, one at a time."""
    tables = load_config()["data"]["tables"]
    raw_dir, out_dir = get_path("raw"), get_path("interim")
    total_before = total_after = 0.0
 
    for name, csv_name in tables.items():
        out_file = out_dir / f"{name}.parquet"
        if out_file.exists() and not force:
            print(f"{name:<24} already converted (use --force to redo)")
            continue
 
        start = time.time()
        df = pd.read_csv(raw_dir / csv_name)
        before = memory_mb(df)
        shrink(df)
        after = memory_mb(df)
        df.to_parquet(out_file, index=False)
 
        total_before += before
        total_after += after
        rows, cols = df.shape
        print(
            f"{name:<24} {rows:>11,} rows x {cols:>3} cols  "
            f"{before:>7.0f} MB -> {after:>5.0f} MB  ({time.time() - start:.0f}s)"
        )
        del df
        gc.collect()  # free the memory before loading the next table
 
    if total_before:
        print(f"{'TOTAL':<52} {total_before:>7.0f} MB -> {total_after:>5.0f} MB")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw CSV files to Parquet.")
    parser.add_argument("--force", action="store_true", help="redo tables already converted")
    args = parser.parse_args()
    convert_all(force=args.force)
 
 
if __name__ == "__main__":
    main()