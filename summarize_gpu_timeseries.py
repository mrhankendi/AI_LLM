#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import re
import pandas as pd

def parse_meta_from_name(name: str):
    """Parse (model, tp, cap, bs) from a filename.
    Expected patterns like: "..._tp4_cap300_bs32_gpu_timeseries.csv"
    """
    low = name.lower()
    # Model guess
    if "gpt2" in low: model = "gpt2"
    elif "deepseek" in low: model = "deepseek"
    elif "olmoe" in low: model = "olmoe"
    elif "qwen" in low: model = "qwen"
    elif "mixtral" in low: model = "mixtral"
    elif "llama" in low: model = "llama"
    elif "microsoft" in low: model = "microsoft"
    elif "mistral" in low: model = "mistral"
    else: model = "unknown"
    # TP, Cap, BS
    tp_match  = re.search(r"_tp(\d+)_", name, flags=re.IGNORECASE)
    cap_match = re.search(r"_cap(\d+)_", name, flags=re.IGNORECASE)
    bs_match  = re.search(r"_bs(\d+)_",  name, flags=re.IGNORECASE)

    tp  = int(tp_match.group(1))  if tp_match  else None
    cap = int(cap_match.group(1)) if cap_match else None
    bs  = int(bs_match.group(1))  if bs_match  else None
    return model, tp, cap, bs

def summarize_file(csv_path: Path) -> pd.Series:
    """Return a Series of column-wise means for numeric columns, plus metadata."""
    try:
        df = pd.read_csv(csv_path)
        # Clean column names: remove surrounding quotes (ASCII and Unicode)
        # and extra whitespace. Some exported CSVs include headers with
        # embedded single/double quotes (e.g. "'gpu_power_w'") or curly
        # quotes; strip those so downstream column names don't contain
        # extra quote characters.
        df.columns = (
            df.columns.astype(str)
            .str.normalize("NFKC")
            .str.replace(r"^[\s'\"\u2018\u2019\u201C\u201D]+|[\s'\"\u2018\u2019\u201C\u201D]+$", "", regex=True)
            .str.strip()
        )
        
        # Skip the first data row
        if len(df) > 1:
            df = df.iloc[1:]
        
        # Clean data cells for object/string columns: strip surrounding
        # single/double quotes and whitespace. If a column becomes fully
        # numeric after stripping, convert it to numeric so downstream
        # aggregation sees it as a number.
        for col in df.columns:
            print(col)
            if df[col].dtype == object:
                cleaned = df[col].astype(str)
                # Normalize unicode forms (e.g. different quote chars)
                try:
                    cleaned = cleaned.str.normalize("NFKC")
                except Exception:
                    # older pandas/objects may not support .str.normalize; ignore
                    pass
                # Remove leading/trailing whitespace and both ASCII + common
                # Unicode quote characters (curly quotes, etc.). This handles
                # values like "'25.3..." or "125".
                cleaned = cleaned.str.replace(
                    r"^[\s'\"\u2018\u2019\u201C\u201D]+|[\s'\"\u2018\u2019\u201C\u201D]+$",
                    "",
                    regex=True,
                )
                # Treat empty strings as NA
                cleaned = cleaned.replace({'': pd.NA})
                # Try converting to numeric; accept the conversion only if
                # all originally-non-null values convert successfully.
                conv = pd.to_numeric(cleaned, errors='coerce')
                orig_non_null = df[col].notna().sum()
                conv_non_null = conv.notna().sum()
                if orig_non_null > 0 and conv_non_null == orig_non_null:
                    df[col] = conv
                else:
                    # Some values may include invisible/control characters or a
                    # stray prefix (e.g. an apostrophe) that prevent direct
                    # conversion. Try extracting a numeric substring from each
                    # cell (handles things like "'25.3" or "\x1925.3").
                    extracted = cleaned.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
                    conv2 = pd.to_numeric(extracted, errors='coerce')
                    conv2_non_null = conv2.notna().sum()
                    if orig_non_null > 0 and conv2_non_null == orig_non_null:
                        df[col] = conv2
                    else:
                        df[col] = cleaned
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}", file=sys.stderr)
        return pd.Series(dtype=float)

    # Keep only numeric columns for averaging
    num_df = df.select_dtypes(include=["number"])
    if num_df.empty:
        print(f"[WARN] No numeric columns in {csv_path}", file=sys.stderr)
        return pd.Series(dtype=float)

    # Column-wise means
    means = num_df.mean(numeric_only=True)

    # Metadata
    model, tp, cap, bs = parse_meta_from_name(csv_path.name)
    meta = pd.Series({
        "model": model,
        "tp": tp,
        "cap_w": cap,
        "batch": bs,
        "filename": str(csv_path)
    })

    # Prefix metric columns to make it obvious these are averages
    means.index = [f"avg_{c}" for c in means.index]

    # Concatenate meta + means
    out = pd.concat([meta, means])
    return out

def main():
    ap = argparse.ArgumentParser(description="Summarize GPU timeseries CSVs into one row per file (time-averaged metrics).")
    ap.add_argument("--root", required=True, help="Root folder to scan (e.g., extracted PROFILE directory).")
    ap.add_argument("--pattern", default="*gpu_timeseries*.csv", help="Filename glob to find timeseries CSVs.")
    ap.add_argument("--out", required=True, help="Output CSV path for merged summary.")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        print(f"[ERROR] Root path does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    # Find CSVs
    files = list(root.rglob(args.pattern))
    if not files:
        print(f"[ERROR] No files matched pattern '{args.pattern}' under {root}", file=sys.stderr)
        sys.exit(3)

    rows = []
    for f in files:
        row = summarize_file(f)
        if not row.empty:
            rows.append(row)

    if not rows:
        print("[ERROR] No valid summaries produced.", file=sys.stderr)
        sys.exit(4)

    # Combine to a single dataframe, align columns
    merged = pd.DataFrame(rows)

    # Sort columns: meta first, then metrics sorted
    meta_cols = ["model", "tp", "cap_w", "batch", "filename"]
    metric_cols = sorted([c for c in merged.columns if c.startswith("avg_")])
    ordered_cols = [c for c in meta_cols if c in merged.columns] + metric_cols
    merged = merged.reindex(columns=ordered_cols)

    # Sort rows by model, tp, cap, batch for readability
    merged = merged.sort_values(by=[c for c in ["model","tp","cap_w","batch"] if c in merged.columns])

    # Save
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} with {len(merged)} rows and {len(merged.columns)} columns.")

if __name__ == "__main__":
    main()
