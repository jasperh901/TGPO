#!/usr/bin/env python3
"""Extract the Eurus code mixture with the paper's 7B-tokenizer prompt filter."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    a.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, expected in [("train", 25092), ("validation", 1016)]:
        source = pq.ParquetFile(a.source / f"{split}.parquet")
        retained = []; before = 0
        for batch in source.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if row["ability"] != "code":
                    continue
                before += 1
                tokens = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
                if len(tokens) <= 1024:
                    retained.append(row)
        if len(retained) != expected:
            raise ValueError(f"{split}: expected {expected} rows, found {len(retained)}; verify the dataset and 7B tokenizer revisions")
        pq.write_table(pa.Table.from_pylist(retained, schema=source.schema_arrow), a.output / f"{split}.parquet")
        report[split] = {"code_rows": before, "rows": len(retained), "sources": dict(Counter(r["data_source"] for r in retained))}
    (a.output / "preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
