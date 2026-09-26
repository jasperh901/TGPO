"""Dataset identities and sampling seeds shared by generation and scoring."""
from collections import Counter
import hashlib
import json

SEEDS = (2026091601, 2026091602, 2026091603, 2026091604)
DATA_ERRORS = {18, 958}
SOURCES = {"apps": 142, "codecontests": 372, "codeforces": 122, "taco": 380}
MAX_TOKENS = 32768


def request_seed(position, sample):
    return int.from_bytes(hashlib.sha256(f"{SEEDS[sample]}:{position}".encode()).digest()[:4], "big") % (2**31 - 1)


def question_identity(row, position):
    value = dict(dataset_position=position, data_source=row["data_source"].lower(),
                 prompt=row["prompt"], ground_truth=row["reward_model"]["ground_truth"], extra_info=row["extra_info"])
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def load_rows(path):
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 1016 or dict(Counter(r["data_source"].lower() for r in rows)) != SOURCES:
        raise ValueError("Expected the 1,016-question code validation mixture; run prepare_code_data.py")
    return rows


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
