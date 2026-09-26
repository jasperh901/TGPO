#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


MODEL_NAME = "TGPO/ToolRL"
BFCL_COMMIT = "ea13468e4423454d0c213704fb87cf7cb3990433"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BFCL-v3 across independent single-GPU vLLM workers."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--test-category", nargs="+", default=["all"])
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--port-base", type=int, default=10600)
    parser.add_argument("--port-stride", type=int, default=100)
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--include-input-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_worker_ports(port_base, port_stride, worker_count, check_availability=True):
    if not 1024 <= port_base <= 65535:
        raise SystemExit(f"--port-base must be in [1024, 65535], got {port_base}")
    if port_stride < 16:
        raise SystemExit(
            "--port-stride must reserve at least 16 ports per vLLM worker, "
            f"got {port_stride}"
        )

    worker_ports = [port_base + index * port_stride for index in range(worker_count)]
    if worker_ports[-1] + 15 > 65535:
        raise SystemExit(
            "The final vLLM worker port range exceeds 65535: "
            f"{worker_ports[-1]}..{worker_ports[-1] + 15}"
        )

    if not check_availability:
        return worker_ports
    unavailable = []
    for worker_port in worker_ports:
        for port in range(worker_port, worker_port + 16):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("", port))
                except OSError:
                    unavailable.append(port)
    if unavailable:
        raise SystemExit(
            "Required vLLM worker ports are already in use: "
            + ", ".join(str(port) for port in unavailable)
        )
    return worker_ports


def iter_response_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_response_strings(item)


def partition_entries(entries, gpu_count, limit_per_category):
    from bfcl_eval.utils import sort_key

    grouped = defaultdict(list)
    for entry in sorted(entries, key=sort_key):
        grouped[entry["id"].rsplit("_", 1)[0]].append(entry)
    if limit_per_category:
        grouped = {
            category: category_entries[:limit_per_category]
            for category, category_entries in grouped.items()
        }

    partitions = [defaultdict(list) for _ in range(gpu_count)]
    ordered = []
    for category in sorted(grouped):
        ordered.extend(grouped[category])
    for index, entry in enumerate(ordered):
        category = entry["id"].rsplit("_", 1)[0]
        partitions[index % gpu_count][category].append(entry["id"])
    expected = {
        category: len(category_entries) for category, category_entries in grouped.items()
    }
    return partitions, expected


def merge_worker_results(output_root, worker_roots, expected):
    from bfcl_eval.utils import sort_key

    model_dir = MODEL_NAME.replace("/", "_")
    merged_by_category = defaultdict(dict)
    for worker_root in worker_roots:
        result_root = worker_root / "result" / model_dir
        for result_file in result_root.glob("BFCL_v3_*_result.json"):
            with result_file.open(encoding="utf-8") as handle:
                for line in handle:
                    entry = json.loads(line)
                    category = entry["id"].rsplit("_", 1)[0]
                    if entry["id"] in merged_by_category[category]:
                        raise ValueError(f"Duplicate BFCL result id: {entry['id']}")
                    merged_by_category[category][entry["id"]] = entry

    final_dir = output_root / "result" / model_dir
    final_dir.mkdir(parents=True, exist_ok=True)
    for category, expected_count in expected.items():
        entries = sorted(merged_by_category[category].values(), key=sort_key)
        if len(entries) != expected_count:
            raise ValueError(
                f"Category {category}: expected {expected_count} results, got {len(entries)}"
            )
        output_file = final_dir / f"BFCL_v3_{category}_result.json"
        with output_file.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return final_dir


def write_format_summary(output_root, result_dir):
    from bfcl_eval.model_handler.local_inference.toolrl_gdpo import (
        is_valid_toolrl_response,
    )

    categories = {}
    total_responses = 0
    valid_responses = 0
    total_entries = 0
    fully_valid_entries = 0
    for result_file in sorted(result_dir.glob("BFCL_v3_*_result.json")):
        category = result_file.name[len("BFCL_v3_") : -len("_result.json")]
        category_stats = {
            "entries": 0,
            "fully_valid_entries": 0,
            "responses": 0,
            "valid_responses": 0,
        }
        with result_file.open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                responses = list(iter_response_strings(entry["result"]))
                valid = [is_valid_toolrl_response(response) for response in responses]
                category_stats["entries"] += 1
                category_stats["fully_valid_entries"] += int(bool(valid) and all(valid))
                category_stats["responses"] += len(valid)
                category_stats["valid_responses"] += sum(valid)
        categories[category] = category_stats
        total_entries += category_stats["entries"]
        fully_valid_entries += category_stats["fully_valid_entries"]
        total_responses += category_stats["responses"]
        valid_responses += category_stats["valid_responses"]

    summary = {
        "definition": "Strict ToolRL structure plus parseable name/parameters JSON",
        "entry_level": {
            "fully_valid": fully_valid_entries,
            "total": total_entries,
            "ratio": fully_valid_entries / total_entries if total_entries else None,
        },
        "response_level": {
            "valid": valid_responses,
            "total": total_responses,
            "ratio": valid_responses / total_responses if total_responses else None,
        },
        "categories": categories,
    }
    write_json(output_root / "format_compliance.json", summary)
    return summary


def main():
    args = parse_args()
    if args.temperature < 0:
        raise SystemExit(f"--temperature must be nonnegative, got {args.temperature}")
    if not 0 < args.top_p <= 1:
        raise SystemExit(f"--top-p must be in (0, 1], got {args.top_p}")
    if not 0 <= args.seed < 2**31 - 1:
        raise SystemExit(f"--seed must be in [0, 2^31-2], got {args.seed}")
    checkpoint = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    bfcl_repo_root = Path(os.environ.get("TGPO_BFCL_DIR", str(Path(__file__).resolve().parents[1] / "third_party/BFCL-v3")))
    bfcl_project_root = bfcl_repo_root / "berkeley-function-call-leaderboard"
    sys.path.insert(0, str(bfcl_project_root))
    try:
        bfcl_head = subprocess.run(
            ["git", "-C", str(bfcl_repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Cannot resolve BFCL git revision: {bfcl_repo_root}") from exc
    if bfcl_head != BFCL_COMMIT:
        raise SystemExit(
            f"BFCL commit must be {BFCL_COMMIT}, got {bfcl_head}: {bfcl_repo_root}"
        )
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise SystemExit(f"Unique GPU ids are required, got: {args.gpus}")
    worker_ports = validate_worker_ports(args.port_base, args.port_stride, len(gpus),
                                        check_availability=not args.dry_run)
    if not (checkpoint / "reproduction_checkpoint.json").is_file():
        raise SystemExit(
            "Checkpoint must be exported by scripts/export_checkpoint.py and include "
            f"reproduction_checkpoint.json: {checkpoint}"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    from bfcl_eval._llm_response_generation import get_involved_test_entries

    _, selected_categories, entries = get_involved_test_entries(
        args.test_category, False
    )
    partitions, expected = partition_entries(
        entries, len(gpus), args.limit_per_category
    )
    worker_roots = []
    commands = []
    bfcl_executable = Path(sys.executable).with_name("bfcl")
    if not bfcl_executable.is_file():
        raise SystemExit(f"BFCL executable not found beside Python: {bfcl_executable}")

    for worker_index, (gpu, partition, worker_port) in enumerate(
        zip(gpus, partitions, worker_ports)
    ):
        worker_root = output_root / "workers" / f"gpu_{gpu}"
        worker_root.mkdir(parents=True, exist_ok=True)
        write_json(worker_root / "test_case_ids_to_generate.json", partition)
        worker_roots.append(worker_root)
        command = [
            str(bfcl_executable),
            "generate",
            "--model",
            MODEL_NAME,
            "--run-ids",
            "--backend",
            "vllm",
            "--num-gpus",
            "1",
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--temperature",
            str(args.temperature),
            "--local-model-path",
            str(checkpoint),
            "--result-dir",
            "result",
        ]
        if args.include_input_log:
            command.append("--include-input-log")
        commands.append(
            {
                "worker": worker_index,
                "gpu": gpu,
                "port": worker_port,
                "command": command,
                "case_count": sum(len(ids) for ids in partition.values()),
            }
        )

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "started_at": utc_now(),
        "model": MODEL_NAME,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": json.loads(
            (checkpoint / "reproduction_checkpoint.json").read_text(encoding="utf-8")
        ),
        "bfcl_commit": bfcl_head,
        "evaluation_code": {
            os.path.relpath(path, Path(__file__).resolve().parents[1]): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                Path(__file__).resolve(),
                bfcl_project_root
                / "bfcl_eval/model_handler/local_inference/toolrl_gdpo.py",
                bfcl_project_root
                / "bfcl_eval/model_handler/local_inference/base_oss_handler.py",
                bfcl_project_root / "bfcl_eval/constants/model_config.py",
                bfcl_project_root / "bfcl_eval/eval_checker/eval_runner.py",
                bfcl_project_root
                / "bfcl_eval/eval_checker/eval_runner_helper.py",
            )
        },
        "selected_categories": selected_categories,
        "expected_results": expected,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "evaluation_seed": args.seed,
        "question_seed_definition": (
            "blake2b-64('bfcl-v3-toolrl:<evaluation_seed>:<question_id>') "
            "modulo (2^31-1); request k uses "
            "(question_seed + 104729*k) modulo (2^31-1)"
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "port_base": args.port_base,
        "port_stride": args.port_stride,
        "gpus": gpus,
        "limit_per_category": args.limit_per_category,
        "workers": commands,
    }
    manifest_path = output_root / "run_manifest.json"
    write_json(manifest_path, manifest)
    if args.dry_run:
        print(f"BFCL dry-run manifest: {manifest_path}")
        return

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(bfcl_project_root)
    base_env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NCCL_IB_DISABLE": "1",
            "PYTHONUNBUFFERED": "1",
            "BFCL_EVAL_SEED": str(args.seed),
            "BFCL_EVAL_TOP_P": str(args.top_p),
            "PATH": f"{Path(sys.executable).parent}:{base_env.get('PATH', '')}",
        }
    )
    processes = []
    started = time.monotonic()
    for worker, worker_root in zip(commands, worker_roots):
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = worker["gpu"]
        env["VLLM_PORT"] = str(worker["port"])
        env["BFCL_PROJECT_ROOT"] = str(worker_root)
        log_path = worker_root / "worker.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            worker["command"],
            cwd=bfcl_project_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((process, log_handle, log_path))
        print(
            f"Started BFCL worker {worker['worker']} on GPU {worker['gpu']} "
            f"for {worker['case_count']} cases (pid={process.pid})"
        )

    failure = None
    while processes:
        remaining = []
        for process, log_handle, log_path in processes:
            return_code = process.poll()
            if return_code is None:
                remaining.append((process, log_handle, log_path))
                continue
            log_handle.close()
            print(f"BFCL worker pid={process.pid} exited with code {return_code}")
            if return_code != 0 and failure is None:
                failure = (process.pid, return_code, log_path)
        processes = remaining
        if failure:
            for process, log_handle, _ in processes:
                process.terminate()
                process.wait(timeout=30)
                log_handle.close()
            break
        if processes:
            time.sleep(5)

    if failure:
        pid, return_code, log_path = failure
        manifest["status"] = "failed"
        manifest["failure"] = {
            "pid": pid,
            "return_code": return_code,
            "log": str(log_path),
        }
        write_json(manifest_path, manifest)
        raise SystemExit(f"BFCL worker failed; inspect {log_path}")

    final_result_dir = merge_worker_results(output_root, worker_roots, expected)
    format_summary = write_format_summary(output_root, final_result_dir)

    manifest["result_dir"] = str(final_result_dir)
    manifest["format_compliance"] = format_summary
    if args.limit_per_category:
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["official_evaluation"] = {
            "status": "skipped",
            "reason": (
                "BFCL's official evaluator requires complete category result files; "
                "--limit-per-category is a generation/merge/format qualification only."
            ),
        }
        write_json(manifest_path, manifest)
        print(f"Merged BFCL qualification results: {final_result_dir}")
        print(f"Format compliance: {output_root / 'format_compliance.json'}")
        print("Official BFCL evaluation skipped for the limited qualification subset.")
        print(f"Run manifest: {manifest_path}")
        return

    eval_env = base_env.copy()
    eval_env["BFCL_PROJECT_ROOT"] = str(output_root)
    evaluation_command = [
        str(bfcl_executable),
        "evaluate",
        "--model",
        MODEL_NAME,
        "--test-category",
        ",".join(args.test_category),
        "--result-dir",
        "result",
    ]
    evaluation_log = output_root / "evaluation.log"
    try:
        with evaluation_log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                evaluation_command,
                cwd=bfcl_project_root,
                env=eval_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["official_evaluation"] = {
            "status": "failed",
            "return_code": exc.returncode,
            "command": evaluation_command,
            "log": str(evaluation_log),
        }
        write_json(manifest_path, manifest)
        raise SystemExit(
            f"Official BFCL evaluation failed; inspect {evaluation_log}"
        ) from exc

    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["score_dir"] = str(output_root / "score")
    manifest["official_evaluation"] = {
        "status": "complete",
        "command": evaluation_command,
        "log": str(evaluation_log),
    }
    write_json(manifest_path, manifest)
    print(f"Merged BFCL results: {final_result_dir}")
    print(f"BFCL scores: {output_root / 'score'}")
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
