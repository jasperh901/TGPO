#!/usr/bin/env python3
"""Four-sample code generation with vLLM 0.8.5 and fixed per-question seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from code_eval_common import MAX_TOKENS, load_rows, question_identity, request_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/code/validation.parquet")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    p.add_argument("--eager", action="store_true")
    a = p.parse_args()
    if a.rank is None:
        if a.output.exists() and any(a.output.iterdir()):
            p.error("Choose a new output directory to avoid mixing evaluations")
        a.output.mkdir(parents=True, exist_ok=True)
        processes = []
        for rank, gpu in enumerate(a.gpus.split(",")):
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [sys.executable, __file__, *sys.argv[1:], "--rank", str(rank)]
            log = (a.output / f"rank{rank}.log").open("w")
            processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        failures = []
        for process, log in processes:
            failures.append(process.wait()); log.close()
        if any(failures):
            raise RuntimeError(f"Generation failed: {failures}; inspect the rank logs")
        (a.output / "generation.json").write_text(json.dumps({"checkpoint": str(a.checkpoint.resolve()),
            "ranks": len(processes), "questions": 1016, "samples_per_question": 4, "max_tokens": MAX_TOKENS}, indent=2))
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    rows = load_rows(a.data)
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
    prompts = [tokenizer.apply_chat_template(r["prompt"], tokenize=True, add_generation_prompt=True) for r in rows]
    if max(map(len, prompts)) > 1024:
        raise ValueError("Code prompt exceeds the 1,024-token budget")
    llm = LLM(model=str(a.checkpoint.resolve()), tokenizer=str(a.checkpoint.resolve()), dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=MAX_TOKENS + 1024,
              max_num_seqs=64, max_num_batched_tokens=8192, enable_chunked_prefill=True,
              enable_prefix_caching=False, disable_async_output_proc=True, enforce_eager=a.eager,
              max_seq_len_to_capture=MAX_TOKENS + 1024, swap_space=0, seed=20260916)
    engine = llm.llm_engine
    for sample in range(4):
        for position in range(len(rows)):
            if (position * 4 + sample) % len(a.gpus.split(",")) != a.rank:
                continue
            parameters = SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=-1, max_tokens=MAX_TOKENS,
                seed=request_seed(position, sample), detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY)
            engine.add_request(f"{position}:{sample}", {"prompt_token_ids": prompts[position]}, parameters)
    with (a.output / f"rank{a.rank}.jsonl").open("w", buffering=1) as handle:
        while engine.has_unfinished_requests():
            for result in engine.step():
                if not result.finished:
                    continue
                position, sample = map(int, result.request_id.split(":"))
                answer = result.outputs[0]; tokens = list(answer.token_ids)
                response = tokenizer.decode(tokens, skip_special_tokens=False)
                if tokenizer.pad_token:
                    response = response.replace(tokenizer.pad_token, "")
                record = dict(position=position, sample=sample, question_id=question_identity(rows[position], position),
                    seed=request_seed(position, sample), source=rows[position]["data_source"].lower(),
                    response=response, response_tokens=len(tokens), token_ids=tokens,
                    finish_reason=answer.finish_reason, model=str(a.checkpoint.resolve()))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
