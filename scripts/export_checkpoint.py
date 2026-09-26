#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a raw Verl/FSDP checkpoint as a validated BF16 inference checkpoint."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_inventory(checkpoint_dir):
    from safetensors import safe_open

    dtype_counts = {}
    tensor_count = 0
    stored_elements = 0
    weight_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"No safetensors weights found in {checkpoint_dir}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor_slice = handle.get_slice(key)
                dtype = str(tensor_slice.get_dtype())
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                elements = 1
                for dimension in tensor_slice.get_shape():
                    elements *= dimension
                stored_elements += elements
                tensor_count += 1
    return weight_files, dtype_counts, tensor_count, stored_elements


def main():
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise SystemExit("Source and output checkpoint directories must differ.")
    if not source.is_dir():
        raise SystemExit(f"Source checkpoint does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output}")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    source_config = AutoConfig.from_pretrained(source, local_files_only=True)
    source_weights, source_dtypes, source_tensors, source_elements = tensor_inventory(
        source
    )

    output.mkdir(parents=True, exist_ok=True)
    print(f"Loading raw checkpoint on CPU: {source}")
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    parameter_count = model.num_parameters()
    model.eval()
    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.save_pretrained(output)
    del model

    output_config = AutoConfig.from_pretrained(output, local_files_only=True)
    output_weights, output_dtypes, output_tensors, output_elements = tensor_inventory(
        output
    )
    if set(output_dtypes) != {"BF16"}:
        raise ValueError(f"Inference weights are not uniformly BF16: {output_dtypes}")

    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "reproduction_checkpoint.json":
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_config_dtype": str(source_config.torch_dtype),
        "source_weight_files": [path.name for path in source_weights],
        "source_dtype_counts": source_dtypes,
        "source_tensor_count": source_tensors,
        "source_stored_elements": source_elements,
        "output": str(output),
        "output_config_dtype": str(output_config.torch_dtype),
        "output_weight_files": [path.name for path in output_weights],
        "output_dtype_counts": output_dtypes,
        "output_tensor_count": output_tensors,
        "output_stored_elements": output_elements,
        "model_parameter_count": parameter_count,
        "files": files,
    }
    manifest_path = output / "reproduction_checkpoint.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Validated BF16 inference checkpoint: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
