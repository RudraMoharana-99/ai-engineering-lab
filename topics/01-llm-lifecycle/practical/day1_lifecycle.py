"""Day 1: inspect tokens, a forward pass, and a short CPU/CUDA generation.

This is an operational learning check, not a throughput benchmark.
Run from the repository root. --help works without third-party packages.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def cpu_model():
    """Best-effort name; core counts are separately reported by psutil."""
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown; inspect the operating system"


def inventory(torch, psutil):
    memory = psutil.virtual_memory()
    names = ["torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "psutil"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "packages": versions,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_vram_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available() else None
        ),
    }


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--revision", default="main", help="Use the CPU run's commit SHA on GPU.")
    parser.add_argument("--prompt", default="Explain what a token is in one short sentence.")
    parser.add_argument("--max-new-tokens", type=positive_int, default=24)
    parser.add_argument("--threads", type=positive_int, help="Optional native CPU thread override.")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Must be a new file; default uses a timestamp.")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    mode = "inventory" if args.inventory_only else args.device
    output = args.output or Path(f"results/local/day1-{mode}-{stamp}.json")
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose a new filename.")

    try:
        import psutil
        import torch
    except ImportError as exc:
        parser.error(f"Missing package: {exc.name}. Follow docs/SETUP_CPU.md.")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    hardware = inventory(torch, psutil)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "kind": "hardware_inventory" if args.inventory_only else "day1_operational_check",
        "hardware": hardware,
    }

    if not args.inventory_only:
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Select a Colab GPU runtime; no CPU fallback was used.")
        try:
            from huggingface_hub import HfApi
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            parser.error(f"Missing package: {exc.name}. Install requirements.txt.")

        # Resolve once, then load both tokenizer and weights from that exact revision.
        revision = HfApi().model_info(MODEL_ID, revision=args.revision).sha
        if not revision:
            raise RuntimeError("The Hub did not return an immutable model revision.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=revision,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            use_safetensors=True,
            trust_remote_code=False,
        ).to(args.device)
        model.eval()
        parameter = next(model.parameters())
        if parameter.device.type != args.device:
            raise RuntimeError("The model did not reach the requested device.")
        messages = [{"role": "user", "content": args.prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # The template already supplies special tokens; avoid adding them twice.
        inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        input_count = inputs["input_ids"].shape[1]
        probe_before = parameter.detach().reshape(-1)[:8].cpu().clone()
        settings = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        # A standalone forward pass exposes [batch, sequence, vocabulary] logits.
        # It is outside the generation timing and does not reuse a cache.
        with torch.inference_mode():
            forward = model(**inputs, use_cache=False)
            logits_shape = list(forward.logits.shape)
            next_id = int(forward.logits[0, -1].argmax().item())
            del forward
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            generated = model.generate(**inputs, **settings)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        new_ids = generated[0, input_count:].cpu().tolist()
        probe_after = parameter.detach().reshape(-1)[:8].cpu()
        report.update({
            "track": "gpu" if args.device == "cuda" else "cpu",
            "model": {"id": MODEL_ID, "revision": revision},
            "model_card_url": f"https://huggingface.co/{MODEL_ID}/blob/{revision}/README.md",
            "confirmed_model_device": str(parameter.device),
            "confirmed_input_device": str(inputs["input_ids"].device),
            "dtype": str(parameter.dtype),
            "attention_implementation": "eager",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "weight_storage_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "prompt": args.prompt,
            "formatted_prompt": formatted,
            "input_token_ids": inputs["input_ids"][0].cpu().tolist(),
            "input_token_count": input_count,
            "logits_shape": logits_shape,
            "first_greedy_token_id_from_forward": next_id,
            "first_greedy_token_text_from_forward": tokenizer.decode([next_id]),
            "generated_token_ids": new_ids,
            "generated_token_count": len(new_ids),
            "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
            "settings": settings,
            "effective_generation_config": model.generation_config.to_dict() | settings,
            "sampled_weights_unchanged": bool(torch.equal(probe_before, probe_after)),
            "weight_check_scope": "Only the first eight values of one parameter; not a full-model checksum.",
            "generate_call_seconds": elapsed,
            "timing_scope": "One generate call after a diagnostic forward; excludes loading/tokenization/decoding. Not TTFT, ITL, or a benchmark.",
            "process_rss_after_bytes": psutil.Process(os.getpid()).memory_info().rss,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
