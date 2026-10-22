# Day 1 practical · One model, CPU and Colab GPU

Goal: connect the theory to observable tokens, parameters, logits, generation, and device placement. Do one step at a time with the mentor.

## 1. Prepare the CPU environment

Follow [setup](../../../docs/SETUP_CPU.md). Read the [model-card worksheet](../model-card.md). The first model run needs internet access to retrieve the public model. It uses float32 weights and an explicit attention implementation on both devices.

## 2. Run one short CPU generation

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe topics/01-llm-lifecycle/practical/day1_lifecycle.py --device cpu --output results/local/cpu.json
```

If `cpu.json` already exists, use a new name. The script will not overwrite a previous result.

## 3. Inspect five things before changing anything

1. Find the original prompt and the formatted prompt. The latter includes chat-template tokens.
2. Find the input and generated token counts. Generated count includes any EOS token; displayed text omits special tokens. `max_new_tokens` is a limit, not a promise that all 24 tokens will be produced.
3. Identify `[batch, input length, vocabulary size]` in the logits shape and find the first greedy token ID.
4. Confirm both model and input devices are `cpu`, and inspect the parameter count and dtype.
5. Read the weight probe. It compares eight parameter values and does not prove equality of every weight. There is no optimizer or weight-update step in the program.

The JSON also records the exact model revision, hardware, package versions, generation configuration, and one `generate()` call duration. The recorded weight storage is not total runtime memory. Process RSS is a snapshot, not peak memory. The elapsed call is not TTFT/ITL or a proper throughput benchmark.

## 4. Repeat on Colab GPU

Open [day1_colab.ipynb](day1_colab.ipynb) in [Google Colab](https://colab.research.google.com/) using **Upload notebook**. Select a GPU runtime if one is available. The notebook asks you to upload this folder's `day1_lifecycle.py` and your saved `cpu.json`, then runs the same revision, prompt, output limit, precision, and decoding policy on CUDA.

Keep Colab's supplied CUDA-enabled PyTorch; do not install the local CPU-only wheel there. GPU types and availability vary, and runtimes are temporary. The notebook downloads the resulting JSON and environment snapshot. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

If a GPU is unavailable, report that and keep the CPU result. Do not relabel a CPU run as GPU. No paid runtime is required by this lesson.

## 5. Compare observations

| Observation | CPU | GPU |
|---|---|---|
| Actual model/input device | Fill from result | Fill from result |
| Model revision and dtype | Fill from result | Must match |
| Input IDs and token count | Fill from result | Must match |
| Generated IDs and actual count | Fill from result | Compare; floating-point differences can affect greedy choices |
| Parameter count and logits shape | Fill from result | Must match |
| Hardware and packages | Fill from result | Record differences |

This compares execution and settings. A single local/Colab timing ratio does not isolate hardware speed, and two different output lengths are unequal work. Controlled performance comparisons come later.

## 6. Return to the mentor

Share the result's device, input/generated counts, logits shape, and any confusing output. We discuss practical doubts, then begin the interview one question at a time.

The `cpu_pipeline` research track is planned in [experiments](../../../experiments/README.md). Day 1 does not implement it or assert a speedup.
