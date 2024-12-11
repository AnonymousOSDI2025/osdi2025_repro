# osdi2025_repro

## Setup

This experiment scripts require 4 nodes that has 8 A100 GPUs each.
We need to install a [modified version of DeepSpeed](https://github.com/AnonymousOSDI2025/DeepSpeed/tree/osdi_repro) and other dependencies.

## Evaluation on throughput and peak memory (Fig. 6 and 7)

The following script runs the throughput benchmark in Fig. 6 of the paper.
This sweeps the following conditions:

- Models: meta-llama/Meta-Llama-3-70B-Instruct, mistralai/Mixtral-8x7B-v0.1
- Batch size: 1, 2, 4
- Sequence length: 512 1024 2048
- Frameworks: DeepSpeed, FSDP, DeepCompile (P), DeepCompile (S), DeepCompile (P+S)

```bash
bash run_bench.sh
```

The following script runs the benchmark with different number of gradient accumulation steps (2, 4, 8, 16).
The batch size and sequence length are fixed to 1 and 1024, respectively.

```bash
bash run_bench_acc.sh
```

The summary of results are output to `$PROFILE_DIR/results.txt`. Please set an appropriate `PROFILE_DIR` in the script.


## Correctness test

The following script runs DeepSpeed and DeepCompile with the same conditions.
`MASTER_ADDR` must be set to the IP address of the master node.

```bash
bash run_correctness_test.sh
```