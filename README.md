# MLX Distributed Inference

This repository contains a small example for running distributed inference with [MLX](https://github.com/ml-explore/mlx).

## Requirements

Requires **Python 3.10+**. All Python dependencies are listed in
`requirements.txt`.

## Environment Setup

Run the provided script to install required packages and create a Python virtual environment:

```bash
./setup_env.sh
source mlx-env/bin/activate
```
Alternatively, create your own virtual environment and install the
dependencies with:

```bash
python3 -m venv mlx-env
source mlx-env/bin/activate
pip install -r requirements.txt
```

The script installs Python, Git and OpenMPI (when `apt-get` or Homebrew is available), then creates a virtual environment and installs the Python dependencies (`mlx`, `mlx_lm`, `huggingface-hub`). It also copies `hosts.json` to `~/.mlx/hosts.json` and generates SSH keys if none are present.

On macOS install OpenMPI using [Homebrew](https://brew.sh/) with:

```bash
brew install openmpi
```
before running the script if it is not already present.

## Usage

After activating the environment you can launch inference with `mlx.launch` or `mpirun`:

```bash
mlx.launch --hostfile hosts.json --backend mpi distributed_inference_mlx.py "Your prompt here"
```

or

```bash
mpirun --hostfile hosts.json -np 3 python distributed_inference_mlx.py "Your prompt here"
```

Replace the prompt with any text you would like to send to the model.

Example:

```bash
mlx.launch --hosts 10.85.100.221,10.85.100.222 \
  python3 distributed_inference_mlx.py \
    --max-tokens 1280 \
    --model-name mlx-community/DeepSeek-R1-Qwen3-0528-8B-4bit-AWQ \
    "What is the relativity theory"
```

Example output:

```
Generation time: 58.3s
Prompt: 14 tokens, 62.7 tokens-per-sec
Generation: 315 tokens, 22.1 tokens-per-sec
Peak memory: 4.96 GB
```

### Benchmarks

Running on two Mac mini M4 machines (10 cores, 16 GB RAM) peaks at roughly
5 GB of memory per host and generates around 22 tokens per second.

## `hosts.json`

The `hosts.json` file lists the machines participating in a run. Each entry
specifies the SSH address and the IPs visible to the other hosts:

```json
[
  {"ssh": "host1.example.com", "ips": ["10.0.0.1"]},
  {"ssh": "host2.example.com", "ips": ["10.0.0.2"]}
]
```

Add one object per machine in the order you want them to appear. Use the IP
addresses reachable by the other nodes for the `ips` field.

## Running Tests

Install the requirements and then execute `pytest` from the repository root:

```bash
pip install -r requirements.txt pytest
pytest
```

## Troubleshooting

If MPI fails to launch or nodes hang during initialization, ensure every machine uses the **exact same** project path. In one attempt the paths differed between systems, leading to hours of confusing errors and ultimately removing the supposed master node before noticing the mismatch.

