# MLX Distributed Inference

This repository contains a small example for running distributed inference with [MLX](https://github.com/ml-explore/mlx).

## Environment Setup

Run the provided script to install required packages and create a Python virtual environment:

```bash
./setup_env.sh
source mlx-env/bin/activate
```

Alternatively, install the same Python packages manually with:

```bash
pip install -r requirements.txt
```

The script installs Python, Git and OpenMPI (when `apt-get` is available), then creates a virtual environment and installs the Python dependencies (`mlx`, `mlx_lm`, `huggingface-hub`). It also copies `hosts.json` to `~/.mlx/hosts.json` and generates SSH keys if none are present.

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
