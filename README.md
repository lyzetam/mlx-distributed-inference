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

### Configuration File

You can use a configuration file to set default values for command-line arguments. The script automatically looks for `config.yaml`, `config.yml`, or `.mlx-config.yaml` in the current directory, or you can specify a custom config file with `--config`:

```bash
# Use default config.yaml
python3 distributed_inference_mlx.py "Your prompt"

# Use custom config file
python3 distributed_inference_mlx.py --config my-config.yaml "Your prompt"
```

See `config.yaml` for an example configuration file with all available options.

### Basic Examples

With configuration file (config.yaml sets defaults):
```bash
# Simple usage - uses all defaults from config.yaml
mlx.launch --hosts 10.85.100.220,10.85.100.221,10.85.100.222 \
  python3 distributed_inference_mlx.py "What is the meaning of life?"

# Override specific settings
mlx.launch --hostfile hosts.json --backend ring \
  python3 distributed_inference_mlx.py \
    --max-tokens 256 \
    "Explain quantum computing"
```

Without configuration file:
```bash
mlx.launch --hosts 10.85.100.221,10.85.100.222 \
  python3 distributed_inference_mlx.py \
    --max-tokens 1280 \
    --model-name mlx-community/DeepSeek-R1-Qwen3-0528-8B-4bit-AWQ \
    "What is the relativity theory"
```

Example output:

```
[2025-01-13 10:30:45] [Rank 0/2] [INFO] Loading model mlx-community/DeepSeek-R1-Qwen3-0528-8B-4bit-AWQ...
[2025-01-13 10:30:46] [Rank 0/2] [INFO] Using cached model: mlx-community/DeepSeek-R1-Qwen3-0528-8B-4bit-AWQ
[2025-01-13 10:30:52] [Rank 0/2] [INFO] Model loaded in 6.23s

Processing prompt: 'What is the relativity theory'
[Generated text appears here...]

Generation time: 58.3s
Prompt: 14 tokens, 62.7 tokens-per-sec
Generation: 315 tokens, 22.1 tokens-per-sec
Peak memory: 4.96 GB

[Distributed across 2 hosts] Done!
```

### Model Caching

Models are automatically cached to `~/.cache/mlx_models/` to avoid re-downloading:

```bash
# List cached models
python3 distributed_inference_mlx.py --list-cached

# Use a custom cache directory
python3 distributed_inference_mlx.py --cache-dir /path/to/cache "Your prompt"

# Force re-download a model
python3 distributed_inference_mlx.py --force-download "Your prompt"

# Clear the cache
python3 distributed_inference_mlx.py --clear-cache "Your prompt"

# Load from a local path
python3 distributed_inference_mlx.py --model-path /path/to/local/model "Your prompt"
```

### Logging Options

Enhanced logging with different levels and file output:

```bash
# Set log level (DEBUG, INFO, WARNING, ERROR)
python3 distributed_inference_mlx.py --log-level DEBUG "Your prompt"

# Log to file (creates separate files per rank)
python3 distributed_inference_mlx.py --log-file inference.log "Your prompt"

# Combined example
mlx.launch --hostfile hosts.json --backend mpi \
  python3 distributed_inference_mlx.py \
    --log-level INFO \
    --log-file logs/inference.log \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    "Explain quantum computing"
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `prompt` | The prompt text to send to the model | Required |
| `--config` | Path to configuration file (YAML format) | None |
| `--model-name` | HuggingFace model identifier | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| `--model-path` | Local path to model (overrides model-name) | None |
| `--cache-dir` | Directory to cache downloaded models | ~/.cache/mlx_models/ |
| `--force-download` | Force re-download even if cached | False |
| `--clear-cache` | Clear model cache before running | False |
| `--list-cached` | List cached models and exit | False |
| `--max-tokens` | Maximum tokens to generate | 100 |
| `--chat-template` | Chat template string or file path | None |
| `--log-level` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--log-file` | Log file path (appends rank for multi-process) | None |

**Note**: Command-line arguments override values from the configuration file.

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

## Distributed Training

This project now includes support for distributed training using MLX. You can train models across multiple nodes with gradient averaging for faster training times.

### Quick Start

1. **Jupyter Notebook** (Recommended for interactive training):
   ```bash
   jupyter notebook distributed_training.ipynb
   ```

2. **Command Line Script**:
   ```bash
   # Single node training
   python examples/train_example.py
   
   # Distributed training
   mpirun --hostfile hosts.json -np 2 python examples/train_example.py --distributed
   ```

### Training Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **Gradient Averaging**: Automatic gradient synchronization across nodes
- **Model Caching**: Reuses downloaded models from inference
- **Dataset Utilities**: Built-in dataset preparation and formatting
- **Progress Tracking**: Real-time loss visualization
- **Checkpointing**: Save and resume training

### Training Configuration

Configure training via `config/training_config.yaml`:

```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

training:
  type: "lora"
  lora:
    rank: 8
    alpha: 16

hyperparameters:
  batch_size: 2
  learning_rate: 1e-5
  num_iterations: 100

distributed:
  enabled: false  # Set to true for multi-node
```

### Dataset Format

Prepare your dataset in JSONL format with these fields:
```json
{"instruction": "Question here", "response": "Answer here", "context": "Optional context"}
```

Use the built-in utilities to prepare your data:
```python
from utils.data_preparation import prepare_dataset

dataset_info = prepare_dataset(
    file_path="raw_data.jsonl",
    output_dir="data",
    val_ratio=0.2,
    format_type="chat"
)
```

### Distributed Training Setup

1. **Configure MPI** (same as inference setup)
2. **Ensure identical code** on all nodes
3. **Run with MPI**:
   ```bash
   mlx.launch --hostfile hosts.json --backend mpi \
     python examples/train_example.py --distributed
   ```

### Using Trained Models

After training, use your adapter for inference:
```bash
mlx_lm.generate \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path adapters \
  --prompt "Your prompt here"
```

### Training Examples

See the `distributed_training.ipynb` notebook for:
- Step-by-step training walkthrough
- Dataset preparation examples
- Visualization of training progress
- Distributed vs single-node comparison
- Troubleshooting tips

For more details on the training implementation, see the example from [DaveAldon's repository](https://github.com/DaveAldon/Distributed-ML-with-MLX) which inspired this implementation.
