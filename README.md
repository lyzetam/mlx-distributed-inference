# Distributed Inference with MLX

This repository contains an example of running a large language model across multiple nodes using [MLX](https://github.com/ml-explore/mlx). The `distributed_inference_mlx.py` script implements pipeline parallelism so that each host only loads part of the model, allowing inference on models that exceed the memory of a single machine.

## Running the Example

1. **Prepare each node**
   - Clone this repository on every machine that will participate in the run.
   - Run the provided `setup_env.sh` script to install the required Python packages and prepare MLX. This should be done once per node:
     ```bash
     ./setup_env.sh
     ```

2. **Update the host list**
   - Edit `hosts.json` to list the SSH address and IPs for each node. Example:
     ```json
     [
       {"ssh": "10.0.0.1", "ips": ["10.0.0.1"]},
       {"ssh": "10.0.0.2", "ips": ["10.0.0.2"]}
     ]
     ```

3. **Launch distributed inference**
   - Use `mlx.launch` with the MPI backend to start the script on all hosts:
     ```bash
     mlx.launch --hostfile hosts.json --backend mpi distributed_inference_mlx.py "Your prompt here"
     ```
   - Alternatively, you can use `mpirun` directly:
     ```bash
     mpirun --hostfile hosts.json -np 2 python distributed_inference_mlx.py "Your prompt here"
     ```

The script will load and shard the specified model across the hosts and then generate text for the provided prompt. Only rank 0 prints the output so you do not see duplicates.

## Adding New Nodes

When bringing a new machine into the cluster, clone this repository and run `setup_env.sh` to install the dependencies. After that, add the node's information to `hosts.json` and re-run the `mlx.launch` command as above.


