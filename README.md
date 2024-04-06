# Unadapable-Foundation-Models

## Setup

1. Install PyTorch separately (for local models). You need torch for most experiments and torchvision for the MNIST toy model.
> See https://pytorch.org/get-started/locally/

2. There are two requirements files: `requirements.txt` and `requirements-dev.txt`. The former is pared back and 
   should include only the essentials for e.g. running on the cluster. The latter is for devs and can include 
   visualization and jupyter bells and whistles etc.
```bash
pip install -r requirements-dev.txt
```

## CAIS Compute Cluster
After logging into the cluster, you can transfer from the login node to a compute node with
```bash
srun --partition=single --pty bash
```
You may want to add an alias to you `.bashrc`.
From there you can run your compute jobs.

You can run a one-off job with `srun`, specifying your various params.

> _All our jobs must be run with `--partition=single` or `interactive`._

Alternative to `srun`, you submit a job to the scheduler with `sbatch`. Simply call e.g. `sbatch <sbatch_file>`. Your 
file should specify various run parameters. We 
have a simple executable `sbatch_run` you can reference.

You can then view your jobs with `squeue`. When plain, this shows all jobs, but you can isolate your own with 
`squeue -u <username>`.

### Setup script

The script `setup.sh` sets up the environment for running jobs. It should be called in the sbatch scripts. Of note it:

- Enters or initializes a `venv` and pip installs packages.
  - **Note:** You can speed up this process by doing these steps in your working directory and the slurm jobs will 
    use them instead of redoing it all. (At least _I think_ it will. It might get weird if you're using multiple 
    nodes or something. Regardless, at worst it just initializes these things from scratch).
- Updates the `PYTHONPATH` to include the working directory
  - This allows us to reference our own modules throughout the project.
  - Note that for your own local development you will want to set up a venv and update the `PYTHONPATH` in the same 
    way.
  - For new modules, don't forget to include a `__init__.py` in the directory.
