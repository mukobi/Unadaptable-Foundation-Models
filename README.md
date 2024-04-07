# Unadapable-Foundation-Models

## Setup

1. Install PyTorch separately (for local models). You need torch for most experiments and torchvision for the MNIST toy model.
> See https://pytorch.org/get-started/locally/

2. Install requirements
```bash
pip install -r requirements.txt
```

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
