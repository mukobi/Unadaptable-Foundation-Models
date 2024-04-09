# Unadapable-Foundation-Models

## Setup

To setup, simply run `pip install .` from project root

### Development

You can install in dev mode to include additional dependencies for data visualization etc.

```bash
pip install -e ".[dev]"
```

### Weights & Biases (wandb)
The `wandb` package allows for those with an account to visualize various information about their runs. 
However, using the package prompts users to enter their account login info (which they only need to do the first time). 
To disable this prompting (and online features), use the `--disable-wandb` argument in your call.

```bash
python main.py --disable-wandb
```
