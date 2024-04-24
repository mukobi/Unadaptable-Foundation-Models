# Unadaptable-Foundation-Models

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
To disable this prompting (and online features), set `disable_wandb: true` in your config or pass it in directly at 
runtime 
```bash
python main.py disable_wandb=true
````

#### Sweeps
See point 5 in Configs below.


### Configs

Configs allow you to configure a run or provide a template to work off of. Leveraging `hydra`, we have a powerful
config system allowing for very detailed runs if you like, but is also simple for quickly creating new runs.

#### 1. `hydra.main`

The `@hydra.main` wrapper

```python
@hydra.main(version_base=None, config_path="configs", config_name="base_config")
```

wraps around our `main` function and seamlessly allows for config parameter changes
"on-the-fly" as arguments to `main.py`, for example

```bash
python main.py pretrain.epochs=20
```

overwrites the `pretrain.epochs` parameter. 
The wrapper specifies `"configs"` as the config path. This is important because it tells `hydra` where the root 
directory is, which allows configs to reference other configs within, among other things. 

#### 2. Inheritance

You can build a config file through inheritance of another to override certain params, or reference another file to 
incorporate it as a structure. Consider dir structure like the following, and then the `mnist.yaml` config.
```text
├── configs
│   ├── base_config.yaml
│   ├── mnist.yaml
│   ├── finetune
│   │   ├── fashion.yaml
│   ├── pretrain
│   │   ├── mnist.yaml
│   ├── unadapt
│   │   ├── gradient.yaml
└── main.py
```
```yaml
# mnist.yaml
defaults:
  - base_config
  # Notice the keywords below match config dirs
  - pretrain: mnist
  - unadapt: gradient
  - /finetune/fashion
  - _self_

# Base name identifying the run
basename: mnist  
```
- `defaults:` is a special param that `hydra` picks up for importing configs
- First we reference the `base_config.yaml` that provides some params like `seed: 0` etc.
- Then we import three other config files as additional params.
- Finally, we overwrite `basename`, which would normally be `basename: default` from `base_config.yaml`

The final config would look like
```yaml
seed: 0
model: MLP
pretrained: false
basename: mnist
pretrain:
  dataset: MNIST
  epochs: 8
  lr: 0.001
  batch_size: 128
  test_batch_size: 1000
unadapt:
  method: gradient
  loss: fim
  reduce: trace_max
  lam: 0.1
  lr: 0.001
  epochs: 1
finetune:
  dataset: FashionMNIST
  epochs: 1
  lr: 0.001
  batch_size: 128
  test_batch_size: 1000
```
Notice how the three `pretrain/unadapt/finetune` structs are automatically named. This actually _isn't_ because we 
wrote `pretrain: ...`, but is because of the configs directory structure (`pretrain:` just tells hydra to look in 
the `configs/pretrain` directory for `"mnist.yaml"`). I also include `/finetune/fashion` to further demonstrate this 
directory-lookup behavior.

#### 3. Overrides at runtime to `main.py`

If we just rain
```bash
python main.py
```
we would load the default `base_config.yaml` (because that is specified in the `hydra.main` wrapper).  
To work from a different config file we call
```bash
python main.py --config-name=mnist
```
This will be found because `mnist.yaml` is in the `configs` root. We can override any of the parameters loaded from 
the command line
```bash
python main.py --config-name=mnist pretrain.epochs=20 seed=8675309
# Selecting a different unadapt method
python main.py --config-name=mnist unadapt=prune
```

#### 4. Experiments

It can be helpful to conceptually separate our config structures from "experiment" configs that would be used for 
defining runs, where you may want to indicate a logging directory, alternate run name etc. 
You can use the `+experiment` argument with hydra to specify a config that has additional params for your given run.
For example you might have an experiment config with overrides for a fast and small run in 
`configs/experiment/fast_small.yaml`:
```bash
python main.py --config-name=mnist +experiment=fast_small
```
Note:
- Your experiment should exist in the `experiment` dir
- Your conf should have `# @package _global_` at the top so that its parameters are imported at root level structure 

#### 5. Sweeps
With WandB you can run hyperparameter sweeps with the `wandb sweep` command.  
Configuring this will take some coding though.
See: https://docs.wandb.ai/guides/sweeps/walkthrough

Note there is also some finagling to get this working with Hydra Experiments.
