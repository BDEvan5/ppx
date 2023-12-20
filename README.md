# PPO Algorithms in JAX

This repo features a high-speed JAX implementation of the Proximal Policy Optimisation (PPO) algorithm.
It also includes that batch size-invariant version, which uses exponetially weighted moving averages to remove dependence on the batch-size hyperparameter.

## Usage

To run a system, you need to execute the following command:
```python
python3 ppox/systems/ppo.py 
```
Since [hydra](https://hydra.cc/docs/intro/) is used for managing configurations, overide parameters can be passed as arguments to this command.
The default parameters can be changes in the relevant config file.

A simple function to plot the return during training is provided in the `notebooks/` directory.


## Installation

We recommend managing dependencies using a virtual environment, which can be installed with the following commands,
```
python3.9 -m venv venv
source venv/bin/activate
```

Install dependencies using the requirements.txt file:

```
pip install -r requirements.txt
```
The codebase is installed as a pip package with the following command:
```
pip install -e .
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).


### Possible improvements
- Add an env wrapper to use the [Jumanji](https://github.com/instadeepai/jumanji/) style step method which returns a `state` and `Timestep`.

## Acknowledgements

The code is based on the format of [Mava](https://github.com/instadeepai/Mava.git) and is inspired from [PureJaxRL](https://github.com/luchris429/purejaxrl) and [CleanRL](https://github.com/vwxyzjn/cleanrl).




