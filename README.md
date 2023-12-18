# PPO Algorithms in JAX

This repo features a high-speed JAX implementation of the Proximal Policy Optimisation (PPO) algorithm.
It also includes that batch size-invariant version, which uses exponetially weighted moving averages to remove dependence on the batch-size hyperparameter.

## Usage

To run a sysem, 


## Installation

Install dependencies using the requirements.txt file:

```
pip install -r requirements.txt
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

We recommend managing dependencies using a virtual environment, which can be installed with the following commands,
```
python3.9 -m venv venv
source venv/bin/activate
```

## Acknowledgements

The code is based on the format of [Mava](https://github.com/instadeepai/Mava.git) and is inspired from [PureJaxRL](https://github.com/luchris429/purejaxrl) and [CleanRL](https://github.com/vwxyzjn/cleanrl).




