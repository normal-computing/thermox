# Examples

You will find three Jupyter notebooks in this folder:

- `associative_scan.ipynb` plots the speedup of the associative scan algorithm implemented in `thermox` using a GPU.
- `diffrax_comparison.ipynb` runs a simple OU process using `thermox` and [`diffrax`](https://github.com/patrick-kidger/diffrax) and compares runtimes (showing a large benefit from using `thermox` for long simulation times)
- `thermodynamic_linear_algebra.ipynb` is a small tutorial on how to use functions from the `thermox.linalg` module.

Additionally the `matrix_exponentials` folder contains code for reproducing the simulations 
in the [thermodynamic matrix exponentials paper](https://arxiv.org/abs/2311.12759).