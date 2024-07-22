# Thermodynamic Matrix Exponentials Simulations

This folder contains code to replicate the convergence time simulations in the
[thermodynamic matrix exponentials paper](https://arxiv.org/abs/2311.12759).

The script `run.py` contains the code to run the simulations and can be executed with 
e.g. the following command from the root of the repository:
```bash
PYTHONPATH=. python examples/matrix_exponentials/run.py --n_repeats=10 --matrix_type=orthogonal --alpha=1.1   
```
where `--matrix_type` represents a function in `matrix_generation.py`.

The script `plot.py` contains the code to plot the results of the simulations and can be executed with e.g. the following command from the root of the repository:
```bash
PYTHONPATH=. python  examples/matrix_exponentials/plot.py --save_dir=examples/matrix_exponentials/results_orthogonal.pkl
```

