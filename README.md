# thermox
## Standalone JAX-accelerated package to simulate Ornstein-Uhlenbeck processes

This package provides a very simple interface to simulate OU processes of the form 

$$ dx = - A(x - b) dt + \mathcal{N}(0, 2D) $$

To collect samples from this process, define $A, b,D$ ($D$ is optional) a JAX random key, and run the `collect_samples` function:

```
collect_samples(key, A, b, num_samples, D=D) 
```

Samples are then collected by exact diagonalization (therefore there is no discretization error) and JAX scans. The user can also provide `burnin`, tune the time at which samples are collected `dt`, set an initial state, etc.

### Thermodynamic linear algebra

The repository also features a Jupyter notebook, `thermo_linar_algebra.ipynb` where we explain how we can do linear algebra thanks to properties of the OU process. The code can be used to reproduce results from https://arxiv.org/abs/2308.05660. 

### TODOs:

- [X] Add tests for linear systems and inversion
- [X] Merge `collect_samples` and `collect_samples_full_diffusion_matrix`
- [X] Merge code with varying timesteps in Sam's fork

