# thermox
## Standalone JAX-accelerated package to simulate Ornstein-Uhlenbeck processes

This package provides a very simple interface to **exactly** simulate [Ornstein-Uhlenbeck](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) (OU) processes of the form 

$$ dx = - A(x - b) dt + \mathcal{N}(0, D dt) $$

To collect samples from this process, define sampling times `ts`, initial state `x0`, drift matrix `A`, mean displacement vector `b`, diffusion matrix `D` and a JAX random key. Then run the `collect_samples` function:

```
collect_samples(key, ts, x0, A, b, D) 
```
Samples are then collected by exact diagonalization (therefore there is no discretization error) and JAX scans. 

Here is a simple code example for a 5-dimensional OU process:
```python
import thermox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Set random seed
key = jax.random.PRNGKey(0)

# Timeframe
dt = 0.01
ts = jnp.arange(0, 1, dt)

# System parameters for a 5-dimensional OU process
A = jnp.array([[2.0, 0.5, 0.0, 0.0, 0.0],
               [0.5, 2.0, 0.5, 0.0, 0.0],
               [0.0, 0.5, 2.0, 0.5, 0.0],
               [0.0, 0.0, 0.5, 2.0, 0.5],
               [0.0, 0.0, 0.0, 0.5, 2.0]])

b, x0 = jnp.zeros(5), jnp.zeros(5) # Zero drift displacement vector and initial state

 # Diffusion matrix with correlations between x_1 and x_2
D = jnp.array([[2, 1, 0, 0, 0],
               [1, 2, 0, 0, 0],
               [0, 0, 2, 0, 0],
               [0, 0, 0, 2, 0],
               [0, 0, 0, 0, 2]])

# Collect samples
samples = thermox.collect_samples(key, ts, x0, A, b, D)

plt.figure(figsize=(12, 5))
plt.plot(ts, samples, label=[f'Dimension {i+1}' for i in range(5)])
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Trajectories of 5-Dimensional OU Process')
plt.legend()
plt.show()
```

<p align="center">
  <img src="https://storage.googleapis.com/normal-blog-artifacts/thermo-playground/ou_trajectories.png" width="600" lineheight = -10%/>
  <br>
</p>

### Thermodynamic linear algebra

The repository also features a Jupyter notebook, `thermo_linar_algebra.ipynb` where we explain how we can do linear algebra thanks to properties of the OU process. The code can be used to reproduce results from https://arxiv.org/abs/2308.05660. 


