import jax
import jax.numpy as jnp
from jax import Array
from matplotlib import pyplot as plt
from functools import partial

# Global variables
MU = 2.0
DT = 0.05
T_END = 20.0
N_PARTICLES = 1000
OBS_STD = 0.3
PROC_STD = 0.05
RND_SEED = 42

# 1. Dynamics
def f_vdp(
    x: jax.Array,
    mu: float = MU,
) -> jax.Array:
    """
    CT Van der Pol RHS. x shape: (2,)
    """
    x1, x2 = x[0], x[1]
    dx1 = x2
    dx2 = mu * (1.0 - x1 ** 2) * x2 - x1
    return jnp.array([dx1, dx2])

def rk4_step(
    x: jax.Array,
    dt: float = DT
) -> jax.Array:
    """
    Single RK4 step for the Van Der Pol ODE.
    """
    k1 = f_vdp(x)
    k2 = f_vdp(x + 0.5 * dt * k1)
    k3 = f_vdp(x + 0.5 * dt * k2)
    k4 = f_vdp(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# 2. Measurement Model
def obs_fn(
    x: jax.Array
) -> jax.Array:
    """
    Returns the scalar x1 only.
    """
    return x[0]

def log_likelihood(
    y: float,
    x: jax.Array,
    std: float = OBS_STD
) -> jax.Array:
    """
    Gaussian log-likelihood of observing y given state x.
    """
    residual = y - obs_fn(x)
    return - 0.5 * (residual / std) **2 - jnp.log(std * jnp.sqrt(2 *jnp.pi))

# 3. Particle Filter steps
@partial(jax.jit, static_argnums=())
def propagate_particles(
    particles: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """
    Propagate each particle throguh RK4 + additive Gaussian process noise.
    """
    particles_pred = jax.vmap(rk4_step)(particles)
    noise = jax.random.normal(key, shape=particles_pred.shape) * PROC_STD
    return particles_pred + noise

@jax.jit
def weight_particles(
    particles: jax.Array,
    y: float,
) -> jax.Array:
    """
    Compute normalized importance weights from log-likelihoods.
    """
    log_w = jax.vmap(partial(log_likelihood, y))(particles)
    log_w -= jnp.max(log_w) # numerically stable soft-max normalization
    w = jnp.exp(log_w)
    return w / jnp.sum(w)

@partial(jax.jit, static_argnums=())
def systematic_resample(
    particles: jax.Array,
    weights: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """
    Resample particles according to their weights using systematic resampling.
    """
    N = particles.shape[0]
    positions = (jnp.arange(N) + jax.random.uniform(key,shape=(1,))) / N
    cumulative_sum = jnp.cumsum(weights)
    indices = jnp.searchsorted(cumulative_sum, positions)
    indices = jnp.clip(indices, 0, N-1) # ensure indices are within bounds
    return particles[indices]

def effective_sample_size(weights: jax.Array) -> float:
    """
    Compute the effective sample size (ESS) from the weights.
    """
    return 1.0 / jnp.sum(weights ** 2)
