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

# 4. Ground truth trajectory generation

def generate_ground_truth(
    key: jax.Array,
    n_steps: int
) -> tuple:
    """
    Simulate true tajectory and noisy observations.
    """
    x0 = jnp.array([0.5,0.0])
    xs = [x0]
    ys = []
    obs_key = key
    x = x0
    for _ in range(n_steps):
        x = rk4_step(x)
        obs_key, subkey = jax.random.split(obs_key)
        y = obs_fn(x) + jax.random.normal(subkey) * OBS_STD
        xs.append(x)
        ys.append(y)
    return jnp.stack(xs[:-1]), jnp.array(ys)

# 5. Main filter loop

def run_particle_filter(
    key: jax.Array
) -> dict:
    n_steps = int(T_END / DT)
    times = jnp.arange(n_steps) * DT

    # Ground truth & observations
    key, gt_key = jax.random.split(key)
    true_states, observations = generate_ground_truth(gt_key, n_steps)

    # Initialize particles around a prior (e.g. broad Gaussian near x0)
    key, init_key = jax.random.split(key)
    particles = jax.random.normal(
        init_key,
        shape=(N_PARTICLES,2)
    ) + 0.5 * jnp.array([0.5,0.0])

    mean_estimates = []
    std_estimates = []
    ess_trace = []

    for t in range(n_steps):
        # propagate particles
        key, prop_key = jax.random.split(key)
        particles = propagate_particles(particles, prop_key)

        # weights
        weights = weight_particles(particles, observations[t])

        # diagnostics
        ess = float(effective_sample_size(weights))
        ess_trace.append(ess)

        # posterior summary
        mean_st = jnp.sum(weights[:,None] * particles, axis=0)
        var_est = jnp.sum(weights[:,None] * (particles - mean_st) ** 2, axis=0)

        mean_estimates.append(mean_st)
        std_estimates.append(jnp.sqrt(var_est))

        # 5. Resample (always here, add ESS threshold if you want to )
        key, rs_key = jax.random.split(key)
        particles = systematic_resample(particles, weights, rs_key)

    return {
        "times": times,
        "true_states": true_states,
        "obs": observations,
        "mean": jnp.stack(mean_estimates),
        "std": jnp.stack(std_estimates),
        "ess": jnp.array(ess_trace)
    }

# 6. Visualize all of the things
def plot_results(res: dict) -> None:
    t   = res["times"]
    tr  = res["true_states"]
    obs = res["obs"]
    mu  = res["mean"]
    sig = res["std"]
    ess = res["ess"]
 
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)
    fig.suptitle(f"Bootstrap PF  —  Van der Pol (μ={MU}, N={N_PARTICLES})", fontsize=13)
 
    # x1 estimate
    ax = axes[0]
    ax.plot(t, tr[:, 0], "k-",  lw=1.5, label="True $x_1$")
    ax.scatter(t, obs, s=4, color="grey", alpha=0.5, label="Obs $y_k$")
    ax.plot(t, mu[:, 0], "C0-", lw=1.5, label="PF mean $x_1$")
    ax.fill_between(t, mu[:, 0] - 2*sig[:, 0], mu[:, 0] + 2*sig[:, 0],
                    alpha=0.25, color="C0", label="±2σ")
    ax.set_ylabel("$x_1$")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
 
    # x2 estimate (unobserved!)
    ax = axes[1]
    ax.plot(t, tr[:, 1], "k-",  lw=1.5, label="True $x_2$")
    ax.plot(t, mu[:, 1], "C1-", lw=1.5, label="PF mean $x_2$")
    ax.fill_between(t, mu[:, 1] - 2*sig[:, 1], mu[:, 1] + 2*sig[:, 1],
                    alpha=0.25, color="C1", label="±2σ")
    ax.set_ylabel("$x_2$ (latent)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
 
    # Phase portrait
    ax = axes[2]
    ax.plot(tr[:, 0],  tr[:, 1],  "k-",  lw=1,   alpha=0.7, label="True")
    ax.plot(mu[:, 0],  mu[:, 1],  "C0-", lw=1.2, alpha=0.8, label="PF mean")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Phase portrait")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    # ESS over time
    ax = axes[3]
    ax.plot(t, ess / N_PARTICLES, "C2-", lw=1)
    ax.axhline(0.5, ls="--", color="red", lw=0.8, label="ESS = 0.5N")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESS / N")
    ax.set_title("Effective Sample Size (relative)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("vdp_pf_results.png", dpi=150)
    print("Saved vdp_pf_results.png")
    plt.show()

# Run file
if __name__ == "__main__":
    key = jax.random.PRNGKey(RND_SEED)
    print(f"Running PF: N={N_PARTICLES}, T={T_END}s, dt={DT}s, obs_std={OBS_STD}, proc_std={PROC_STD}")
    results = run_particle_filter(key)
    # Print shapes of all parts of dict
    print("Shape of all parts of results dict:")
    print({k: v.shape for k, v in results.items()})

    rmse_x1 = jnp.sqrt(jnp.mean((results["mean"][:, 0] - results["true_states"][:, 0]) ** 2))
    rmse_x2 = jnp.sqrt(jnp.mean((results["mean"][:, 1] - results["true_states"][:, 1]) ** 2))
    mean_ess = jnp.mean(results["ess"])
    print(f"RMSE x1: {rmse_x1:.3f}, RMSE x2: {rmse_x2:.3f}, Mean ESS: {mean_ess:.1f}")
    plot_results(results)
