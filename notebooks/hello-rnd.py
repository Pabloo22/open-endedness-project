# RND implementation in JAX

# 1. target network produces random but consistent numbers
# 2. predictor network tries to memorize what target network outputs for each state
# 3. new states yield high error rate from predictor network -> novel states
# 4. reoccuring states yield low error rate from predictor network -> boring states

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from matplotlib import pyplot as plt


# 1. Define target and predcitor network
class RNDNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        return x


key = jax.random.PRNGKey(42)
key, target_key, predictor_key = jax.random.split(
    key, 3
)  # Global key, key for target network inference, key for predictor network training + inference

dummy_input = jnp.zeros((1, 2))
print(dummy_input)

target_params = RNDNetwork().init(
    target_key, dummy_input
)  # Different params for target!
predictor_params = RNDNetwork().init(
    predictor_key, dummy_input
)  # Different params for predictor!


# 2. We define the intrinsic reward function
def intrinsic_reward(target_params, predictor_params, states):
    target_out = RNDNetwork().apply(target_params, states)
    predictor_out = RNDNetwork().apply(predictor_params, states)

    reward = jnp.mean((target_out - predictor_out) ** 2, axis=-1)
    return reward


# 3. Training loop
def loss_fn(predictor_params, target_params, states):
    return jnp.mean(intrinsic_reward(target_params, predictor_params, states))


# optimizer needs to know the parameters of the predictor network
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(predictor_params)  # initialize optimizer parameters


@jax.jit
def train_step(predictor_params, opt_state, target_params, states):
    # Compute the loss with respect to the predictor network using custom loss_fn
    loss, grads = jax.value_and_grad(loss_fn)(
        predictor_params, target_params, states
    )  # Compute loss
    updates, opt_state = optimizer.update(grads, opt_state)  # prepare updates
    predictor_params = optax.apply_updates(predictor_params, updates)
    return predictor_params, opt_state, loss


# Generate data and train
key, data_key = jax.random.split(key)
train_points = jax.random.uniform(
    data_key, (1000, 2)
)  # 1000 points in 2D explored region from 0 to 1

for i in range(500):
    predictor_params, opt_state, loss = train_step(
        predictor_params, opt_state, target_params, train_points
    )
    if i % 100 == 0:
        print(f"Step {i}: Loss {loss:.8f}")

# Visualize the HeatMap for intrinsic reward
print(
    "The Cyan dashed lines show the training regions i.e. the regions where the target network is able to memorize the predictor network."
)
x = jnp.linspace(-2, 2, 100)
y = jnp.linspace(-2, 2, 100)
xx, yy = jnp.meshgrid(x, y)
grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)


rewards = intrinsic_reward(target_params, predictor_params, grid_points)
rewards = rewards.reshape(100, 100)

plt.figure(figsize=(8, 6))
plt.imshow(rewards, extent=[-2, 2, -2, 2], origin="lower", cmap="hot")
plt.colorbar(label="Intrinsic Reward")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("RND Intrinsic Reward (dark = familiar, bright = novel)")
plt.axhline(y=0, color="cyan", linestyle="--", alpha=0.5)
plt.axhline(y=1, color="cyan", linestyle="--", alpha=0.5)
plt.axvline(x=0, color="cyan", linestyle="--", alpha=0.5)
plt.axvline(x=1, color="cyan", linestyle="--", alpha=0.5)
plt.savefig("rnd_heatmap.png", dpi=150)
plt.show()
