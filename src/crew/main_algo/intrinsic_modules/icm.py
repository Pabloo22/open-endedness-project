from typing import Any
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from crew.main_algo.types import (
    IntrinsicModulesUpdateData,
    IntrinsicUpdateMetrics,
    TransitionDataBase,
)
from crew.networks.encoders import ObsEncoderFlatSymbolic


ACTION_DIMS = {
    "Craftax-Classic-Symbolic-v1" : 17,
    "Craftax-Symbolic-v1" : 43
}


class ICMModuleState(struct.PyTreeNode):
    """ICM module state carried by the training loop."""
    icm_train_state: TrainState

def mlp(hidden_dims: list[int], out_dim: int, activation=nn.relu) -> nn.Sequential:
    """Helper function to create an MLP with given hidden dimensions, output dimension, and activation function.
    
    Args:
        hidden_dims: List of hidden layer dimensions.
        out_dim: Output dimension of the MLP.
        activation: Activation function to use between layers (default: nn.relu).
    Returns:
        nn.Sequential module representing the MLP.
    """
    layers = []
    for h in hidden_dims:
        layers += [nn.Dense(h), activation]
    layers += [nn.Dense(out_dim)]
    return nn.Sequential(layers)


class ICMNet(nn.Module):
    """Neural network for the ICM module, containing the observation encoder, forward model, and inverse model."""
    obs_emb_dim: int
    action_dim: int
    forward_hidden_dims: list[int]
    inverse_hidden_dims: list[int]
    activation_fn: str
    
    def setup(self):
        """Initialize the encoder, forward network and inverse network of the ICMNet."""
        self.obs_encoder = ObsEncoderFlatSymbolic(obs_emb_dim=self.obs_emb_dim)
        activation_fn = self.get_activation_fn()
        self.forward_net = mlp(self.forward_hidden_dims, self.obs_emb_dim, activation=activation_fn)
        self.inverse_net = mlp(self.inverse_hidden_dims, self.action_dim, activation=activation_fn)
    
    def get_activation_fn(self) -> callable:
        """
        Helper function to get the activation function based on the config string.
        
        Returns:
            Activation function corresponding to the config string.
        """
        if self.activation_fn == "relu":
            return nn.relu
        elif self.activation_fn == "tanh":
            return nn.tanh
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")
    
    def encode_observations(self, observation: jax.Array) -> jax.Array:
        """
        Encode observations using the obs_encoder.
        
        Args:
            observation: Raw observations from the environment, shape (*batch_dims, obs_dim).
        Returns:            
            Encoded observations, shape (*batch_dims, obs_emb_dim).
        """
        return self.obs_encoder(observations=observation)
    
    def forward(self, z: jax.Array, a: jax.Array) -> jax.Array:
        """
        Forward model to predict next state embedding from current state embedding and action.
        
        Args:
            z: Current state embedding, shape (*batch_dims, obs_emb_dim).
            a: Action taken, shape (*batch_dims,).    
        Returns:
            Predicted next state embedding, shape (*batch_dims, obs_emb_dim).
        """
        a_oh = jax.nn.one_hot(a, self.action_dim)
        x = jnp.concatenate([z, a_oh], axis=-1)
        return self.forward_net(x)
    
    def inverse(self, z: jax.Array, z_next: jax.Array) -> jax.Array:
        """
        Inverse model to predict action taken from current and next state embeddings.
        Args:
            z: Current state embedding, shape (*batch_dims, obs_emb_dim).
            z_next: Next state embedding, shape (*batch_dims, obs_emb_dim).
        Returns:
            Predicted action logits, shape (*batch_dims, action_dim).
        """
        return self.inverse_net(jnp.concatenate([z, z_next], axis=-1))
    
    def init_all(
        self, 
        obs: jax.Array, 
        next_obs: jax.Array, 
        action: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
        """
        Helper function to initialize all subnets of the ICM module.
        
        Args:
            obs: Batch of observations, shape (*batch_dims, obs_dim).
            next_obs: Batch of next observations, shape (*batch_dims, obs_dim).
            action: Batch of actions taken, shape (*batch_dims,).    
        Returns:
            Tuple of (predicted next state embedding, predicted action logits) for the given inputs.
        """
        z = self.encode_observations(obs)
        z_next = self.encode_observations(next_obs)
        z_hat_next = self.forward(z, action)
        a_hat = self.inverse(z, z_next)
        
        return z_hat_next, a_hat
        
class ICMIntrinsicModule:
    """Intrinsic Curiosity Module for intrinsic reward."""
    
    name = "icm"
    is_episodic = True
    
    def init_state(
        self,
        rng: jax.Array,
        obs_shape: tuple[int, ...],
        config: Any,
    ) -> ICMModuleState:
        """
        Initializes module state.
        
        Args:
            rng: Jax random key for initialization.
            obs_shape: Shape of the environment observations.
            config: Config with hyperparameters for the ICM module.
            
        Returns:
            ICMModuleState containing initialized train states for the ICM module.
        """
        init_observations = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        init_actions = jnp.zeros((1,), dtype=jnp.int32)
        icm_net = ICMNet(
            activation_fn=config.icm.activation_fn,
            obs_emb_dim=config.icm.obs_emb_dim,
            action_dim=ACTION_DIMS[config.env_id],
            forward_hidden_dims=config.icm.forward_hidden_dims,
            inverse_hidden_dims=config.icm.inverse_hidden_dims,
        )
        params = icm_net.init(rng, init_observations, init_observations, init_actions, method=ICMNet.init_all)

        icm_train_state = self._init_train_states(
            net=icm_net,
            params=params,
            config=config
            )
        
        return ICMModuleState(
            icm_train_state=icm_train_state
            )

    def _init_train_states(
        self,
        net: nn.Module, 
        params: Any, 
        config: Any
        ) -> TrainState:
        """
        Helper function to initialize train states for the ICM module from config.
        
        Args:
            net: The ICMNet module to apply.
            params: Initialized parameters for the ICMNet.
            config: Config with training hyperparameters for the ICM module.
        
        Returns:
            TrainState initialized with the given parameters and optimizer.
        """
        tx = optax.chain(
            optax.inject_hyperparams(optax.adam)(
                learning_rate= config.icm.lr,
                eps=config.adam_eps,
            ),
        )
        
        icm_train_state = TrainState.create(
            apply_fn=net.apply,
            params=params,
            tx=tx,
        )
        return icm_train_state

    
    def compute_rewards(
        self,
        rng: jax.Array,
        module_state: ICMModuleState,
        transitions: TransitionDataBase,
        config: Any,
    ) -> jax.Array:
        """
        Compute intrinsic rewards for the given transitions.
        
        Args:
            rng: Jax random key, not used in this implementation.
            module_state: Current state of the ICM module containing train states.
            transitions: Batch of transitions for which to compute rewards.
            config: Config with hyperparameters for reward computation.
        Returns:
            Intrinsic rewards for the given transitions, shape (T, B).
        """
        del rng
        net = module_state.icm_train_state.apply_fn        
        params = module_state.icm_train_state.params
        eta = config.icm.reward_eta
        num_chunks = config.icm.num_chunks_in_rewards_computation
        T, B = transitions.next_obs.shape[:2]

        transitions = self._flatten_transitions(transitions)
        chunked_transitions = self._chunk_transitions(transitions, num_chunks)
        
        _, chunked_rewards = jax.lax.scan(
            lambda carry, chunk: (carry, self._compute_reward_chunk(chunk, net, params, eta)),
            None,
            chunked_transitions,
        )
        
        rewards = chunked_rewards.reshape((T, B))
        return rewards
    
    def _compute_reward_chunk(
        self, 
        chunk: TransitionDataBase, 
        net: nn.Module, 
        params: Any, 
        eta: float
    ) -> jax.Array:
        """
        Helper function to compute intrinsic rewards for a chunk of transitions.
        
        Args:
            chunk: A chunk of transitions, with obs, next_obs, action, and done fields
            net: The ICMNet module to apply.
            params: Parameters for the ICMNet.
            eta: Scaling factor for the intrinsic reward.
        Returns:
            Intrinsic rewards for the given chunk, shape (chunk_size,).
        """
        
        z = net(params, chunk.obs, method=ICMNet.encode_observations)
        z_tp1 = net(params, chunk.next_obs, method=ICMNet.encode_observations)
        
        z_tp1_hat = net(params, z, chunk.action, method=ICMNet.forward)
        
        r_int = eta * self._half_sq_l2(z_tp1_hat, z_tp1)
        r_int = r_int * (1.0 - chunk.done.astype(r_int.dtype))
        
        return r_int

    def update(
        self,
        rng: jax.Array,
        module_state: ICMModuleState,
        transitions: IntrinsicModulesUpdateData,
        config: Any,
    ) -> tuple[ICMModuleState, IntrinsicUpdateMetrics]:
        """
        Update module state and return metrics.
        
        Args:
            rng: Jax random key.
            module_state: Current state of the ICM module.
            transitions: Batch of transitions for updating the module.
            config: Config with training hyperparameters.    
        Returns:
            Updated module state and a dictionary of metrics for logging.
        """
        # From config
        net = module_state.icm_train_state.apply_fn
        train_state = module_state.icm_train_state        
        num_minibatches = config.icm.num_minibatches
        update_epochs = config.icm.update_epochs
        beta = config.icm.beta
        eps = config.icm.eps
        
        # Flatten transitions for batching
        transitions = self._flatten_transitions(transitions)
                
        # run train epochs
        (_, new_train_state), epoch_losses = jax.lax.scan(
            lambda c, _ : self._epoch_step(c, transitions, num_minibatches, net, beta, eps),
            (rng, train_state),
            None,
            update_epochs,
        )
        updated_module_state = module_state.replace(
            icm_train_state=new_train_state
            )
        
        loss, fwd, inv = jnp.mean(epoch_losses, axis=0)
        metrics: IntrinsicUpdateMetrics = {
            "intrinsic_modules/icm/loss": loss,
            "intrinsic_modules/icm/forward_loss": fwd,
            "intrinsic_modules/icm/inverse_loss": inv,
        }
        
        return updated_module_state, metrics


    def _epoch_step(self, carry, transitions, num_minibatches, net, beta, eps):
        """
        Perform a single epoch of training on the ICM module.
        
        Args:
            carry: Tuple of (rng, train_state) for the current epoch.
            transitions: Batch of transitions for training, already flattened.
            num_minibatches: Number of minibatches to split the transitions into for training.
            net: The ICMNet module to apply.
            beta: Relative weight of forward loss in the combined loss.
            eps: Small epsilon to avoid division by zero in loss normalization.
        Returns:
            Updated carry with new RNG and train state, and the mean losses for the epoch.
        """
        rng, train_state = carry
        batch_transitions, rng = self._batch_transitions(transitions, num_minibatches, rng)
        
        train_state, batch_losses = jax.lax.scan(
            lambda ts, b: self._minibatch_step(ts, b, net, beta, eps),
            train_state,
            batch_transitions,
        )

        epoch_losses = jnp.mean(batch_losses, axis=0)
        
        return (rng, train_state), epoch_losses

    def _minibatch_step(
        self,
        train_state: TrainState,
        batch: IntrinsicModulesUpdateData,
        net: nn.Module,
        beta: float,
        eps: float
        ) -> tuple[TrainState, jax.Array]: 
        """
        Perform a single optimization step on a minibatch of transitions.
        
        Args:
            train_state: Current TrainState for the ICM module.
            batch: Minibatch of transitions for training.
            net: The ICMNet module to apply.
        
        Returns:
            Updated TrainState and a tuple of (loss, forward_loss, inverse_loss) for logging
        """
        
        (loss, (fwd, inv)), grads = jax.value_and_grad(
            lambda p: self._loss_fn(p, net, batch, beta, eps),
            has_aux=True,
        )(train_state.params)
        
        new_train_state = train_state.apply_gradients(grads=grads)
        return new_train_state, jnp.stack([loss, fwd, inv])

    def _loss_fn(
        self,
        params: jax.Array, 
        net: ICMNet, 
        batch: IntrinsicModulesUpdateData,
        beta: float,
        eps: float
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        Compute the combined forward and inverse loss for a batch of transitions.
        
        Args:
            params: Model parameters.
            net: Neural network function.
            batch: Batch of transitions.
            beta: Relative weight of forward loss.
            eps: epsilon.
        Returns:
            Tuple of (loss, forward_loss, inverse_loss).
        """
        a = batch.action
        z = net(params, batch.obs, method=ICMNet.encode_observations)
        
        z_tp1 = net(params, batch.next_obs, method=ICMNet.encode_observations)
        z_tp1_sg = jax.lax.stop_gradient(z_tp1)

        z_tp1_hat = net(params, z, a, method=ICMNet.forward)
        stepwise_fwd = self._half_sq_l2(z_tp1_hat, z_tp1_sg)
        
        a_logits = net(params, z, z_tp1, method=ICMNet.inverse)
        stepwise_inv = optax.softmax_cross_entropy_with_integer_labels(a_logits, a)
                
        stepwise_loss = beta * stepwise_fwd + (1-beta) * stepwise_inv
        mask = 1.0 - batch.done.astype(jnp.float32)
        mask_denom = (jnp.sum(mask) + eps)
        
        loss = jnp.sum(stepwise_loss * mask) /  mask_denom
        fwd  = jnp.sum(stepwise_fwd * mask)  /  mask_denom
        inv  = jnp.sum(stepwise_inv * mask)  /  mask_denom
        
        return loss, (fwd, inv)
        
        
    def done_mask(self, env_done: jax.Array, config: Any) -> jax.Array:
        """
        Computes the done mask for ICM. Same as environment done mask.
        
        Args: 
            env_done (jax.Array): Done mask from the environment, shape (*batch_dims,).
            config (Any): Config, not used.
        Returns:
            jax.Array: Done mask for ICM, shape (*batch_dims,), where True indicates episode termination.
        """
        del config
        return env_done.astype(jnp.bool_)
    
    @staticmethod
    def _flatten_transitions(transitions: struct.PyTreeNode) -> struct.PyTreeNode:
        """
        Helper function to flatten transitions for training.

        Args:
            transitions: Batch of transitions with shape (T, B, ...).
        Returns:
            Batch of transitions reshaped to (T*B, ...).
        """
        return jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[2:])) if x.ndim >= 3 else x.reshape((-1,)),
            transitions,
        )
    
    @staticmethod
    def _half_sq_l2(arr1: jax.Array, arr2: jax.Array) -> jax.Array:
        """
        Helper function to half squared L2 norm.
        Args:
            arr1: First array, shape (*batch_dims, dim).
            arr2: Second array, shape (*batch_dims, dim).
        Returns:
            Half of the squared L2 norm between arr1 and arr2, shape (*batch_dims,).
        """
        return 0.5 * jnp.sum((arr1 - arr2) ** 2, axis=-1)
    
    def _batch_transitions(
        self,
        transitions: struct.PyTreeNode, 
        num_minibatches: int, 
        rng: jax.Array
    ) -> tuple[struct.PyTreeNode, jax.Array]:
        """
        Helper function to shuffle and batch transitions for training.
        
        Args:
            transitions: Batch of transitions to shuffle and batch.
            num_minibatches: Number of minibatches to create.    
        Returns:
            Transitions reshaped into (num_minibatches, batch_size_per_minibatch, ...)
            RNG key after shuffling.
        """
        num_steps = transitions.action.shape[0]
        
        rng, shuffle_rng = jax.random.split(rng)
        shuffled_indices = jax.random.permutation(shuffle_rng, num_steps)    
        transition_order = jax.tree_util.tree_map(
            lambda x: jnp.take(x, shuffled_indices, axis=0),
            transitions,
        )
        
        transitions_batched = self._chunk_transitions(transition_order, num_minibatches)
        
        return transitions_batched, rng
    
    def _chunk_transitions(
        self,
        transitions: struct.PyTreeNode,
        num_minibatches: int,
    ) -> struct.PyTreeNode:
        """
        Helper function to reshape transitions into batches.
        
        Args:
            transitions: Batch of transitions to reshape, shape (T*B, ...).
            num_minibatches: Number of minibatches to create.    
        Returns:
            Batch of transitions reshaped into (num_minibatches, batch_size_per_minibatch, ...).
        """
        num_steps = transitions.action.shape[0]
        minibatch_size = num_steps // num_minibatches
        chunked_transitions = jax.tree_util.tree_map(
            lambda x: x.reshape((num_minibatches, minibatch_size, *x.shape[1:])),
            transitions,
        )
        return chunked_transitions