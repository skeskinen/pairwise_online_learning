# This is old code and doesn't follow the patterns that are used in the updated main file.
# Apologies in advance if you try to run or modify this.

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import optax
from typing import Any

import flax
from flax.core import FrozenDict
import flax.linen as nn

import math
import numpy as np
import numpy.random as npr
from tqdm import tqdm
import wandb
import sys

from collections import deque
from functools import partial

from einops import rearrange

from experiments import train_loader, test_loader, input_size

BATCH_SIZE = 64
SPLIT = True
CONV = False

NUM_CLASSES = 10

def top_p_subtract(x, hyperparams):
    pick_k = int(x.shape[-1] * hyperparams['top_p'])
    k, _ = jax.lax.top_k(x, pick_k)
    if x.ndim == 1:
        v = k[-1]
    else:
        v = jnp.expand_dims(k[:, :, -1], -1)
    # if hyperparams['clip_top_k']:
    #     return jnp.clip(x - v, 0., 2.)
    return jax.nn.relu(x - v)

class PairwiseExpand(nn.Module):
    features: int

    def setup(self):
        self.rows, self.cols = jnp.tril_indices(self.features, k = -1)

    def __call__(self, x):
        z = jnp.outer(x, x)
        z = z[self.rows, self.cols]
        return z

class PairwiseLinear(nn.Module):
    in_features: int
    features: int
    weights_init: Any
    size: Any = None

    def setup(self):
        if self.size is None:
            l = self.in_features * (self.in_features - 1) // 2
            l = l - (l % self.features)
        else:
            l = self.size
            if self.size % self.features != 0:
                print('Pairwise size is not divisible with output features. Rounding down.')
                l -= l % self.features
        is_initialized = self.has_variable('pairwise', 'rows')
        self.rows = self.variable('pairwise', 'rows', jnp.zeros, (l,))
        self.cols = self.variable('pairwise', 'cols', jnp.zeros, (l,))
        if not is_initialized:
            rows, cols = jnp.tril_indices(self.in_features, k = -1)
            r = jnp.arange(0, rows.shape[0])
            if len(r) < l:
                r = jnp.repeat(r, l // len(r) + 1, total_repeat_length=l)
            r = random.permutation(random.PRNGKey(1234), r)
            self.rows.value = rows[r][:l]
            self.cols.value = cols[r][:l]

        self.weights = self.param(
            'weights',
            self.weights_init,
            (l // self.features, self.features)
        )

    def __call__(self, x):
        z = x[self.rows.value] * x[self.cols.value]
        z = z.reshape(-1, self.features)
        z = z * self.weights
        # z = jnp.pad(z, (0, self.padding))
        
        return jnp.sum(z, axis = 0)

init_fn = nn.initializers.he_uniform()
# init_fn = nn.initializers.constant(0.5)
#init_fn = nn.initializers.normal(0.001) # gave better results, but why?
class Model(nn.Module):

    hyperparams: FrozenDict

    @nn.compact
    def __call__(self, x, out_reprs=False):
        outputs = []
        h_size = self.hyperparams['mlp_hidden_size']

        for i in range(self.hyperparams['layer_count']):
            if i == 0:
                if CONV:
                    stride = 2
                    z = nn.Conv(32, (stride, stride), strides=stride, use_bias=False, kernel_init=nn.initializers.he_normal(), name=f'conv_{i}_0')(x)
                    z = nn.Conv(64, (stride, stride), strides=stride, use_bias=False, kernel_init=nn.initializers.he_normal(), name=f'conv_{i}_1')(z)
                    # z = top_p_subtract(z, self.hyperparams)

                    # z = nn.Conv(48, (stride, stride), strides=stride, use_bias=False, kernel_init=nn.initializers.he_normal(), name=f'conv_{i}_1')(z)
                    # s = z.shape
                    # z = jnp.ravel(z)
                    # z = top_p_subtract(z, self.hyperparams)
                    # z = PairwiseLinear(z.shape[-1], z.shape[-1], init_fn, 100_000, name=f'test_pairwise')(z)

                    # z = z.reshape(s)

                    # z = nn.Conv(48, (3, 3), strides=1, use_bias=False, kernel_init=nn.initializers.he_normal(), name=f'conv_{i}_2')(z)

                else:
                    x = jnp.ravel(x)
                    z = nn.Dense(h_size, use_bias=False, kernel_init=nn.initializers.he_normal(), name=f'dense_{i}')(x)
                    # z = nn.Dense(h_size, use_bias=False, kernel_init=nn.initializers.he_normal(), name=f'dense_{i}2')(z)
                    # z = nn.Dense(h_size, use_bias=False, kernel_init=nn.initializers.he_normal(), name=f'dense_{i}3')(z)
                    # z = nn.Dense(h_size, use_bias=False, kernel_init=nn.initializers.he_normal(), name=f'dense_{i}4')(z)
            else:
                z = nn.Dense(h_size, use_bias=False, kernel_init=nn.initializers.he_normal(), name=f'dense_{i}')(z)
                # z = PairwiseLinear(z.shape[-1], h_size, weights_init=init_fn, size=2_000_000, name=f'pairwise_{i}')(z)

            # z = hard_ash(z, self.hyperparams)
            z = top_p_subtract(z, self.hyperparams)
            # z = nn.relu(z)
            if out_reprs:
                outputs.append(z)

        if CONV:
            z = jnp.ravel(z)
        z = PairwiseLinear(z.shape[-1], NUM_CLASSES, weights_init=init_fn, size=self.hyperparams['output_layer_weights'], name=f'pairwise_output')(z)
        # z = nn.Dense(NUM_CLASSES, use_bias=False, kernel_init=init_fn, name=f'dense_output')(z)

        if out_reprs:
            return z, jnp.array(outputs)
        else:
            return z

@partial(jax.jit, static_argnums=(0))
def step_classifier(statics, weights, state, omega, old_weights, xs, ys):
    model, hyperparams = statics

    @jax.value_and_grad
    def classifier_loss(weights, xs, ys):
        @jax.vmap
        def go(x):
            return model.apply({'params': weights, **state}, x)

        logits = go(xs)
        
        class_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=ys))
        mas_losses = jtu.tree_map(lambda x, y, o: hyperparams['mas_lambda'] * jnp.sum(o * (x - y) ** 2.0), weights, old_weights, omega)
        mas_loss = jtu.tree_reduce(lambda acc, x: acc + x, mas_losses, 0.)

        return class_loss + mas_loss

    loss, gs = classifier_loss(weights, xs, ys)

    lr = hyperparams['learning_rate']
    updates = jtu.tree_map(lambda g: -lr * g, gs)

    weights = optax.apply_updates(weights, updates)

    return loss, weights

@partial(jax.jit, static_argnums=(0))
def step_mas(statics, weights, state, acc_importance, xs):
    model, hyperparams = statics

    @jax.grad
    def mas_o_squared(weights, xs):
        @jax.vmap
        def go(x):
            return model.apply({'params': weights, **state}, x)

        logits = go(xs)
        return jnp.mean(logits ** 2.0)
    importance = mas_o_squared(weights, xs)
    acc_importance = optax.apply_updates(acc_importance, jtu.tree_map(jnp.abs, importance))
    return acc_importance


@partial(jax.jit, static_argnums=(0,))
def evaluate_step(model, xs, ys, weights, state):
    @jax.vmap
    def go(x):
        return model.apply({'params': weights, **state}, x, out_reprs=True)

    logits, outputs = go(xs)
    predicted_classes = jnp.argmax(logits, axis=1)

    correct_predictions_mask = predicted_classes == ys

    correct_counts = jnp.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        correct_counts = correct_counts.at[c].set(jnp.sum(correct_predictions_mask & (predicted_classes == c)))

    total_counts = jnp.bincount(ys, length=NUM_CLASSES)

    return correct_counts, total_counts, outputs

def evaluation(statics, weights, state):
    model, hyperparams = statics
    
    correct_counts = jnp.zeros(NUM_CLASSES)
    total_counts = jnp.zeros(NUM_CLASSES)

    representations = []
    for xs, ys in test_loader('mnist', BATCH_SIZE):
        correct, total, repr = evaluate_step(model, xs, ys, weights, state)
        representations.append(repr)
        correct_counts += correct
        total_counts += total
    
    representations = jnp.concatenate(representations)
    accuracy = jnp.sum(correct_counts) / jnp.sum(total_counts)
    task_correct = jnp.sum(jnp.reshape(correct_counts, (5, 2)), axis=1)
    task_total = jnp.sum(jnp.reshape(total_counts, (5, 2)), axis=1)
    return accuracy.item(), task_correct / task_total, representations

def train(hyperparams):
    hyperparams = FrozenDict(hyperparams)

    key = random.PRNGKey(5678)
    
    model = Model(hyperparams)
    

    dummy_input = jnp.zeros(input_size('mnist'))
    variables = model.init(key, dummy_input)
    state, weights = flax.core.pop(variables, 'params')
    print(f'Model has {jtu.tree_reduce(lambda x, y: x + np.size(y), weights, 0):,} weights')
    # opt_state = optimizer.init(weights)
    omega = jtu.tree_map(lambda x: jnp.zeros_like(x), weights)

    statics = (model, hyperparams)

    step = 0
    old_weights = weights
    if SPLIT:
        for split in tqdm(range(5)):
            
            for e in range(hyperparams['epochs']):
                for xs, ys in train_loader('mnist', BATCH_SIZE, split):
                    step += 1
                    loss, weights = step_classifier(statics, weights, state, omega, old_weights, xs, ys)
                    # print(loss)
                    if np.isnan(loss):
                        raise Exception(f'loss is nan')
            accuracy, per_task_accuracy, _ = evaluation(statics, weights, state)
            # if not is_sweep:
            log({
                'accuracy': accuracy,
                'task_0_accuracy': per_task_accuracy[0].item(),
                'task_1_accuracy': per_task_accuracy[1].item(),
                'task_2_accuracy': per_task_accuracy[2].item(),
                'task_3_accuracy': per_task_accuracy[3].item(),
                'task_4_accuracy': per_task_accuracy[4].item(),
            }, step)
            if split > 0 and accuracy < 0.25:
                return
            acc_importance = jtu.tree_map(lambda x: jnp.zeros_like(x), weights)
            for xs, ys in train_loader('mnist', BATCH_SIZE, split):
                step += 1
                acc_importance = step_mas(statics, weights, state, acc_importance, xs)

            omega = optax.apply_updates(omega, acc_importance)
            old_weights = weights
    else:
        raise Exception()

    jax.clear_backends()

def log(x, step):
    print(x)
    if wandb.run is not None:
        wandb.log(x, step=step)
def V(x):
    return {'value': x} if is_sweep else x
def Vs(xs, manual_value=None):
    return {'values': xs} if is_sweep else manual_value if manual_value is not None else np.random.choice(xs).item()
def U(a, b, distribution = 'uniform', manual_value = None):
    if distribution not in ['uniform', 'q_uniform', 'log_uniform_values', 'inv_log_uniform_values']:
        raise Exception('set valid distribution')
    return {'distribution': distribution, 'min': a, 'max': b} if is_sweep else manual_value if manual_value is not None else a + (b - a) / 2.0

# Configs
is_sweep = False
log_run = False

optimizers = {
    'sgd' : {
        'learning_rate': Vs([1e-3, 2e-3, 4e-3], manual_value=5e-4), #logU(4e-6, 6e-6),
    },
    'adagrad' : {
        'learning_rate': Vs([3e-3, 5e-3], manual_value=3e-4), #logU(4e-6, 6e-6),
    },
    's-mas' : {
        'mas_root': V(True),
        'learning_rate': Vs([3e-3, 5e-3], manual_value=4e-4), #logU(4e-6, 6e-6),
        'mas_lambda': V(0.05),
        
        # 'mas_root': V(False),
        # 'learning_rate': Vs([3e-3, 5e-3], manual_value=9e-3),
        # 'mas_lambda': V(0.1)
    },
    
}

optimizers_to_run = ['sgd']
# optimizers_to_run = ['adam']
project = 'mnist split, pairwise, permute'
for chosen_optimizer_config in optimizers_to_run:

    sweep_config = {
        'method': 'grid',
        'name': f'{chosen_optimizer_config}',
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },

        'parameters': {
            'top_p': Vs([0.09, 0.13, 0.16, 0.2], manual_value=0.15),

            # 'loss_weight': V(0.01),
            'mlp_hidden_size': V(700),
            # 'layer_count': Vs([1, 2], manual_value=1),
            'layer_count': V(1),
            'output_layer_weights': V(100_000),

            'mas_lambda': V(0.3),

            'permutations': V(10),
            'epochs': V(1),
        }
    }

    sweep_config['parameters']['optimizer'] = V(chosen_optimizer_config)
    sweep_config['parameters'].update(optimizers[chosen_optimizer_config])
            
    # Function to be called by wandb sweep agent
    def sweep_train():
        with wandb.init() as run:
            jax.clear_caches()
            hyperparams = wandb.config
            train(hyperparams)

    # if is_sweep:
    #     # sweep_id = wandb.sweep(sweep_config, project=project, entity='')
    #     wandb.agent(sweep_id, sweep_train)
    # else:
    hyperparams = sweep_config['parameters']
    print(hyperparams)
    # if log_run:
    #     run = wandb.init(project=project, entity='', config = hyperparams, name=sweep_config['name'])

    train(hyperparams)
        