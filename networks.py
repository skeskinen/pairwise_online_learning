from functools import partial
from typing import Tuple

from flax import linen as nn
from jax import numpy as jnp
from jax import random

import jax

from typing import Any

def wta_p_subtract(x, p):

    pick_k = int(x.shape[-1] * p)
    k, _ = jax.lax.top_k(x, pick_k)
    if x.ndim == 1:
        v = k[-1]
    else:
        v = jnp.expand_dims(k[:, :, -1], -1)
    return jax.nn.relu(x - v)

def hard_ash(x, hyperparams):
    mean = jnp.mean(x)
    std = jnp.std(x)
    alpha = 3.
    z = jnp.clip(x, a_min=0., a_max=2.) * jax.nn.sigmoid(alpha * (x - mean - hyperparams['ash_zk'] * std))
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

class Ash(nn.Module):
    @nn.compact
    def __call__(self, x):
        mean = jnp.mean(x)
        std = jnp.std(x)
        alpha = self.param('alpha', nn.initializers.ones, ())
        zk = self.param('zk', nn.initializers.zeros, ())
        # z = jnp.clip(x, a_min=0., a_max=2.) * jax.nn.hard_sigmoid(alpha * (x - mean - zk * std))
        z = x * jax.nn.sigmoid(alpha * (x - mean - zk * std))
        return z
    
class MLP(nn.Module):
    dim: Any
    layers: int
    act: Any
    init_fn: Any

    @nn.compact
    def __call__(self, x):
        x = jnp.ravel(x)
        for i in range(self.layers):
            x = nn.Dense(int(self.dim), use_bias=False, kernel_init=self.init_fn, name=f'dense_{i}')(x)
            x = self.act()(x)
        return x

class CNN(nn.Module):
    dims: Any
    strides: Any
    sizes: Any
    act: Any
    init_fn: Any

    extra_fc: Any = None

    @nn.compact
    def __call__(self, x):
        layers = len(self.dims)
        for i, (d, k, s) in enumerate(zip(self.dims, self.sizes, self.strides)):
            x = nn.Conv(int(d), kernel_size=(k, k), strides=s, use_bias=False, kernel_init=self.init_fn, name=f'conv_{i}')(x)
            x = self.act()(x)
        if self.extra_fc is not None:
            x = jnp.ravel(x)
            x = nn.Dense(self.extra_fc, use_bias=False, kernel_init=self.init_fn, name=f'dense_extra')(x)
            x = self.act()(x)
        return x

class Block(nn.Module):
    features: int
    use_bias: bool
    act: Any
    kernel_init: Any

    @nn.compact
    def __call__(self, x):
        input = x
        x = nn.Conv(self.features, (5, 5), padding=2, feature_group_count=self.features, use_bias=self.use_bias, kernel_init=self.kernel_init)(x)  # depthwise conv
        x = nn.LayerNorm()(x)
        x = nn.Dense(4 * self.features, use_bias=self.use_bias, kernel_init=self.kernel_init)(x)
        x = self.act()(x)
        # x = nn.gelu(x)
        x = nn.Dense(self.features, use_bias=self.use_bias, kernel_init=self.kernel_init)(x)
        # gamma = self.param('gamma', nn.initializers.constant(1e-6), (self.features,))
        # x = gamma * x
        # x = DropPath(drop_path)(x) if drop_path > 0. else x
        return x + input


class ConvNeXt(nn.Module):
    final_dim: int = 0
    strides: Tuple[int] = (2, 2, 2)
    # ~70%
    depths: Tuple[int] = (2, 6, 2)
    dims: Tuple[int] = (64, 128, 256)
    # depths: Tuple[int] = (3, 9, 3)
    # dims: Tuple[int] = (96, 192, 384)

    # strides: Tuple[int] = (4, 2)
    # depths: Tuple[int] = (6, 2)
    # dims: Tuple[int] = (128, 256)
    
    act: Any = None
    init_fn: Any = None

    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        for i, (stride, depth, dim) in enumerate(zip(self.strides, self.depths, self.dims)):
            if i == 0:
                x = nn.Conv(dim, kernel_size=(stride, stride), strides = stride, use_bias=self.use_bias, kernel_init=self.init_fn, name=f'down_conv_{i}')(x)
                x = nn.LayerNorm(name=f'down_norm_{i}')(x)
            else:
                x = nn.LayerNorm(name=f'down_norm_{i}')(x)
                x = nn.Conv(dim, kernel_size=(stride, stride), strides = stride, use_bias=self.use_bias, kernel_init=self.init_fn, name=f'down_conv_{i}')(x)

            for b in range(depth):
                x = Block(dim, use_bias=self.use_bias, act=self.act, kernel_init=self.init_fn, name=f'block_{i}_{b}')(x)

        x = jnp.mean(x, axis=(0, 1))
        x = nn.LayerNorm(name='final_norm')(x)    
        x = nn.Dense(self.final_dim, use_bias=self.use_bias, kernel_init=self.init_fn, name ='final_fc')(x)
        return x
