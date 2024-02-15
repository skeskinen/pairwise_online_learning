import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import optax

import flax
from flax.core import FrozenDict
import flax.linen as nn

import math
import numpy as np
import numpy.random as npr
import scipy

from tqdm import tqdm
import wandb
import sys
import json

from collections import deque
from functools import partial
import os
import sys

from einops import rearrange

from networks import *
from experiments import get_experiment_hyperparams, train_loader, test_loader, input_size

HYPERPARAMETER_SWEEP = False
LOG_TO_WANDB = False
wandb_project = 'streaming_continual_learning_2'
wandb_user = ''

def run_training():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            config = json.load(file)

        if LOG_TO_WANDB:
            wandb.init(project=wandb_project, entity=wandb_user, config = config, name = None)

        return train_with_hyperparams(config)

    config = {
        # 'experiment': 'permuted_mnist',
        'experiment': 'split_mnist',
        # 'experiment': 'split_fashion_mnist_mh',
        # 'experiment': 'split_cifar10_mh',
        # 'experiment': 'split_cifar10',
        # 'architecture': 'cnn-e|fc',
        # 'architecture': 'cnn-s|100000',
        # 'architecture': 'cnn-m|250000',
        # 'architecture': 'cnn-l|500000',
        # 'architecture': 'mlp-1000-1|fc',
        # 'architecture': 'mlp-700-3|1250000',
        # 'architecture': 'mlp-10000-1|fc',
        'architecture': 'mlp-700-1|250000',
        
        # 'architecture': 'cnn-l|250000',

        'batch_size': 64,

        'backbone_act': 'gelu',
        'output_act': 'wta',
        'wta_p': 0.15,
        # 'ash_zk': 1.9,
        # 'output_act': 'gelu',
        
        'learning_rate': 4e-4,
        'optimizer': 's-mas',
        'adagrad_lambda': 0.8,
        # 'mas_lambda': 0.01, # comes from experiment config

        'num_runs': 1,
        'trial_runs': 3,

        'base_seed': 555
    }
    if not HYPERPARAMETER_SWEEP:
        if LOG_TO_WANDB:
            wandb.init(project=wandb_project, entity=wandb_user, config = config, name = None)

        exp_hyperparams = get_experiment_hyperparams(config)
        
        # Uncomment the next line to train with specific hyperparameters
        # return train_with_hyperparams({**exp_hyperparams, **config})

        # Uncomment and modify the following for a local hyperparameter sweep
        return train_trial_sweep(config, {
            # 'backbone_act': ['none', 'gelu', 'relu'],
            # 'learning_rate': [1e-5, 2.5e-5, 5e-5],
            'learning_rate': [2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4],
            # 'ash_zk': [1.8, 1.9, 2.0, 2.1, 2.2]
            # 'learning_rate': [1.2e-3],
            'wta_p': [0.1, 0.15],

            # 'mas_lambda': [0.01],
            # 'adagrad_lambda': [0.01, 0.05, 0.1],
            # 'backbone_act': ['ash'],
            # 'adagrad_lambda': [0.6, 0.8, 1.0, 1.2]
        })

    wandb_sweep_config = {
        'method': 'grid',
        'name': 'reruns',
        'metric': {
            'name': 'run_mean' if config['num_runs'] > 1 else 'accuracy',
            'goal': 'maximize'
        },

        'parameters': {
            # 'experiment': Vs(['split_mnist', 'split_mnist_mh', 'split_fashion_mnist', 'split_fashion_mnist_mh']),
            # 'experiment': Vs(['split_mnist', 'split_mnist_mh', 'split_fashion_mnist', 'split_fashion_mnist_mh']),
            'experiment': Vs(['split_mnist', 'split_fashion_mnist']),
            # 'experiment': Vs(['split_mnist']),
            # 'experiment': Vs(['permuted_mnist']),
            # 'experiment': Vs(['split_cifar10_mh']),
            # 'experiment': Vs(['split_cifar10']),
            # 'experiment': Vs(['split_mnist_mh', 'split_fashion_mnist_mh', 'split_cifar10_mh']),
            'batch_size': V(64),

            # Try 4x the cnn dim, per stride 2

            'architecture': Vs([
                # 'mlp-1000-1|fc',
                'mlp-700-1|250000',
                
                # 'mlp-3000-1|5000000',
                # 'mlp-10000-1|fc',

                # 'mlp-1000-3|fc',
                # 'mlp-700-3|1250000',
                
                # 'cnn-s|fc',
                # 'cnn-s|100000',

                # 'cnn-m|fc',
                # 'cnn-m|250000',

                # 'cnn-l|fc',
                # 'cnn-l|500000',

                # e and w are 'extra fc' and 'wide'. Both are pretty bad
                # 'cnn-e|fc',
                # 'cnn-e|100000',

                # 'cnn-w|fc',
                # 'cnn-w|100000',

                #'convnext|m',
                #'convnext|l',
            ]),

            'backbone_act': V('gelu'),
            'output_act': V('wta'),

            'optimizer': Vs(['adagrad', 's-mas']),
            # 'wta_p': V(0.15),
            
            'adagrad_lambda': V(0.8),
            # 'mas_lambda': V(0.01),

            # 'trial_runs': V(1),
            # 'num_runs': V(3),

            'base_seed': V(1337),
        }
    }

    def wandb_sweep_fn():
        with wandb.init() as run:
            train_trial_sweep(wandb.config, {
                # 'learning_rate': [1e-5, 2.5e-5, 5e-5],
                'learning_rate': [2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4],
                # 'learning_rate': [1e-4, 2e-4, 4e-4],
                # 'wta_p': [0.10, 0.15],
                'wta_p': [0.1, 0.15],
                # 'backbone_act': ['gelu', 'ash'],
            }, should_save_hyperparams=True)

            jax.clear_caches()
    
    sweep_id = wandb.sweep(wandb_sweep_config, project=wandb_project, entity=wandb_user)
    wandb.agent(sweep_id, wandb_sweep_fn)

def get_act(str, hyperparams):
    match str:
        case 'wta':
            return lambda: lambda x: wta_p_subtract(x, hyperparams['wta_p'])
        case 'relu':
            return lambda: jax.nn.relu
        case 'gelu':
            return lambda: jax.nn.gelu
        case 'ash':
            return Ash
        case 'hard_ash':
            return lambda: lambda x: hard_ash(x, hyperparams)
        case 'none':
            return lambda: lambda x: x
        case _:
            return Exception()

backbone_init_fn = nn.initializers.he_normal()
pairwise_init_fn = nn.initializers.normal(0.001) # 86.1 %
# pairwise_init_fn = nn.initializers.lecun_normal() # 85.4 %
# pairwise_init_fn = nn.initializers.he_normal() # 84.9 %

def get_backbone(backbone_desc, act, hyperparams):
    info = backbone_desc.split('-')
    match info[0]:
        case 'mlp':
            assert len(info) == 3
            return MLP(int(info[1]), int(info[2]), act, backbone_init_fn)
        case 'cnn':
            assert len(info) == 2
            match info[1]:
                case 's':
                    # dims, strides, kernel sizes
                    return CNN((64,), (4,), (7,), act, backbone_init_fn)
                case 'e':
                    return CNN((64,), (4,), (7,), act, backbone_init_fn, extra_fc=1000)
                case 'w':
                    return CNN((64, 64*4), (4, 2), (7, 5), act, backbone_init_fn)
                case 'm':
                    return CNN((64, 128), (4, 2), (7, 5), act, backbone_init_fn)
                case 'l':
                    return CNN((64, 128, 256), (4, 2, 2), (7, 5, 5), act, backbone_init_fn)
                case _:
                    Exception()
        case 'convnext':
            return ConvNeXt(2000, (2, 2, 2), (1, 2, 1), (64, 128, 256), act, backbone_init_fn)
        case _:
            raise Exception()

def get_output_head(head_desc, hyperparams):
    if head_desc == 'fc':
        return lambda z: nn.Dense(hyperparams['num_classes'], use_bias=False, kernel_init=backbone_init_fn, name=f'output_head')(z)
    weights = int(head_desc)    
    return lambda z: PairwiseLinear(z.shape[-1], hyperparams['num_classes'], weights_init=pairwise_init_fn, size=weights, name=f'output_head')(z)

class Model(nn.Module):
    hyperparams: FrozenDict

    @nn.compact
    def __call__(self, x):
        backbone_desc, head_desc = self.hyperparams['architecture'].split('|')
        backbone_act = get_act(self.hyperparams['backbone_act'], self.hyperparams)
        backbone = get_backbone(backbone_desc, backbone_act, self.hyperparams)

        output_act = get_act(self.hyperparams['output_act'], self.hyperparams)
        head = get_output_head(head_desc, self.hyperparams)

        z = backbone(x)
        z = output_act()(jnp.ravel(z))
        return head(z)

def batched_permute(xs, permutation):
    s = xs.shape
    xs = jnp.reshape(xs, (s[0], -1))
    xs = xs[:, permutation]
    xs = jnp.reshape(xs, s)
    return xs

def task_mask(ys, hyperparams):
    classes_per_task = hyperparams['num_classes'] // hyperparams['num_tasks']
    r = jnp.arange(0, classes_per_task)
    ys = (ys // classes_per_task) * classes_per_task
    ys = jnp.expand_dims(ys, -1)
    ys = ys + r

    # Reshape ys to be a 1D array for indexing
    b = ys.shape[0]
    ys = ys.reshape(-1)
    batch_indices = jnp.repeat(jnp.arange(b), classes_per_task)
    
    mask = jnp.zeros((b, hyperparams['num_classes'],))
    mask = mask.at[batch_indices, ys].set(1.)
    return mask

@partial(jax.jit, static_argnums=(0))
def step_classifier(statics, weights, state, omega, xs, ys, permutation=None):
    model, hyperparams = statics

    @jax.value_and_grad
    def classifier_loss(weights):
        @partial(jax.vmap, in_axes=(0, None))
        def go(x, weights):
            return model.apply({'params': weights, **state}, x)

        logits = go(xs, weights)
        if hyperparams['labels_trick_train'] or hyperparams['labels_trick_train_test']:
            logits *= task_mask(ys, hyperparams)
        
        class_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=ys))

        return class_loss

    if permutation is not None:
        xs = batched_permute(xs, permutation)

    loss, gs = classifier_loss(weights)

    @jax.grad
    def mas_squared_output_grad(weights):
        @jax.vmap
        def go(x):
            return model.apply({'params': weights, **state}, x)

        logits = go(xs)
        return jnp.mean(logits ** 2.0)

    lr = hyperparams['learning_rate'] * hyperparams['task_learning_rate']
    eps = 1e-6
    match hyperparams['optimizer']:
        case 'sgd':
            updates = jtu.tree_map(lambda g: -lr * g, gs)
        case 'adagrad':
            omega = jtu.tree_map(lambda o, g: o + hyperparams['adagrad_lambda'] * g ** 2.0, omega, gs)
            updates = jtu.tree_map(lambda o, g: -lr * g * jax.lax.rsqrt(o + eps), omega, gs)
        case 's-mas':
            importance = mas_squared_output_grad(weights)
            omega = jtu.tree_map(lambda o, i: o + hyperparams['mas_lambda'] * jnp.abs(i), omega, importance)
            updates = jtu.tree_map(lambda o, g: -lr * g * jax.lax.rsqrt(o + eps), omega, gs)
        case _:
            raise Exception()
    weights = optax.apply_updates(weights, updates)

    return loss, weights, omega

@partial(jax.jit, static_argnums=(0,))
def evaluate_step(statics, xs, ys, weights, state, permutation = None):
    model, hyperparams = statics

    @jax.vmap
    def go(x):
        return model.apply({'params': weights, **state}, x)

    if permutation is not None:
        xs = batched_permute(xs, permutation)

    logits = go(xs)
    if hyperparams['labels_trick_train_test']:
        logits *= task_mask(ys, hyperparams)

    predicted_classes = jnp.argmax(logits, axis=1)

    correct_predictions_mask = predicted_classes == ys

    correct_counts = jnp.zeros(hyperparams['num_classes'])
    for c in range(hyperparams['num_classes']):
        correct_counts = correct_counts.at[c].set(jnp.sum(correct_predictions_mask & (predicted_classes == c)))

    total_counts = jnp.bincount(ys, length=hyperparams['num_classes'])

    return correct_counts, total_counts

def evaluation(statics, weights, state, permutation = None):
    model, hyperparams = statics
    
    correct_counts = jnp.zeros(hyperparams['num_classes'])
    total_counts = jnp.zeros(hyperparams['num_classes'])

    for xs, ys in test_loader(hyperparams['dataset'], hyperparams['batch_size']):
        correct, total = evaluate_step(statics, xs, ys, weights, state, permutation)
        correct_counts += correct
        total_counts += total
    
    accuracy = jnp.sum(correct_counts) / jnp.sum(total_counts)
    task_correct = jnp.sum(jnp.reshape(correct_counts, (5, 2)), axis=1)
    task_total = jnp.sum(jnp.reshape(total_counts, (5, 2)), axis=1)
    return accuracy.item(), task_correct / task_total

def save_hyperparams(path, hyperparams):
    path = f'repro_configs/{path}.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(hyperparams)
    with open(path, 'w') as file:
        json.dump(hyperparams, file)

def train_trial_sweep(base_config, trial_definition, should_save_hyperparams = False):
    best_result = 0.
    best_config = {}

    configs_to_try = []
    def go(c, ks):
        if len(ks) == 0:
            configs_to_try.append({**c})
        else:
            k = ks[0]
            for v in trial_definition[k]:
                c[k] = v
                go(c, ks[1:])

    keys = list(trial_definition.keys())
    go({}, keys)
    
    print(f'Starting trials with {len(configs_to_try)} possible configurations.')

    experiment_config = get_experiment_hyperparams(base_config)
    base_config = {**experiment_config, **base_config}
    for c in configs_to_try:
        trial_config = {**base_config, **c,
            'num_runs': base_config['trial_runs'], 'revisits': 1, 'base_seed': 1}
        r = train_with_hyperparams(trial_config, do_logging=False)
        if r is None:
            print(f'r is none, {c}')
        elif r > best_result:
            best_result = r
            best_config = c
            print(f'new best trial result: {r}. {c}')
        else:
            print(f'no improvement: {r}, {c}')

    print('start final run')
    log(best_config)
    final_hyperparams = {**base_config, **best_config}
    
    if should_save_hyperparams:
        save_path = f"{base_config['architecture']}_{base_config['optimizer']}/{base_config['experiment']}"
        save_hyperparams(save_path, final_hyperparams)
    
    return train_with_hyperparams(final_hyperparams, do_logging=True)

def train_with_hyperparams(hyperparams, do_logging=True):
    if isinstance(hyperparams, dict):
        hyperparams = FrozenDict(hyperparams)

    if (not hyperparams['split_experiment']) and (hyperparams['labels_trick_train'] or hyperparams['labels_trick_train_test']):
        return 0.
    if ('cnn' in hyperparams['architecture'] and hyperparams['permuted_experiment']):
        return 0.

    run_accuracies = []

    for i_run in range(hyperparams['num_runs']):
        key = random.PRNGKey(hyperparams['base_seed'] + i_run * 113)

        model = Model(hyperparams)
        
        dummy_input = jnp.zeros(input_size(hyperparams['dataset']))
        key, model_key = random.split(key)
        variables = model.init(model_key, dummy_input)
        state, weights = flax.core.pop(variables, 'params')

        if do_logging and i_run == 0:
            log({'model_weights': jtu.tree_reduce(lambda x, y: x + np.size(y), weights, 0)})
        match hyperparams['optimizer']:
            case 'sgd':
                omega = None
            case _:
                # omega = jtu.tree_map(lambda x: jnp.full_like(x, 1e-6), weights)
                omega = jtu.tree_map(lambda x: jnp.zeros_like(x), weights)

        statics = (model, hyperparams)

        if hyperparams['split_experiment']:
            tasks = random.permutation(key, hyperparams['num_tasks'])
            for visit in range(hyperparams['revisits']):
                for i_split, split in enumerate(tqdm(tasks)):
                    for epoch in range(hyperparams['epochs']):
                        for xs, ys in train_loader(hyperparams['dataset'], hyperparams['batch_size'], split=split):
                            loss, weights, omega = step_classifier(statics, weights, state, omega, xs, ys)

                            if np.isnan(loss):
                                raise Exception(f'loss is nan')
            accuracy, per_task_accuracy = evaluation(statics, weights, state)
            if do_logging and i_run == 0:
                log_dict = {'accuracy': accuracy, 'visit': visit, 'split': i_split, 'epoch': epoch}
                for i, task_accuracy in enumerate(per_task_accuracy):
                    log_dict[f'acc_{i}'] = task_accuracy.item()
                log(log_dict)
            # if split > 0 and accuracy < 0.20:
            #     print('early exit')
            #     return accuracy # early exit if results are bad

        elif hyperparams['permuted_experiment']:
            p_keys = random.split(key, hyperparams['num_tasks'])
            permutations = [random.permutation(k, np.prod(dummy_input.shape)) for k in p_keys]

            for task, permutation in enumerate(tqdm(permutations)):
                for e in range(hyperparams['epochs']):
                    for xs, ys in train_loader(hyperparams['dataset'], hyperparams['batch_size']):
                        loss, weights, omega = step_classifier(statics, weights, state, omega, xs, ys, permutation)

                        if np.isnan(loss):
                            raise Exception(f'loss is nan')
                        
            test_accuracies = []
            for permutation in permutations:
                acc, _ = evaluation(statics, weights, state, permutation)
                test_accuracies.append(acc)

            accuracy = np.mean(test_accuracies)
            if do_logging and i_run == 0:
                log({
                    'Last task accuracy': test_accuracies[-1],
                    'First task accuracy': test_accuracies[0],
                    'accuracy': accuracy
                })
                
        else: # iid
            for e in tqdm(range(hyperparams['epochs'])):
                for xs, ys in train_loader(hyperparams['dataset'], hyperparams['batch_size']):
                    loss, weights, omega = step_classifier(statics, weights, state, omega, xs, ys)

                    if np.isnan(loss):
                        raise Exception(f'loss is nan')
                    
                accuracy, _ = evaluation(statics, weights, state)

                if do_logging:
                    log({'accuracy': accuracy})
        run_accuracies.append(accuracy)

        if do_logging and i_run > 0:
            run_mean = np.mean(run_accuracies)
            rounds = len(run_accuracies)
            
            # https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html
            confidence = 0.95 
            t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=rounds - 1)

            sd = np.std(run_accuracies, ddof=1)
            se = sd / np.sqrt(rounds)

            ci_length = t_value * se

            log({
                'run_mean': run_mean,
                'run_se': se,
                'run_95_confidence_interval_length': ci_length,
                'run_count': rounds,
            })

    return np.mean(run_accuracies)


def log(x, step=None):
    print(x)
    if wandb.run is not None:
        wandb.log(x, step=step)
def V(x):
    return {'value': x}
def Vs(xs):
    return {'values': xs}

if __name__ == '__main__':
    run_training()