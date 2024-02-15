import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], device_type='GPU')

import keras_cv

import os
import shutil

def get_experiment_hyperparams(config):
    experiment_config = {}
    experiment_config['split_experiment'] = True
    experiment_config['permuted_experiment'] = False
    experiment_config['num_tasks'] = 5
    experiment_config['num_classes'] = 10
    experiment_config['revisits'] = 1
    experiment_config['epochs'] = 1
    experiment_config['task_learning_rate'] = 1.0
    
    experiment_config['num_runs'] = 30
    experiment_config['trial_runs'] = 3

    experiment_config['mas_lambda'] = 0.01

    experiment_config['labels_trick_train'] = False
    experiment_config['labels_trick_train_test'] = False

    match config['experiment']:
        case 'split_mnist':
            experiment_config['dataset'] = 'mnist'
        case 'split_mnist_mh':
            experiment_config['dataset'] = 'mnist'
            experiment_config['labels_trick_train_test'] = True
            experiment_config['task_learning_rate'] = 10.
        case 'permuted_mnist':
            experiment_config['dataset'] = 'mnist'
            experiment_config['num_tasks'] = 10
            experiment_config['task_learning_rate'] = 10.
            experiment_config['split_experiment'] = False
            experiment_config['permuted_experiment'] = True

            experiment_config['num_runs'] = 10
            experiment_config['trial_runs'] = 1
        case 'iid_mnist':
            experiment_config['dataset'] = 'mnist'
            experiment_config['task_learning_rate'] = 10.
            experiment_config['split_experiment'] = False
            experiment_config['num_tasks'] = 1
        case 'split_fashion_mnist':
            experiment_config['dataset'] = 'fashion_mnist'
        case 'split_fashion_mnist_mh':
            experiment_config['dataset'] = 'fashion_mnist'
            experiment_config['labels_trick_train_test'] = True
            experiment_config['task_learning_rate'] = 10.
        case 'iid_cifar10':
            experiment_config['dataset'] = 'cifar10'
            experiment_config['task_learning_rate'] = 10.
            experiment_config['split_experiment'] = False
            experiment_config['epochs'] = 20
        case 'split_cifar10':
            experiment_config['dataset'] = 'cifar10'
            experiment_config['revisits'] = 1
            experiment_config['epochs'] = 1
            experiment_config['mas_lambda'] = 0.001
            experiment_config['trial_runs'] = 1
            experiment_config['num_runs'] = 3

        case 'split_cifar10_mh':
            experiment_config['dataset'] = 'cifar10'
            experiment_config['epochs'] = 3
            experiment_config['mas_lambda'] = 0.001
            experiment_config['labels_trick_train_test'] = True
            experiment_config['task_learning_rate'] = 10.

            experiment_config['trial_runs'] = 1
            experiment_config['num_runs'] = 3

        case 'split_cifar100':
            experiment_config['dataset'] = 'cifar100'
            experiment_config['num_tasks'] = 20
            experiment_config['num_classes'] = 100
            experiment_config['revisits'] = 5
            experiment_config['epochs'] = 30
        case _:
            raise Exception()
    return experiment_config


def input_size(dataset):
    match dataset:
        case 'mnist':
            return (28, 28, 1)
        case 'fashion_mnist':
            return (28, 28, 1)
        case 'cifar10':
            return (32, 32, 3)
        case 'cifar100':
            return (32, 32, 3)
        case _:
            raise Exception()

# mnist
SPLIT_MNIST_PATH = 'data_split_mnist'
SPLIT_FASHION_MNIST_PATH = 'data_split_fashion_mnist'


def preprocess_mnist(image, label):    
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - tf.constant(0.1307)) / tf.constant(0.3081)
    return image, label

def mnist_loader(dataset, batch_size):
    dataset = dataset.map(preprocess_mnist)
    # dataset_length = dataset.cardinality().numpy()
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(64).as_numpy_iterator()
    return dataset#, dataset_length // batch_size

def create_split_mnist_data(dataset_name, path):
    full_dataset = tfds.load(dataset_name, split='train', as_supervised=True, shuffle_files=True)
    for i in range(5):
        def task_filter(x, y):
            return (y // 2) == i
        dataset = full_dataset.filter(task_filter).shuffle(1000)
        dataset.save(f'{path}/{i}')

def mnist_train_loader(batch_size, split = None):
    if split is not None:
        if not os.path.exists(SPLIT_MNIST_PATH):
            create_split_mnist_data('mnist', SPLIT_MNIST_PATH)

        return mnist_loader(tf.data.Dataset.load(f'{SPLIT_MNIST_PATH}/{split}'), batch_size)
    return mnist_loader(tfds.load('mnist', split='train', as_supervised=True), batch_size)

def preprocess_fashion_mnist(image, label):    
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - tf.constant(0.2859)) / tf.constant(0.3530)
    return image, label

def fashion_mnist_loader(dataset, batch_size):
    dataset = dataset.map(preprocess_fashion_mnist)
    # dataset_length = dataset.cardinality().numpy()
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(64).as_numpy_iterator()
    return dataset#, dataset_length // batch_size

def fashion_mnist_train_loader(batch_size, split = None):
    if split is not None:
        if not os.path.exists(SPLIT_FASHION_MNIST_PATH):
            create_split_mnist_data('fashion_mnist', SPLIT_FASHION_MNIST_PATH)

        return fashion_mnist_loader(tf.data.Dataset.load(f'{SPLIT_FASHION_MNIST_PATH}/{split}'), batch_size)
    return fashion_mnist_loader(tfds.load('fashion_mnist', split='train', as_supervised=True), batch_size)

# cifar
SPLIT_CIFAR10_PATH = 'data_split_cifar10'

augmenter = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomCropAndResize(
            target_size=(32, 32),
            crop_area_factor=(0.8, 1),
            aspect_ratio_factor=(3 / 4, 4 / 3),
        ),
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255), magnitude=0.3, augmentations_per_image=2),
    ],
)

def preprocess_cifar(images, labels, augment = False):
    images = tf.cast(images, tf.float32)
    if augment:
        images = augmenter(images)

    images = ((images / 255.0) - tf.constant([0.5071, 0.4866, 0.4409])) / tf.constant([0.2673, 0.2564, 0.2762])

    return images, labels

def cifar_loader(dataset, batch_size, augment = False):
    dataset_length = dataset.cardinality().numpy()

    dataset = dataset.map(lambda x, y: preprocess_cifar(x, y, augment = augment), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # dataset = dataset.take(5)
    dataset = dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
    return dataset#, dataset_length // batch_size

def create_split_cifar10_data():
    full_dataset = tfds.load('cifar10', split='train', as_supervised=True, shuffle_files=True)
    for i in range(5):
        def task_filter(x, y):
            return (y // 2) == i
        dataset = full_dataset.filter(task_filter).shuffle(1000)
        dataset.save(f'{SPLIT_CIFAR10_PATH}/{i}')

# cifar100

# NUM_TASKS = 6
# first_task = 50
# other_tasks = 10
# path = f'cifar100_split_T{NUM_TASKS}'
# shutil.rmtree(path, ignore_errors=True)
# if not os.path.exists(path):
#     full_dataset = tfds.load('cifar100', split='train', as_supervised=True, shuffle_files=True)

#     for i in range(NUM_TASKS):
#         def task_filter(x, y):
#             f = y // first_task
#             t = f + f * (y - first_task) // other_tasks
#             return t == i
#         dataset = full_dataset.filter(task_filter).shuffle(1000)
#         dataset.save(f'{path}/{i}')


def cifar10_train_loader(batch_size, split = None, augment = True):
    if split is not None:
        if not os.path.exists(SPLIT_CIFAR10_PATH):
            create_split_cifar10_data()
        return cifar_loader(tf.data.Dataset.load(f'{SPLIT_CIFAR10_PATH}/{split}'), batch_size, augment = augment)
    return cifar_loader(tfds.load('cifar10', split='train', as_supervised=True, shuffle_files=True), batch_size, augment = augment)

def train_loader(dataset, batch_size, split = None):
    match dataset:
        case 'mnist':
            return mnist_train_loader(batch_size, split=split)
        case 'fashion_mnist':
            return fashion_mnist_train_loader(batch_size, split=split)
        case 'cifar10':
            return cifar10_train_loader(batch_size, split=split, augment=False)
        case _:
            raise Exception()

def test_loader(dataset, batch_size):
    match dataset:
        case 'mnist':
            return mnist_loader(tfds.load('mnist', split='test', as_supervised=True), batch_size)
        case 'fashion_mnist':
            return fashion_mnist_loader(tfds.load('fashion_mnist', split='test', as_supervised=True), batch_size)
        case 'cifar10':
            return cifar_loader(tfds.load('cifar10', split='test', as_supervised=True), batch_size)
        case _:
            raise Exception()