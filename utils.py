import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import efficientnet.tfkeras as efn


def plot_loss(fold, loss, val_loss):
    plt.figure(figsize=(14, 6))

    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.title('Loss fold n. ' + str(fold))
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.show()


def plot_acc(fold, acc, val_acc):
    plt.figure(figsize=(14, 6))

    plt.plot(acc, label='Training accuracy')
    plt.plot(val_acc, label='Validation accuracy')
    plt.title('Acc. fold n. ' + str(fold))
    plt.legend(loc='lower right')
    plt.ylabel('Acc')
    plt.xlabel('epoch')
    plt.grid()
    plt.show()


def plot_fold(fold, filename_pattern='training_retina-%i.csv'):
    data = pd.read_csv(filename_pattern % (fold - 1))

    epochs = data['epoch'].values
    loss = data['loss'].values
    acc = data['accuracy'].values
    val_acc = data['val_accuracy'].values
    val_loss = data['val_loss'].values

    plot_loss(fold, loss, val_loss)
    plot_acc(fold, acc, val_acc)


def infer_strategy(device, verbose=False):
    strategy = None
    if device == "GPU":
        n_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
        if verbose:
            print("Num GPUs Available: ", n_gpu)

        if n_gpu > 1:
            if verbose:
                print("Using strategy for multiple GPU")
            strategy = tf.distribute.MirroredStrategy()
        else:
            if verbose:
                print('Standard strategy for GPU...')
            strategy = tf.distribute.get_strategy()

    auto = tf.data.experimental.AUTOTUNE
    replicas = strategy.num_replicas_in_sync

    if verbose:
        print(f'Autotune: {"True" if auto != -1 else "False"}')
        print(f'Replicas count: {replicas}')

    return strategy, auto, replicas


def efns():
    return [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
            efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]


if __name__ == '__main__':
    for i in range(1, 4):
        plot_fold(i, 'original/training_retina-%i.csv')
