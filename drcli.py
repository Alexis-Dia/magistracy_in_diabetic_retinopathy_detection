import os.path

import click
import numpy as np
from tfgenerator import (preprocess_gaussian,
                         preprocess_no_gaussian,
                         TFGenerator)
from tftrainer import (TFTrainer)
import tensorflow as tf


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    if not debug:
        # disabling tensorflow warning and info messages
        import logging
        logger = tf.get_logger()
        logger.setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def ps_callback_stub(*args, **kwargs):
    for arg in args:
        print(f'arg: {arg}')
    for kwarg in kwargs:
        print(f'kwarg: {kwarg}')


PROCESS_IMAGE_CALLBACKS = {
    'gaussian': preprocess_gaussian,
    'no_gaussian': preprocess_no_gaussian,
    'greenen': ps_callback_stub
}


@cli.command()
@click.option('-l', '--file-labels', type=click.STRING, default='trainLabels.csv',
              help='train labels file, format csv: id_<side>,level')
@click.option('-t', '--train-dir', type=click.STRING, default='train',
              help='train dir, directory with source images, used for training, dir must exist')
@click.option('-d', '--dest-dir', type=click.STRING, default='dest',
              help='destination dir for output tfrec files, dir must exist')
@click.option('-s', '--sample-dir', type=click.STRING, default='sample',
              help='sample dir, used for printing preprocessed images, used temporary for test purposes only')
@click.option('-c', '--batch-size', type=click.INT, default=2000,
              help='tfrec row size, number of tensors per single tfrec file')
@click.option('-p', '--image-callback', type=click.STRING, default='no_gaussian',
              help='preprocess callback, choose from "gaussian", "no gaussian" so far')
@click.option('-k', '--kind', type=click.STRING, default='train',
              help='tfrec, used for testing or training, use either "train" or "test"')
def generate(file_labels,
             train_dir,
             dest_dir,
             sample_dir,
             batch_size,
             image_callback,
             kind):
    g = TFGenerator(
        file_labels=file_labels,
        source_dir=train_dir,
        destination_dir=dest_dir,
        sample_dir=sample_dir,
        batch_size=batch_size,
        preprocess_image_callback=PROCESS_IMAGE_CALLBACKS[image_callback]
    )
    g.generate(kind)


@cli.command()
@click.option('-r', '--run-on', type=click.STRING, default='TPU',
              help='on Windows, use GPU, on  GKE use TPU')
@click.option('-s', '--source-dir', type=click.STRING, default='.',
              help='source dir: directory with train*.tfrec files, must exist')
@click.option('-d', '--dest-dir', type=click.STRING, default='.',
              help='directory for storing weight coefficients, logs')
@click.option('-i', '--img-res', type=click.INT, default=512,
              help='tensor resolution')
@click.option('-f', '--folds-count', type=click.INT, default=5,
              help='number of folds for KFold, lower if going to train tiny number of tfrecs')
@click.option('-e', '--epoch-count', type=click.INT, default=10,
              help='epoch count')
@click.option('-b', '--batch-size', type=click.INT, default=4,
              help='train batch size, 4 is maximum for GPU with 11GB of VRAM')
@click.option('-n', '--efn-index', type=click.INT, default=4,
              help='EFN index: 0 for EfficientNetB0, 4 for EfficientNetB4, etc')
@click.option('-v', '--verbose', type=click.BOOL, default=True)
@click.option('-p', '--should-plot', type=click.BOOL, default=True,
              help='If true - plot learning intermediate results')
def train(run_on,
          source_dir,
          dest_dir,
          img_res,
          folds_count,
          epoch_count,
          batch_size,
          efn_index,
          verbose,
          should_plot):
    t = TFTrainer(
        device=run_on,
        source_dir=source_dir,
        destination_dir=dest_dir,
        img_size=img_res,
        folds_count=folds_count,
        epoch_count=epoch_count,
        batch_size=batch_size,
        eff_net=efn_index,
        show_files=True,
        verbose=verbose
    )
    t.train()
    if should_plot:
        t.plot()


@cli.command()
@click.argument('filename')
def predict(filename):
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f'file {filename} must exist')
        print(f'File {filename}')
        img = TFGenerator.get_img(filename)
        tensor = TFGenerator.get_tensor(img)
        model = TFTrainer.load_model(fold_index=2)
        categorical_vector = model.predict(tensor, steps=1)
        level = np.argmax(categorical_vector, axis=-1)
        prob = float(categorical_vector[0][level][0])
        print(f'Severity {level[0]} with probability {prob * 100:3.2f}%')
        print(f'Probability distribution:')
        [print(f'{e}: {x * 100:3.2f}%') for e, x in enumerate(categorical_vector[0])]
        pass
    except Exception as e:
        print(f'error: {e}')


if __name__ == '__main__':
    cli()
