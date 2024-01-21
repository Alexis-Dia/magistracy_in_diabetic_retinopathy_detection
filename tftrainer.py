import logging
import os.path as osp
import re
import time

import efficientnet.tfkeras as efn
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold


class TFTrainer:
    EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
            efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]

    def __init__(self,
                 device: str,
                 source_dir: str,
                 destination_dir: str,
                 img_size: int = 512,
                 folds_count: int = 5,
                 epoch_count: int = 10,
                 batch_size: int = 10,
                 eff_net: int = 4,
                 show_files: bool = True,
                 verbose: bool = True
                 ):
        self.device = device
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        self.img_size = img_size
        self.folds_count: int = folds_count
        self.epoch_count: int = epoch_count
        self.batch_size: int = batch_size
        self.eff_net: int = eff_net
        self.show_files = show_files
        self.verbose = verbose

        self._tpu = None
        self._strategy = None
        self._histories = []
        self._num_classes = 5
        self._data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.4),
        ])
        self._wghts = 1 / self.folds_count
        self._img_size_list = [self.img_size, self.img_size]

        self._tune_logger()
        self._init_device_and_strategy()

        self._auto = tf.data.experimental.AUTOTUNE
        #self._replicas = self._strategy.num_replicas_in_sync
        self._replicas = 1
        self._model_test = self._build_model(ef=self.eff_net)
        self._skf = KFold(n_splits=self.folds_count, shuffle=True, random_state=42)

        #assert self._strategy
        assert osp.exists(self.source_dir)
        assert osp.exists(self.destination_dir)
        assert img_size == 512  # only 512 is currently supported
        assert self.device in ['TPU', 'GPU', 'CPU']
        assert self._model_test is not None

        self._num_total_train_files = len(tf.io.gfile.glob(self.source_dir + '\\train*.tfrec'))
        #assert self._num_total_train_files >= self.folds_count

        if self.verbose:
            #print(f'replicas: {self._replicas}')
            print(f'source dir: {self.source_dir}')
            self._model_test.summary()
            print(f'num total train files {self._num_total_train_files}')

    @staticmethod
    def _tune_logger():
        logger = tf.get_logger()
        logger.setLevel(logging.ERROR)

    def _init_device_and_strategy(self):
        if self.device == "TPU":
            print("connecting to TPU...")
            try:
                self._tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                print('Running on TPU ', self._tpu.master())
            except ValueError:
                print("Could not connect to TPU")
                self._tpu = None

            if self._tpu:
                try:
                    print("initializing  TPU ...")
                    tf.config.experimental_connect_to_cluster(self._tpu)
                    tf.tpu.experimental.initialize_tpu_system(self._tpu)
                    self._strategy = tf.distribute.experimental.TPUStrategy(self._tpu)
                    print("TPU initialized")
                except Exception as e:
                    print(f"failed to initialize TPU, error: {e}")
            else:
                self.device = "GPU"

        if self.device == "GPU":
            n_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
            if self.verbose:
                print("Num GPUs Available: ", n_gpu)

            if n_gpu > 1:
                if self.verbose:
                    print("Using strategy for multiple GPU")
                self._strategy = tf.distribute.MirroredStrategy()
            else:
                if self.verbose:
                    print('Standard strategy for GPU...')
                self._strategy = tf.distribute.get_strategy()

    def _read_labeled_tfrecord(self, example):
        labeled_tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            'patient_id': tf.io.FixedLenFeature([], tf.int64),
            'side': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example, labeled_tfrec_format)

        image = self._decode_image(example['image'])
        patient_id = example['patient_id']
        side = example['side']
        label = example['label']

        return image, label

    def _decode_image(self, image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

        image = tf.reshape(image, [*self._img_size_list, 3])  # explicit size needed for TPU
        return image

    def _load_dataset(self, filenames, labeled=True, ordered=False):
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False

        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=self._auto)
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.map(self._read_labeled_tfrecord)
        return dataset

    def _get_training_dataset(self, filenames):
        dataset = self._load_dataset(filenames, labeled=True)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self._auto)
        return dataset

    @staticmethod
    def _count_data_items(filenames):
        n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
        return np.sum(n)

    @classmethod
    def load_model(cls, img_size=512, eff_net=4, fold_index=1):
        loader = cls(
            device='CPU',
            #source_dir=r'H:\Bsuir\Diplom\от Ильи Рябоконя\cshnick_crew-drcli-9d43a42af299\tfrec_balanced',
            source_dir=r'H:\Bsuir\Diplom\magistracy_in_diabetic_retinopathy_detection\results',
            #destination_dir=r'H:\Bsuir\Diplom\от Ильи Рябоконя\cshnick_crew-drcli-9d43a42af299\dist',
            destination_dir=r'H:\Bsuir\Diplom\dest',
            img_size=img_size,
            folds_count=5,
            epoch_count=25,
            batch_size=4,
            eff_net=eff_net,
            verbose=False)
        model = loader._model_test
        model.load_weights(f'fold-{fold_index}.h5')

        return model

    def _build_model(self, dim=256, ef=0):
        inp = tf.keras.layers.Input(shape=(*self._img_size_list, 3))
        x = self._data_augmentation(inp)
        base = self.EFNS[ef](input_shape=(*self._img_size_list, 3), weights='imagenet', include_top=False)
        x = base(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self._num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inp, outputs=x)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def _get_lr_callback(self, batch_size=8):
        lr_start = 0.000005
        lr_max = 0.000020 * self._replicas * batch_size / 16
        lr_min = 0.000001
        lr_ramp_ep = 5
        lr_sus_ep = 2
        lr_decay = 0.8

        def lrfn(epoch):
            if epoch < lr_ramp_ep:
                lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            elif epoch < lr_ramp_ep + lr_sus_ep:
                lr = lr_max
            else:
                lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            return lr

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
        return lr_callback

    @staticmethod
    def _plot_loss(tf_hist, fold):
        plt.figure(figsize=(14, 6))

        plt.plot(tf_hist.history['loss'], label='Training loss')
        plt.plot(tf_hist.history['val_loss'], label='Validation loss')
        plt.title('Loss fold n. ' + str(fold + 1))
        plt.legend(loc='upper right')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.grid()
        plt.show()

    @staticmethod
    def _plot_accuracy(tf_hist, fold):
        plt.figure(figsize=(14, 6))

        plt.plot(tf_hist.history['accuracy'], label='Training accuracy')
        plt.plot(tf_hist.history['val_accuracy'], label='Validation accuracy')
        plt.title('Accuracy fold n. ' + str(fold + 1))
        plt.legend(loc='lower right')
        plt.ylabel('Acc')
        plt.xlabel('epoch')
        plt.grid()
        plt.show()

    def _train_fold(self, fold: int, idxT, idxV):
        tStart = time.time()

        # display fold info
        print('#' * 60)
        print('#### FOLD', fold + 1)

        # print('#### Image Size %i, EfficientNet B%i, batch_size %i' %
        #       (self.img_size, self.eff_net, self.batch_size * self._replicas))
        print('#### Epochs: %i' % self.epoch_count)

        # create trade and validation subsets
        files_train = tf.io.gfile.glob([self.source_dir + '/train%.2i*.tfrec' % x for x in idxT])

        np.random.shuffle(files_train)
        print('#' * 60)

        files_valid = tf.io.gfile.glob([self.source_dir + '/train%.2i*.tfrec' % x for x in idxV])

        if self.show_files:
            print('Number of training images', self._count_data_items(files_train))
            print('Number of validation images', self._count_data_items(files_valid))

        if self.device == 'TPU':
            # to avoid OOM
            tf.tpu.experimental.initialize_tpu_system(self._tpu)

        K.clear_session()
        model = self._build_model(dim=self.img_size, ef=self.eff_net)
        # with self._strategy.scope():
        #     model = self._build_model(dim=self.img_size, ef=self.eff_net)

        # callback to save best model for each fold
        sv = tf.keras.callbacks.ModelCheckpoint(
            'fold-%i.h5' % fold, monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True, mode='min', save_freq='epoch')

        csv_logger = tf.keras.callbacks.CSVLogger('training_retina-%i.csv' % fold)

        # TRAIN
        history = model.fit(
            self._get_training_dataset(files_train),
            epochs=self.epoch_count,
            callbacks=[sv, self._get_lr_callback(self.batch_size), csv_logger],
            steps_per_epoch=self._count_data_items(files_train) / self.batch_size // self._replicas,
            validation_data=self._get_training_dataset(files_valid),
            validation_steps=self._count_data_items(files_valid) / self.batch_size // self._replicas,
            verbose=self.verbose)

        # save all histories
        self._histories.append(history)

        tElapsed = round(time.time() - tStart, 1)

        print(' ')
        print('Time (sec) elapsed for fold: ', tElapsed)
        print('...')
        print('...')

    def train(self):
        for fold, (idxT, idxV) in enumerate(self._skf.split(np.arange(self._num_total_train_files))):
            self._train_fold(fold, idxT, idxV)

    def plot(self):
        for fold in range(self.folds_count):
            self._plot_loss(self._histories[fold], fold)

        for fold in range(self.folds_count):
            self._plot_accuracy(self._histories[fold], fold)

        files_test = tf.io.gfile.glob(self.source_dir + '/train*.tfrec')
        num_total_test_files = len(tf.io.gfile.glob(self.source_dir + '/train*.tfrec'))
        wi = [1 / self.folds_count] * self.folds_count
        avg_a = 0

        def calc_fold_accuracy():
            self._model_test.load_weights('fold-%i.h5' % fold)

            test_loss, test_acc = self._model_test.evaluate(
                self._get_training_dataset(files_test),
                verbose=0,
                batch_size=4 * self.batch_size,
                steps=num_total_test_files / 4 * self.batch_size // self._replicas)

            print('Train accuracy fold n.', fold + 1, ': ', round(test_acc, 4))
            return test_acc

        for fold in range(self.folds_count):
            avg_a += calc_fold_accuracy() * wi[fold]

        print('Average accuracy: ', round(avg_a, 4))


if __name__ == '__main__':
    t = TFTrainer(
        device='CPU',
        source_dir=r'H:\Bsuir\Diplom\от Ильи Рябоконя\cshnick_crew-drcli-9d43a42af299\tfrec_no_gaussian',
        destination_dir=r'H:\Bsuir\Diplom\от Ильи Рябоконя\cshnick_crew-drcli-9d43a42af299\dist',
        img_size=512,
        folds_count=5,
        epoch_count=25,
        batch_size=4,
        eff_net=4)

    t.train()
    t.plot()
