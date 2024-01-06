import logging
import os
import re
import time
from typing import List

import keras.backend as K
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import utils
from utils import infer_strategy

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def _side_map(side):
    if side == 0:
        s_side = 'left'
    else:
        s_side = 'right'
    return s_side


class BatchData:
    def __init__(self, image, patient_id, side, level):
        self.image = image
        self.patient_id = patient_id
        self.side = side
        self.level = level

    def __call__(self):
        return self.image, self.patient_id, self.side, self.level


class BatchValidator:
    def __call__(self, batch_num, batch_data: BatchData, predict):
        pass

    def fold_start(self, fold):
        pass

    def fold_finish(self, fold):
        pass

    def validation_start(self, folds_count, n_test_images, num_classes):
        pass

    def validation_finish(self):
        pass


class CommonBatchValidator(BatchValidator):
    def __init__(self):
        self._folds_count = None
        self._n_test_images = None
        self._num_classes = None
        self.img_names = []

        self.class_prob_predictions = None
        self.class_y_predictions = None

        self.class_predictions_fold = None
        self.class_y_fold = None
        self.class_p_fold = None

        self._v_side_map = np.vectorize(_side_map)

    def __call__(self, batch_num, batch_data: BatchData, predict):
        image, patient_id, side, level = batch_data()
        # I have to construct the names of the files
        s_patient_id = np.char.mod('%d', patient_id.numpy())
        name = np.char.add(s_patient_id, '_')

        # side I have to transform 0 into left, 1 into right
        s_side = self._v_side_map(side.numpy())
        name = np.char.add(name, s_side)
        self.img_names = np.append(self.img_names, name)

        start = batch_num * len(predict)
        end = ((batch_num + 1) * len(predict))
        self.class_predictions_fold[start:end] = predict
        self.class_y_fold[start:end] = level.numpy()

    def fold_start(self, fold):
        self.img_names = np.array([])
        self.class_predictions_fold = np.zeros(shape=(self._n_test_images, self._num_classes))
        self.class_y_fold = np.zeros(shape=self._n_test_images, dtype='int64')
        self.class_p_fold = np.zeros(shape=self._n_test_images, dtype='int64')

    def fold_finish(self, fold):
        self.class_prob_predictions[fold] = self.class_predictions_fold
        self.class_y_predictions[fold] = self.class_y_fold

    def validation_start(self, folds_count, n_test_images, num_classes):
        self._folds_count = folds_count
        self._n_test_images = n_test_images
        self._num_classes = num_classes
        self.class_prob_predictions = np.zeros(shape=(folds_count, n_test_images, num_classes))
        self.class_y_predictions = np.zeros(shape=(folds_count, n_test_images), dtype='int64')

    def validation_finish(self):
        # I average the n folds
        class_prob_predictions_avg = np.mean(self.class_prob_predictions, axis=0)
        classes_predicted = np.argmax(class_prob_predictions_avg, axis=1)

        # convert to integers
        classes_predicted = classes_predicted.astype('int64')
        diff = np.equal(classes_predicted, self.class_y_predictions[0])
        unique, counts = np.unique(diff, return_counts=True)
        diff = dict(zip(unique, counts))
        acc = 100.0 - (diff[False] * 100.0 / (diff[True] + diff[False]))
        print(f'general accuracy: {acc}')


class ExtendedStatsValidator(CommonBatchValidator):
    def __init__(self):
        super().__init__()

    def __call__(self, batch_num, batch_data: BatchData, predict):
        super().__call__(batch_num, batch_data, predict)

    def fold_start(self, fold):
        super().fold_start(fold)

    def fold_finish(self, fold):
        super().fold_finish(fold)

    def validation_start(self, folds_count, n_test_images, num_classes):
        super().validation_start(folds_count, n_test_images, num_classes)

    def validation_finish(self):
        class_prob_predictions_avg = np.mean(self.class_prob_predictions, axis=0)
        y_pred = np.argmax(class_prob_predictions_avg, axis=1).astype('int64')
        y_true = self.class_y_predictions[0]

        print('Accuracy score: %2.2f%%' % (accuracy_score(y_true, y_pred)))
        print(classification_report(y_true, y_pred))

        sns.heatmap(confusion_matrix(y_true, y_pred),
                    annot=True,
                    fmt="d",
                    cbar=False,
                    cmap="YlGnBu",
                    vmax=self._n_test_images // 16)
        plt.show()


def _count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


def filenames(base_dir, pattern):
    return tf.io.gfile.glob(os.path.join(base_dir, f'{pattern}*.tfrec'))


class TFValidator:
    def __init__(self,
                 device: str,
                 source_dir: str,
                 validators: List[BatchValidator],
                 img_size: int = 512,
                 folds_count: int = 5,
                 batch_size: int = 128,
                 eff_net: int = 4,
                 verbose: bool = True
                 ):
        self._device = device
        self._source_dir = source_dir
        self.img_size = img_size
        self._folds_count = folds_count
        self._batch_size = batch_size
        self._eff_net = eff_net
        self.verbose = verbose
        self.validators = validators

        s, a, r = infer_strategy(device, verbose)
        self._strategy = s
        self._auto = a
        self._replicas = r

        self._num_classes = 5

        self._img_size_list = [self.img_size, self.img_size]
        self._data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.4),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.2)
        ])

        assert (os.path.exists(self._source_dir))

    def _decode_image(self, image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
        image = tf.reshape(image, [*self._img_size_list, 3])  # explicit size needed for TPU
        return image

    def _parse_tfrec(self, example):
        LABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            'patient_id': tf.io.FixedLenFeature([], tf.int64),
            'side': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

        image = self._decode_image(example['image'])
        patient_id = example['patient_id']
        side = example['side']
        level = example['label']

        return image, patient_id, side, level

    def _build_dataset(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=self._auto)
        dataset = dataset.cache()
        dataset = dataset.map(self._parse_tfrec)

        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(self._auto)  # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def _build_model(self, ef=0):
        inp = tf.keras.layers.Input(shape=(*self._img_size_list, 3))
        # apply data augmentation as part of the model
        x = self._data_augmentation(inp)

        base = utils.efns()[ef](input_shape=(*self._img_size_list, 3), weights='imagenet', include_top=False)

        x = base(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self._num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=inp, outputs=x)

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def validate(self, pattern='train', limit=2 ** 32):
        if self.verbose:
            print('Reading tfrecords')

        validation_filenames = filenames(self._source_dir, pattern)
        n_test_images = min(_count_data_items(validation_filenames), limit)

        if self.verbose:
            print(f'Preparing validation for {n_test_images} images')

        for validator in self.validators:
            validator.validation_start(self._folds_count, n_test_images, self._num_classes)

        tStart = time.time()

        for fold in range(0, self._folds_count):
            # reload the data
            test_dataset = self._build_dataset(validation_filenames)

            print('Predictions using model fold ', fold)

            K.clear_session()
            with self._strategy.scope():
                model = self._build_model(ef=4)

            model.load_weights(f'fold-{fold}.h5')

            for validator in self.validators:
                validator.fold_start(fold)

            batch_num = 0

            for image, patient_id, side, level in iter(test_dataset):
                if batch_num * self._batch_size + self._batch_size > limit:
                    print(f'limit exceeded, stopping on {batch_num * self._batch_size}')
                    break

                if self.verbose:
                    print(f'Working on batch num: {batch_num}, img num: {batch_num * self._batch_size}...')

                # single batch prediction
                preds_batch = model.predict(image.numpy())

                batch_data = BatchData(image, patient_id, side, level)
                for validator in self.validators:
                    validator(batch_num, batch_data, preds_batch)

                batch_num += 1

            # end of fold
            for v in self.validators:
                v.fold_finish(fold)

        # in the end
        for validator in self.validators:
            validator.validation_finish()

        tElapsed = round(time.time() - tStart, 1)

        print(' ')
        print('Time (sec) elapsed for fold: ', tElapsed)
        print('...')
        print('...')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    v = TFValidator(
        device='GPU',
        source_dir=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\tfrec_no_gaussian',
        validators=[ExtendedStatsValidator()],
        img_size=512,
        folds_count=5,
        batch_size=64,
        eff_net=4,
        verbose=True
    )
    v.validate(pattern='train', limit=10_048)
