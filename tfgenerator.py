import os
import time
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

sns.set()


def preprocess_no_gaussian(image, img_pixel):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image.ndim == 3
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray_img > 7

    check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if (check_shape != 0):  # image is not too dark
        img1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        image = np.stack([img1, img2, img3], axis=-1)
    image = cv2.resize(image, (img_pixel, img_pixel), interpolation=cv2.INTER_AREA)

    return image


def plot_images(images: List[List], height=5):
    assert len(images)
    assert len(images[0])

    rows = len(images)
    cols = len(images[0])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig = plt.figure(figsize=(height * cols, height * rows))
    for i in range(0, rows * cols):
        img = images[i // cols][i % cols]
        fig.add_subplot(rows, cols, i + 1)
        if img.ndim == 2:
            plt.gray()
        plt.imshow(img)
    plt.show()


class TFGenerator:
    IMG_PIXEL = 512

    def __init__(self,
                 file_labels,
                 source_dir,
                 destination_dir,
                 sample_dir,
                 batch_size,
                 preprocess_image_callback):

        self.file_labels = file_labels
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        self.sample_dir = sample_dir
        self.batch_size = batch_size
        self.preprocess_image_callback = preprocess_image_callback

        assert os.path.exists(self.file_labels)
        assert os.path.exists(self.source_dir)
        assert os.path.exists(self.destination_dir)

    @staticmethod
    def get_img(filename: str):
        return cv2.imread(filename)

    @staticmethod
    def get_tensor(cv2_img, img_size=512, preprocess_image_callback=preprocess_no_gaussian):
        img = preprocess_image_callback(cv2_img, img_size)
        img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tobytes()
        # img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        # img_tensor = tf.expand_dims(img_tensor, 0)
        img_tensor = tf.image.decode_jpeg(img, channels=3)
        img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
        img_tensor = tf.expand_dims(img_tensor, 0)

        return img_tensor

    def generate_all(self, file_label, tf_limit):
        # dataframe where to take metadata
        df_labels = pd.read_csv(self.file_labels)
        print(f'df_labels head: {df_labels.head()}')
        print(f'df_labels hist:')
        print(df_labels.groupby(["level"]).size())
        df = df_labels
        SIZE = self.batch_size

        # imgs to process
        IMGS = df['image'].values

        CT = len(IMGS) // SIZE + int(len(IMGS) % SIZE != 0)
        # for test purposes: lower number of train batches to 1
        CT = min(CT, tf_limit or 1_000_000)

        count = 0

        for j in range(CT):
            print()
            print('Writing TFRecord %i of %i...' % (j + 1, CT))
            tStart = time.time()

            CT2 = min(SIZE, len(IMGS) - j * SIZE)

            with tf.io.TFRecordWriter(
                    os.path.join(self.destination_dir, '%s%.2i-%i.tfrec' % (file_label, j, CT2))) as writer:
                for k in range(CT2):
                    index = SIZE * j + k
                    img_path = os.path.join(self.source_dir, df.iloc[index].image) + '.jpeg'

                    imgs = []
                    source = cv2.imread(img_path)
                    imgs.append(source)

                    # per default CV2 legge in BGR
                    # img = cv2.resize(img, (IMG_PIXEL, IMG_PIXEL), interpolation = cv2.INTER_AREA)
                    # imgs.extend(preprocess_image(source, self.IMG_PIXEL))
                    # img = imgs[-1]

                    img = self.preprocess_image_callback(source, self.IMG_PIXEL)
                    cv2.imwrite(
                        os.path.join(self.sample_dir,
                                     f'{df.iloc[index].level}_transformed_{df.iloc[index].image}.jpeg'),
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                    # plot_images([imgs])

                    # potrei cambiare la qualità !!! portarla al 100%
                    img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tobytes()

                    name = IMGS[index]

                    # get the row from Dataframe
                    row = df.iloc[index]

                    # get patientId
                    patientID = int(row['image'].split('_')[0])

                    # encode side: left = 0, right = 1
                    if 'left' in row['image']:
                        side = 0
                    else:
                        side = 1

                    level = row['level']

                    # build the record
                    # image, patientid, side, label
                    example = serialize_example(img, patientID, side, level)

                    writer.write(example)

                    # print progress
                    if k % 100 == 0: print('#', '', end='')

            tEnd = time.time()

            print('')
            print('Elapsed: ', round((tEnd - tStart), 1), ' (sec)')

    def generate_balanced(self, file_label, tf_limit):
        retina_df = pd.read_csv(self.file_labels)
        retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
        retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(self.source_dir,
                                                                          '{}.jpeg'.format(x)))
        retina_df['exists'] = retina_df['path'].map(os.path.exists)
        print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')
        retina_df['eye'] = retina_df['image'].map(lambda x: 1 if x.split('_')[-1] == 'left' else 0)
        from keras.utils.np_utils import to_categorical
        retina_df['level_cat'] = retina_df['level'].map(lambda x: to_categorical(x, 1 + retina_df['level'].max(), 5))

        retina_df.dropna(inplace=True)
        retina_df = retina_df[retina_df['exists']]
        print(f'samples: {retina_df.sample(3, ignore_index=False)}')
        print(f'Initial hist:')
        retina_df[['level', 'eye']].hist(figsize=(10, 5), backend='matplotlib')
        plt.show()
        from sklearn.model_selection import train_test_split
        rr_df = retina_df[['PatientId', 'level']].drop_duplicates()
        train_ids, valid_ids = train_test_split(rr_df['PatientId'],
                                                test_size=0.25,
                                                random_state=2018,
                                                stratify=rr_df['level'])
        raw_train_df = retina_df[retina_df['PatientId'].isin(train_ids)]
        valid_df = retina_df[retina_df['PatientId'].isin(valid_ids)]
        print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
        train_df = raw_train_df.groupby(['level', 'eye']).apply(lambda x: x.sample(100, replace=True)
                                                                ).reset_index(drop=True)
        print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
        train_df[['level', 'eye']].hist(figsize=(10, 5))
        plt.show()
        df = train_df
        SIZE = self.batch_size

        # imgs to process
        IMGS = df['image'].values

        CT = len(IMGS) // SIZE + int(len(IMGS) % SIZE != 0)
        # for test purposes: lower number of train batches to 1
        CT = min(CT, tf_limit or 1_000_000)

        count = 0

        for j in range(CT):
            print()
            print('Writing TFRecord %i of %i...' % (j + 1, CT))
            tStart = time.time()

            CT2 = min(SIZE, len(IMGS) - j * SIZE)

            with tf.io.TFRecordWriter(
                    os.path.join(self.destination_dir, '%s%.2i-%i.tfrec' % (file_label, j, CT2))) as writer:
                for k in range(CT2):
                    index = SIZE * j + k
                    item = df.iloc[index]
                    img_path = item.path
                    source = cv2.imread(img_path)
                    img = self.preprocess_image_callback(source, self.IMG_PIXEL)
                    cv2.imwrite(
                        os.path.join(self.sample_dir,
                                     f'{item.level}_transformed_{item.image}.jpeg'),
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tobytes()
                    patientID = int(item.PatientId)
                    side = 1 if int(item.eye) == 0 else 0
                    level = item['level']
                    example = serialize_example(img, patientID, side, level)
                    writer.write(example)
                    if k % 10 == 0: print('#', '', end='')
            tEnd = time.time()
            print('')
            print('Elapsed: ', round((tEnd - tStart), 1), ' (sec)')

    def generate(self, file_label='train', tf_limit=None, balanced=False):
        if balanced:
            self.generate_balanced(file_label, tf_limit)
        else:
            self.generate_all(file_label, tf_limit)


def crop_image_from_gray(img, tol=7):
    result = []
    if img.ndim == 2:
        mask = img > tol
        result.append(img[np.ix_(mask.any(1), mask.any(0))])
        return result
    elif img.ndim == 3:
        print(f'ndim before gray: {img.ndim}')
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        red_img = img[:, :, 1]
        print(f'ndim after gray: {gray_img.ndim}')
        print(f'ndim red img: {red_img.ndim}')
        result.append(gray_img)
        result.append(red_img)
        mask = red_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]

            img = np.stack([img1, img2, img3], axis=-1)
            result.append(img)

        return result


def preprocess_image(image, img_pixel):
    result = []
    crop = True
    blur = True
    IMG_PIXEL = img_pixel

    sigmaX = 10
    # CV2 per default tratta le immagini come BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result.append(image)

    if crop == True:
        imgs = crop_image_from_gray(image)
        result.extend(imgs)
        image = imgs[-1]

    image = cv2.resize(image, (IMG_PIXEL, IMG_PIXEL), interpolation=cv2.INTER_AREA)
    result.append(image)

    if blur == True:
        result.append(cv2.GaussianBlur(image, (0, 0), sigmaX))
        result.append(cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 96))
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        result.append(np.stack([r, g + 20, b], axis=-1))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
        result.append(image)

    return result


def preprocess_gaussian(image, img_pixel):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image.ndim == 3
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray_img > 7

    check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if (check_shape != 0):  # image is not too dark
        img1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        image = np.stack([img1, img2, img3], axis=-1)
    image = cv2.resize(image, (img_pixel, img_pixel), interpolation=cv2.INTER_AREA)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)

    return image


def serialize_example(img, patient_id, side, label):
    feature = {
        'image': _bytes_feature(img),
        'patient_id': _int64_feature(patient_id),
        'side': _int64_feature(side),  # 0,1, left,right
        'label': _int64_feature(label)  # [0, 4]
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Test data generation
def generate_all():
    print(f'tensorflow version: {tf.__version__}')
    gtrn = TFGenerator(
        file_labels=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\trainLabels.csv',
        source_dir=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\train',
        destination_dir=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\tfrec_no_gaussian',
        sample_dir=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\sample\processed',
        batch_size=2000,
        preprocess_image_callback=preprocess_no_gaussian
    )
    gtrn.generate('train')

    # gtst = TFGenerator(
    #     file_labels=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\sampleSubmission.csv',
    #     source_dir=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\test',
    #     destination_dir=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\tfrec_no_gaussian',
    #     sample_dir=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\sample\processed',
    #     batch_size=2000,
    #     preprocess_image_callback=preprocess_no_gaussian
    # )
    # gtst.generate('test', tf_limit=10)


def generate_balanced():
    print(f'tensorflow version: {tf.__version__}')
    gtrn = TFGenerator(
        file_labels=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\trainLabels.csv',
        source_dir=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\train',
        destination_dir=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\tfrec_balanced',
        sample_dir=r'G:\Storage\datasets\retinopaty\source\diabetic-retinopathy-detection\sample\processed',
        batch_size=200,
        preprocess_image_callback=preprocess_no_gaussian
    )
    gtrn.generate('train', tf_limit=5, balanced=True)


if __name__ == '__main__':
    if True:
        generate_all()
    else:
        generate_balanced()
