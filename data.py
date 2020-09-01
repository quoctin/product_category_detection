import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import pathlib
from utils import *
import random
import math
import numpy as np
import os
from PIL import Image
import cv2
import copy
import glob
from tqdm import tqdm
import hashlib
from collections import defaultdict

flags = tf.app.flags
FLAGS = flags.FLAGS

class Data:
    def __init__(self, data_path, batch_size=32, augmentation=False, split_ratio=[0.6,0.2,0.2], use_tfrecord=True):
        self.data_path = data_path
        self.im_size = [FLAGS.im_size, FLAGS.im_size]
        self.batch_size = batch_size 
        self.augmentation = augmentation
        self.split_ratio = split_ratio
        self.use_tfrecord = use_tfrecord
    
    def preprocess_image(self, image, augmentation):
        def is_crop(image):
            cands = tf.cast(tf.where((image[:,:,1]<200) & (image[:,:,1]>50)), tf.int32)
            ret = tf.cond(
                tf.greater(tf.shape(cands)[0], 16*16),
                lambda: tf.constant(True),
                lambda: tf.constant(False)
            )
            return ret

        def smart_crop_(image):
            shape = tf.shape(image)
            h,w = shape[0], shape[1]
            cands = tf.cast(tf.where((image[:,:,1]<200) & (image[:,:,1]>50)), tf.int32)
            min_x, max_x = tf.argmin(cands[:,1]), tf.argmax(cands[:,1], axis=0)
            min_y, max_y = tf.argmin(cands[:,0]), tf.argmax(cands[:,0], axis=0)
            dx, dy = cands[max_x,1]-cands[min_x,1], cands[max_y,0]-cands[min_y,0]
            delta_yx, delta_xy = dy-dx, dx-dy
            t_x = tf.cond(
                dy > dx, \
                lambda: cands[min_x,1] - tf.floordiv(delta_yx,2),
                lambda: cands[min_x,1]
            )
            b_x = tf.cond(
                dy > dx, \
                lambda: cands[max_x,1] + tf.floordiv(delta_yx,2),
                lambda: cands[max_x,1]
            )
            t_y = tf.cond(
                dx > dy, \
                lambda: cands[min_y,0] - tf.floordiv(delta_xy,2),
                lambda: cands[min_y,0]
            )
            b_y = tf.cond(
                dx > dy, \
                lambda: cands[max_y,0] + tf.floordiv(delta_xy,2),
                lambda: cands[max_y,0]
            )
            t_x = tf.clip_by_value(t_x, 0, w-1)
            t_y = tf.clip_by_value(t_y, 0, h-1)
            b_x = tf.clip_by_value(b_x, 0, w-1)
            b_y = tf.clip_by_value(b_y, 0, h-1)
            image = tf.cond(
                tf.math.logical_and(b_x - t_x > 0, b_y - t_y > 0),
                lambda: tf.image.crop_to_bounding_box(image, t_y, t_x, b_y - t_y, b_x - t_x),
                lambda: image
            )
            return image
            
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cond(is_crop(image), lambda: smart_crop_(image), lambda: image)
        image = tf.cast(image, tf.float32)/255.0
        
        if augmentation:
            image = self.augment_data(image)
        image = tf.image.resize(image, self.im_size)
        image = tf.clip_by_value(image, 0., 1.)
        return image
    
    def preprocess_tfrecord_image(self, image, augmentation):
        image = tf.decode_raw(image, tf.float32)
        size = copy.deepcopy(self.im_size)
        size.append(3)
        image = tf.reshape(image, size)
        image = image/255.0
        if augmentation:
            image = self.augment_data(image)
        image = tf.clip_by_value(image, 0., 1.)
        return image

    def augment_data(self, image):
        '''
        image: 3-channel decoded image
        '''
        def rotate_(x):
            # Rotate 0, 90, 180, 270 degrees
            return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        def center_crop_(x):
            return tf.image.resize(tf.image.central_crop(x, 0.8), self.im_size)
        def flip_(x):
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)
            return x
        def color_(x):
            x = tf.image.random_hue(x, 0.08)
            x = tf.image.random_saturation(x, 0.6, 1.6)
            x = tf.image.random_brightness(x, 0.05)
            x = tf.image.random_contrast(x, 0.7, 1.3)
            return x
        image = tf.cond(tf.random_uniform([], 0, 1) > 0.95, lambda: rotate_(image), lambda: image)
        image = tf.cond(tf.random_uniform([], 0, 1) > 0.95, lambda: flip_(image), lambda: image)
        image = tf.cond(tf.random_uniform([], 0, 1) > 0.95, lambda: color_(image), lambda: image)
        image = tf.cond(tf.random_uniform([], 0, 1) > 0.95, lambda: center_crop_(image), lambda: image)
        return image

    def convert_to_tf_example(self, image, label):
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        def _int_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int_feature(label),
            'image':_bytes_feature(tf.compat.as_bytes(image.tostring()))}))
        return example

    def smart_crop_np(self, I):
        h,w,_ = I.shape
        cands = np.where((I[:,:,1] < 200) & (I[:,:,1] > 50))
        if cands[1].shape[0] < 16*16:
            return I
        min_x = np.argmin(cands[1])
        max_x = np.argmax(cands[1])
        min_y = np.argmin(cands[0])
        max_y = np.argmax(cands[0])
        dx, dy = cands[1][max_x]-cands[1][min_x], cands[0][max_y]-cands[0][min_y]
        if dy > dx:
            delta = dy-dx
            tx = (cands[1][min_x] - delta//2) if (cands[1][min_x] - delta//2) > 0 else 0
            bx = (cands[1][max_x] + delta//2) if (cands[1][max_x] + delta//2) < w else w - 1
        else:
            tx, bx = cands[1][min_x], cands[1][max_x]
        if dx > dy:
            delta = dx-dy
            ty = (cands[0][min_y] - delta//2) if (cands[0][min_y] - delta//2) > 0 else 0
            by = (cands[0][max_y] + delta//2) if (cands[0][max_y] + delta//2) < h else h - 1
        else:
            ty, by = cands[0][min_y], cands[0][max_y]
        return I[ty:by, tx:bx, :]
    
    def save_to_tfrecord(self, b_train, b_val, b_test):
        if not os.path.exists(self.data_path+'_tfrecords'):
            os.makedirs(self.data_path+'_tfrecords')
        else:
            record_files = glob.glob(os.path.join(self.data_path+'_tfrecords', '*.tfrecords'))
            if len(record_files) > 0:
                return # do nothing
        data = [b_train, b_val, b_test]
        tfrec_size = int(10**8 / (FLAGS.im_size*FLAGS.im_size*12))
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

        error_idx = defaultdict(set)
        for j,cat in enumerate(['train', 'val', 'test']):
            for i,(file_path,label) in tqdm(enumerate(zip(data[j][0], data[j][1]))):
                try:
                    if i%tfrec_size == 0:
                        # open new file writer every tfrec_size
                        f = os.path.join(self.data_path+'_tfrecords', cat+str(i//tfrec_size)+'.tfrecords')
                        writer = tf.python_io.TFRecordWriter(f, options)
                    image = Image.open(file_path)
                    image = np.array(image, np.float32)
                    if image.shape[2] > 3:
                        print(file_path, '{} channels'.format(image.shape[2]))
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

                    #image = self.smart_crop_np(image)
                    image = cv2.resize(image, tuple(self.im_size))
                    example = self.convert_to_tf_example(image, np.int64(label))

                    if image.shape[2] != 3: 
                        #print(file_path, '{} dims, {} channels'.format(image.ndim, image.shape[2]))
                        raise Exception(file_path + '{} dims, {} channels'.format(image.ndim, image.shape[2]))

                    writer.write(example.SerializeToString())
                    if (i+1)%tfrec_size == 0:
                        writer.close()
                        writer = None

                except Exception as e:
                    error_idx[cat].add(i)
                    print(file_path, ' error: ', str(e))
                    if (i+1)%tfrec_size == 0:
                        writer.close()
                        writer = None
                    continue
            if writer is not None:
                writer.close()

    
    def extract_and_preprocess_tfrecord(self, serialized_example):
        features = {
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        }
        sample = tf.parse_single_example(serialized_example, features)
        with tf.device('/cpu:0'):
            return self.preprocess_tfrecord_image(sample['image'], self.augmentation), tf.cast(sample['label'], tf.float32)

    def load_and_preprocess_image(self, path, augmentation):
        image = tf.io.read_file(path)
        return self.preprocess_image(image, augmentation)

    def load_and_preprocess_image_label(self, path, label, augmentation):
        with tf.device('/cpu:0'):
            return self.load_and_preprocess_image(path, augmentation), tf.to_float(label)
    
    def train_val_test_split(self, data, labels):
        assert len(data) == len(labels), 'Dimensions of data and labels are mismatched'
        img_count = len(labels)
        train_data = data[:int(math.floor(img_count*self.split_ratio[0]))]
        train_image_labels = labels[:int(math.floor(img_count*self.split_ratio[0]))]
        offset = int(math.floor(img_count*self.split_ratio[0]))
        val_data = data[offset:offset+int(math.floor(img_count*self.split_ratio[1]))]
        val_image_labels = labels[offset:int(offset+math.floor(img_count*self.split_ratio[1]))]
        offset += int(math.floor(img_count*self.split_ratio[1]))
        test_data = data[offset:offset+int(math.floor(img_count*self.split_ratio[2]))]
        test_image_labels = labels[offset:offset+int(math.floor(img_count*self.split_ratio[2]))]
        return (train_data, train_image_labels), \
            (val_data, val_image_labels), \
            (test_data, test_image_labels)

    def get_data(self, label_to_index=None, file_paths=None):
        data_root = pathlib.Path(self.data_path)
        if file_paths is None:
            image_paths = list(data_root.glob('*/*.jpg')) + list(data_root.glob('*/*.JPG'))
        else:
            image_paths = [line.rstrip('\n') for line in open(file_paths)]
        image_paths = [str(p) for p in image_paths]
        image_paths = sorted(image_paths, key=lambda p: hashlib.sha1(p.encode()).hexdigest())
        
        # get labels
        if label_to_index is None:
            label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
            label_to_index = dict((name, index) for index,name in enumerate(label_names))
        else:
            #folders = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
            label_names = label_to_index.keys()
            #for name in label_names:
            #    assert name in folders, 'cannot get data corresponding to label name ' + name
        image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in image_paths]
        
        # split into training, testing and validation
        b_train, b_val, b_test = self.train_val_test_split(image_paths, image_labels)
        img_count = len(b_train[0])
        compression = 'GZIP'
        if self.use_tfrecord:
            # preprocess from tfrecord
            self.save_to_tfrecord(b_train, b_val, b_test)
            record_files = glob.glob(os.path.join(self.data_path + '_tfrecords', 'train*.tfrecords'))
            ds_train = tf.data.TFRecordDataset(record_files, compression_type=compression).repeat()
            ds_train = ds_train.map(self.extract_and_preprocess_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds_train = ds_train.shuffle(buffer_size=img_count//10)
            ds_train = ds_train.batch(self.batch_size)
            ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            
            record_files = glob.glob(os.path.join(self.data_path + '_tfrecords', 'val*.tfrecords'))
            ds_val = tf.data.TFRecordDataset(record_files, compression_type=compression).repeat()
            augmentation = self.augmentation
            self.augmentation = False
            ds_val = ds_val.map(self.extract_and_preprocess_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.augmentation = augmentation
            ds_val = ds_val.batch(self.batch_size)
            ds_val = ds_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            record_files = glob.glob(os.path.join(self.data_path + '_tfrecords', 'test*.tfrecords'))
            ds_test = tf.data.TFRecordDataset(record_files, compression_type=compression).repeat()
            augmentation = self.augmentation
            self.augmentation = False
            ds_test = ds_test.map(self.extract_and_preprocess_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.augmentation = augmentation
            ds_test = ds_test.batch(self.batch_size)
            ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        else:
            # preprocess image, label
            ds_train = tf.data.Dataset.from_tensor_slices(b_train)
            ds_train = ds_train.shuffle(buffer_size=img_count)
            ds_train = ds_train.repeat()
            ds_train = ds_train.map(lambda x, y: self.load_and_preprocess_image_label(x, y, self.augmentation), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds_train = ds_train.batch(self.batch_size)
            ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            ds_val = tf.data.Dataset.from_tensor_slices(b_val)
            ds_val = ds_val.repeat()
            ds_val = ds_val.map(lambda x, y: self.load_and_preprocess_image_label(x, y, False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds_val = ds_val.batch(self.batch_size)
            ds_val = ds_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            
            ds_test = tf.data.Dataset.from_tensor_slices(b_test)
            ds_test = ds_test.repeat()
            ds_test = ds_test.map(lambda x, y: self.load_and_preprocess_image_label(x, y, False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds_test = ds_test.batch(self.batch_size)
            ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds_train, len(b_train[0]), ds_val, len(b_val[0]), ds_test, len(b_test[0])

if __name__ == "__main__":
    import matplotlib 
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    data = Data('data', augmentation=False)
    _,_,ds,_,_,_ = data.get_data(label_to_index={'logo':1, 'nonlogo':0}, file_paths='data/file_paths.txt')
    iterator = tf.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(ds), \
                tf.compat.v1.data.get_output_shapes(ds))
    
    val_initializer = iterator.make_initializer(ds)
    images, labels = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        sess.run(val_initializer)
        for i in range(5):
            images_ = sess.run(images)
            print(images_[0,50:55,50:55,0])
            plt.imshow(images_[0,:,:,:])
            plt.savefig(str(i)+'.png')

    # tf.enable_eager_execution()
    # ds_train, train_n, ds_val, val_n, ds_test, test_n = data.get_data()
    # iterator = tf.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(ds_train),\
    #                 tf.compat.v1.data.get_output_shapes(ds_train))
    # for i in range(test_n//32):
    #     images, labels = next(iter(ds_test))
    #     labels = tf.expand_dims(labels, axis=1)
    #     print('Images shape: ', images.shape)

            
