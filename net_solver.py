from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import os
import pickle
import shutil
import numpy as np 
from tqdm import tqdm
slim = tf.contrib.slim
from data import Data
summary = tf.compat.v1.summary
import sys
import glob
from sklearn.metrics import *

class NetSolver:
    def __init__(self, working_dir, batch_size=32, max_iter=1e5, val_iter=5e3, save_iter=5e3, log_iter=100, \
                 learning_rate=0.0001, lr_start_decay=None, lr_decay_every=None, pretrained_resnet=None, \
                 colab_drive=None, label_to_index=None):
        self.working_dir = working_dir
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.val_iter = val_iter
        self.save_iter = save_iter
        self.log_iter = log_iter
        self.pretrained_resnet = pretrained_resnet
        self.learning_rate = learning_rate
        self.lr_start_decay = lr_start_decay
        self.lr_decay_every = lr_decay_every
        self.colab_drive = colab_drive
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=config)
        self.label_to_index = label_to_index

        self.input_val_err = tf.placeholder(dtype=tf.float32, shape=())
        self.current_val_err = tf.Variable(1.0, trainable=False, dtype=tf.float32, shape=())
        self.val_err_op = self.current_val_err.assign(self.input_val_err)
    
    def save_log(self):
        path = os.path.join(self.working_dir,'model','log.pkl')
        with open(path,'wb') as f:
            pickle.dump(self.log, f)
    
    def load_log(self):
        path = os.path.join(self.working_dir,'model','log.pkl')
        if os.path.exists(path):
            with open(path,'rb') as f:
                data = pickle.load(f)
                self.log['costs'] = data['costs']
                self.log['val_err'] = data['val_err']
                self.log['train_err'] = data['val_err']

    def load_model(self, ckpt_id=None):
        saver = tf.train.Saver()
        path = os.path.join(self.working_dir, 'model')
        if ckpt_id is not None:
            ckpt =  os.path.join(path, 'saved-model-' + str(ckpt_id))
            saver.restore(self.sess, ckpt)
            print('\nLoaded %s\n'%ckpt)
        else:
            ckpt = tf.train.latest_checkpoint(path)
            print('\nFound latest model: %s\n'%ckpt)
            if ckpt:
                saver.restore(self.sess, ckpt)
                print('\nLoaded %s\n'%ckpt)

    def save_model(self):
        saver = tf.train.Saver()

        if not os.path.isdir(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
        if not os.path.isdir(os.path.join(self.working_dir, 'figure')):
            os.makedirs(os.path.join(self.working_dir, 'figure'), exist_ok=True)
        if not os.path.isdir(os.path.join(self.working_dir, 'model')):
            os.makedirs(os.path.join(self.working_dir, 'model'), exist_ok=True)
        path = os.path.join(self.working_dir, 'model','saved-model')
        save_path = saver.save(self.sess, path, global_step=self.net.global_iter.eval(self.sess))
        print('\nSave dir %s\n' % save_path)
        if self.colab_drive is not None and \
           os.path.exists(self.colab_drive):
            print('Start copying ')
            print('Current checkpoint: %s' % str(self.net.global_iter.eval(self.sess)))
            # copy saved model to Google drive
            dst = os.path.join(self.colab_drive, self.working_dir, 'model')
            if not os.path.exists(dst):
                os.makedirs(dst)
            for f in glob.glob(os.path.join(self.working_dir, 'model', \
                '*-'+str(self.net.global_iter.eval(self.sess))+'.*')):
                shutil.copy(f, dst)
                print('copy from {} to {}'.format(f, dst))
                
            # copy log to Google drive
            shutil.copy(os.path.join(self.working_dir, 'model', 'log.pkl'), dst)
            # copy summaries to Google drive
            src = os.path.join(self.working_dir, 'summaries')
            dst = os.path.join(self.colab_drive, self.working_dir, 'summaries')
            if not os.path.exists(dst):
                os.makedirs(dst)
            for parent in glob.glob(os.path.join(src, 'train_it*')):
                dst = os.path.join(self.colab_drive, parent)
                if not os.path.exists(dst):
                    os.makedirs(dst)
                for f in glob.glob(os.path.join(parent, 'events*')):
                    shutil.copy(f, dst)
                    print('copy from {} to {}'.format(f, dst))

    def load_resnet(self):
        all_variables = slim.get_model_variables()
        vars_to_restore = []
        exclusions = self.net.exclude_finetune_scopes()
        for var in all_variables:
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
                vars_to_restore.append(var)
        init_fn = slim.assign_from_checkpoint_fn(self.pretrained_resnet, vars_to_restore,
            ignore_missing_vars=False)
        init_fn(self.sess)
        print('\nLoaded variables from %s\n'%self.pretrained_resnet)

    def setup_summary(self):
        self.summary = [
            summary.scalar('total_loss', self.net.loss),
            summary.scalar('learning_rate', self.net._opt._lr),
            summary.scalar('validation_error', self.current_val_err)
        ]
        for grad, var in self.net.grad:
            self.summary.append(summary.histogram(var.name + '/gradient', grad))
        # self.stats_summary = summary.text('dataset_stats', self.stats_aggregator.get_summary())
        # tf.add_to_collection(tf.GraphKeys.SUMMARIES, stats_summary)
        self.merged_summary = summary.merge_all()
    
    def add_summary(self):
        self.train_writer.add_summary(self.sess.run(self.merged_summary, \
            feed_dict={self.net.learning_rate:self.learning_rate}), global_step=self.i)
        # self.train_writer.add_summary(self.sess.run(self.stats_summary), global_step=self.i)
        
    def setup_net(self, net, create_summary=True, ckpt_id=None):
        self.net = net
        self.log = {'costs':[], 'val_err':[], 'train_err':[]}
        if ckpt_id:
            self.load_model(ckpt_id=ckpt_id)
            self.load_log()
            self.i = self.net.global_iter.eval(session=self.sess)
            if self.i >= self.lr_start_decay:
                self.learning_rate /= 2**((self.i-self.lr_start_decay)//self.lr_decay_every)
                self.learning_rate = min(1e-5, self.learning_rate)
        else:
            print('Initializing from scratch')
            self.sess.run(tf.global_variables_initializer())
            self.i = 0
            if self.pretrained_resnet is not None:
                assert os.path.exists(self.pretrained_resnet), 'Resnet checkpoint not found'
                self.load_resnet()
        
        if create_summary:
            summary_dir = os.path.join(self.working_dir, 'summaries', 'train_it_%d' % self.i)
            if os.path.isdir(summary_dir):
                shutil.rmtree(summary_dir)
            os.makedirs(summary_dir, exist_ok=True)
            self.train_writer = summary.FileWriter(summary_dir, self.sess.graph)
        self.start_i = self.i

    def get_data_source(self, src='train'):
        assert src in ['train', 'val', 'test'], 'src is unsupported'
        # setup net's data source
        initializer = {'train':self.train_initializer, \
                       'val':self.val_initializer, \
                       'test':self.test_initializer}
        with tf.device('/cpu:0'):
            self.sess.run(initializer[src])
        return self.images, self.labels

    def setup_data(self, data_path, augmentation=False, use_tfrecord=True):
        #with tf.device('/cpu:0'):
        data = Data(data_path, augmentation=augmentation, use_tfrecord=use_tfrecord, split_ratio=[0.8,0.1,0.1], batch_size=self.batch_size) # set up also image size, batch size, augmentation, split ratio if needed
        self.ds_train, self.train_n, self.ds_val, self.val_n, self.ds_test, self.test_n = data.get_data(self.label_to_index)

        # to monitor tf datasets
        # self.stats_aggregator = tf.data.experimental.StatsAggregator()
        # options = tf.data.Options()
        # options.experimental_stats.aggregator = self.stats_aggregator
        # options.experimental_stats.latency_all_edges = True
        # self.ds_train = self.ds_train.with_options(options)

        self.iterator = tf.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(self.ds_train),\
                            tf.compat.v1.data.get_output_shapes(self.ds_train))
        self.train_initializer = self.iterator.make_initializer(self.ds_train)
        self.val_initializer = self.iterator.make_initializer(self.ds_val)
        self.test_initializer = self.iterator.make_initializer(self.ds_test)
        self.images, self.labels = self.iterator.get_next()
        self.labels = tf.expand_dims(self.labels, axis=1)
        

    def _train(self):
        [loss, _] = self.sess.run([self.net.loss, self.net.opt], feed_dict={
            self.net.is_training:True,
            self.net.learning_rate:self.learning_rate
        })
        return loss
    
    def validate(self, src='val', num_batches=None):
        if src == 'val':
            num_batches_ = (self.val_n // self.batch_size) if num_batches is None else num_batches
            print('num_batches: ', num_batches_)
        elif src == 'test': 
            num_batches_ = (self.test_n // self.batch_size) if num_batches is None else num_batches
        elif src == 'train':
            num_batches_ = (self.train_n // self.batch_size) if num_batches is None else num_batches
        else:
            raise ValueError('Data source is unsupported')
        
        # get data source
        self.get_data_source(src=src)
        sum_ones = 0
        try:
            for i in range(num_batches_):
                try:
                    X, y = self.sess.run((self.images, self.labels))
                    sum_ones += np.sum(np.array(y,np.int32)==1)
                except Exception as ex:
                    print(str(ex))
                    continue
                labels = y if i==0 else np.concatenate((labels, y))
                predictions = self.sess.run(self.net.cls, feed_dict={self.net.is_training:False, self.net.X:X}) if i==0  else \
                              np.concatenate((predictions, self.sess.run(self.net.cls, feed_dict={self.net.is_training:False, self.net.X:X})))
                soft_scores = self.sess.run(self.net.pred, feed_dict={self.net.is_training:False, self.net.X:X}) if i==0  else \
                              np.concatenate((soft_scores, self.sess.run(self.net.pred, feed_dict={self.net.is_training:False, self.net.X:X})))
                print('Label ', labels[-5:].T, ', soft scores ', soft_scores[-5:].T, ', predict ', predictions[-5:].T)
        except tf.errors.OutOfRangeError:
            print('OutOfRangeError')
        # change data source back to train (deprecated, no need when using feedable iterator)
        self.get_data_source(src='train')
        # compute error
        err = np.sum((labels.astype(np.int32) != predictions.astype(np.int32)), dtype=np.float32)/labels.shape[0]
        if src == 'val':
            self.sess.run(self.val_err_op, feed_dict={self.input_val_err:err})
        precision, recall, fscore, support = precision_recall_fscore_support(labels.astype(np.int32), predictions.astype(np.int32), labels=sorted(list(self.label_to_index.values())))
        return err, (precision, recall, fscore, support)
        

    def train(self):
        n_iters = int(self.max_iter - self.start_i)
        print('Train for %d iterations' % n_iters)
        t_obj = tqdm(range(n_iters))
        for t in t_obj:
            loss = self._train()
            print('Loss: %f\n' % loss)
            self.i += 1
            if self.i % self.save_iter == 0:
                self.save_model()
                self.save_log()
            if self.i % self.log_iter == 0:
                self.log['costs'].append(loss)
                self.add_summary()
            if self.i % self.val_iter == 0:
                val_err, _ = self.validate(src='val')
                print('Val error %f' % val_err)
                self.log['val_err'].append(val_err)
            if self.lr_start_decay is not None and \
                self.i >= self.lr_start_decay and \
                    (self.i-self.lr_start_decay)%self.lr_decay_every == 0:
                self.learning_rate = min(1e-5, self.learning_rate/2)

    def export_to_pb(self, export_path):
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        os.makedirs(export_path)
        tf.saved_model.simple_save(self.sess, \
                                    export_path, \
                                    inputs={'input_image': self.net.X, 
                                            'is_training': self.net.tensor_is_training}, \
                                    outputs={'predict_class': self.net.cls, 
                                             'score': self.net.pred})
        
def initialize(args):
    return NetSolver(**args)
