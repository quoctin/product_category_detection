from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import pathlib
import random
from data import Data
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
import tensorflow.contrib.layers as layers
import numpy as np

slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS

class Model():
    def __init__(self, 
                batch_size=32, 
                use_gpu=0, 
                num_class=2, 
                is_training=True, 
                loss_weight=1.0, 
                use_focal_loss=False,
                use_os_loss=False):
        self.batch_size = batch_size
        self.im_size = FLAGS.im_size
        self.use_gpu = use_gpu
        self.num_class = num_class
        self.is_training = is_training
        self.learning_rate = tf.placeholder(tf.float32, shape=())
        self.loss_weight = loss_weight
        self.use_focal_loss = use_focal_loss
        self.use_os_loss = use_os_loss

    def open_set_encode(self, y):
        return tf.math.mod(y + tf.expand_dims(y[:, -1], axis=-1), 2) # last index is for unknown class

    def set_data_source(self, im=None, label=None):
        if im is not None and label is not None:
            self.X = tf.placeholder_with_default(im, (None, self.im_size, self.im_size, 3), 'input_image')
            self.y = tf.placeholder_with_default(label, (None, 1), 'input_label')
        else:
            self.X = tf.placeholder(tf.float32, shape=(None, self.im_size, self.im_size, 3), name='input_image')
            self.y = tf.placeholder(tf.float32, shape=(None, 1), name='input_label')
        if self.num_class > 2:
            self.y_oh = tf.one_hot(tf.cast(tf.squeeze(self.y), tf.int32), depth=self.num_class, on_value=1)
            if self.use_os_loss:
                self.y_oh = self.open_set_encode(self.y_oh)

    def predict(self):
        with tf.device('/gpu:%d' % self.use_gpu):
            with tf.name_scope('extract_feature'):
                im_feat = self.extract_features(self.X, scope_name='feature_extraction', arch='resnet_v2_50')
            with tf.name_scope('predict'):
                logits = self.dense(im_feat, 'densenet')
        tf.get_variable_scope().reuse_variables()
        return logits

    def focal_loss(self, labels, logits, gamma=2.):
        alpha = self.loss_weight
        if self.num_class == 2:
            y_preds = tf.nn.sigmoid(logits)
            log_y_preds = tf.math.log_sigmoid(logits) # = -softplus(-logits)
            loss = -tf.reduce_mean((alpha/(alpha+1)) * labels * ((1 - y_preds) ** gamma) * log_y_preds + \
                    (1/(alpha+1)) * (1-labels) * (y_preds ** gamma) * (-logits + log_y_preds))
        else:
            alpha = tf.expand_dims(tf.constant(alpha, tf.float32), 0)
            y_preds = tf.nn.softmax(logits)
            log_y_preds = tf.nn.log_softmax(logits)
            loss = -tf.reduce_mean(alpha * labels * ((1 - y_preds) ** gamma) * log_y_preds)
        return loss

    def build_model(self):
        with tf.variable_scope(tf.get_variable_scope()):
            self.global_iter = tf.Variable(0, trainable=False)
            self.tensor_is_training = tf.placeholder_with_default([self.is_training], [1])
            self.is_training = tf.placeholder_with_default(tf.reshape(self.tensor_is_training, []), None)
            self.logits = self.predict()
            if self.use_focal_loss:
                if self.num_class == 2:
                    self.loss = self.focal_loss(labels=self.y, logits=self.logits)
                else:
                    self.loss = self.focal_loss(labels=tf.cast(self.y_oh, tf.float32), logits=self.logits)
            else:
                if self.num_class == 2:
                    self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=self.y, logits=self.logits, pos_weight=self.loss_weight))
                else:
                    #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_oh, logits=self.logits))
                    self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y_oh, 
                                                                               logits=self.logits, 
                                                                               weights=self.loss_weight, 
                                                                               reduction=tf.losses.Reduction.NONE))
            self._opt = tf.train.AdamOptimizer(self.learning_rate)
            self.grad = self._opt.compute_gradients(self.loss, var_list=self.get_variables())

        apply_grad_op = self._opt.apply_gradients(self.grad, global_step=self.global_iter)
        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if len(batchnorm_updates) != 0:
            batchnorm_updates_op = tf.group(*batchnorm_updates)
            self.opt = tf.group(apply_grad_op, batchnorm_updates_op)
            #print(batchnorm_updates_op)
        else:
            self.opt = apply_grad_op

        if self.num_class == 2:
            self.pred = tf.nn.sigmoid(self.logits)
            self.cls = tf.round(self.pred)
        else:
            self.pred = tf.nn.softmax(self.logits)
            self.cls = tf.expand_dims(tf.math.argmax(self.pred, axis=-1), axis=-1)
            if self.use_os_loss: # thresholding by 0.75
                max_pred = tf.math.reduce_max(self.pred[:,:-1], axis=1, keepdims=True)
                self.cls = tf.cast((max_pred>0.75), dtype=tf.int64)*self.cls + \
                           tf.cast((max_pred<=0.75), dtype=tf.int64)*(self.num_class-1)

    def get_variables(self):
        """
        Returns only variables that are needed.
        """
        var_list = tf.trainable_variables()
        assert len(var_list) > 0, 'No variables are linked to the optimizer'
        return var_list

    def dense(self, feat, name, reuse=False):
        out_dim = 1 if self.num_class==2 else self.num_class
        with tf.variable_scope(name, reuse=reuse):
            out = layers.stack(feat, layers.fully_connected, [128,64], scope='fc', reuse=reuse)
            if self.use_focal_loss and self.num_class==2:
                b = 0.01
                bias_init = None if self.use_os_loss else tf.compat.v1.constant_initializer(-np.log((1-b)/b)) 
                out = layers.fully_connected(out, 
                                             out_dim, 
                                             biases_initializer=bias_init, 
                                             activation_fn=None, 
                                             scope='fc_out', 
                                             reuse=reuse
                                             )
            else:
                bias_init = None if self.use_os_loss else tf.zeros_initializer()
                out = layers.fully_connected(out, 
                                             out_dim, 
                                             biases_initializer=bias_init,
                                             activation_fn=None, 
                                             scope='fc_out', 
                                             reuse=reuse)
        return out 

    def extract_features_resnet50(self, im, scope_name, reuse=False):
        use_global_pool = True
        output_dim = None
        with tf.name_scope(scope_name):
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                out, _ = resnet_v2.resnet_v2_50(inputs=im,
                                                num_classes=output_dim,
                                                global_pool=use_global_pool,
                                                is_training=self.is_training,
                                                scope='resnet_v2_50',
                                                reuse=reuse)
        out = layers.flatten(out)
        print('Output size of Resnet: ', out.shape)
        return out
    
    def extract_features_resnet101(self, im, scope_name, reuse=False):
        use_global_pool = True
        output_dim = 512
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            out, _ = resnet_v2.resnet_v2_101(im,
                                                num_classes=output_dim,
                                                is_training=False,
                                                global_pool=use_global_pool,
                                                reuse=reuse)
        out = layers.flatten(out)
        return out
    
    def extract_lenet(self, im, scope_name):
        with tf.variable_scope(scope_name, 'LeNet', [im]):
            out = slim.conv2d(im, 32, [5, 5], scope='conv1')
            out = slim.max_pool2d(out, [2, 2], 2, scope='pool1')
            out = slim.conv2d(out, 64, [5, 5], scope='conv2')
            out = slim.max_pool2d(out, [2, 2], 2, scope='pool2')
            out = slim.flatten(out)
        return out

    def extract_custom(self, im, scope_name):
        with tf.variable_scope(scope_name, 'custom', [im]):
            out = slim.conv2d(im, 16, [5, 5], scope='conv1')
            out = slim.max_pool2d(out, [2, 2], 2, scope='pool1')
            out = slim.conv2d(out, 32, [5, 5], scope='conv2')
            out = slim.max_pool2d(out, [2, 2], 2, scope='pool2')
            out = slim.conv2d(out, 64, [5, 5], scope='conv3')
            out = slim.max_pool2d(out, [2, 2], 2, scope='pool3')
            out = slim.flatten(out)
        return out
    
    def extract_features(self, im, scope_name, reuse=False, arch='resnet_v2_50'):
        if arch == 'resnet_v2_50':
            return self.extract_features_resnet50(im, scope_name, reuse=False)
        if arch == 'resnet_v2_101':
            return self.extract_features_resnet101(im, scope_name, reuse=False)
        if arch == 'lenet':
            return self.extract_lenet(im, scope_name)
        if arch == 'custom_net':
            return self.extract_custom(im, scope_name)
        raise ValueError('Net architecture is unsupported!!!')

    def exclude_finetune_scopes(self):
        return ['densenet', 'resnet_v2_50/logits', 'resnet_v2_101/logits']

def initialize(args):
    return Model(**args)