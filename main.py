from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import net_solver 
import model
from data import Data
from utils import *
import pickle

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('im_size', 224, help='image size (integer)')
flags.DEFINE_integer('epoch', 100, help='number of epochs (integer)')
flags.DEFINE_integer('batch_size', 32, help='number of images in a batch (integer)')
flags.DEFINE_string('working_dir', 'model-config_2', help='working directory')
flags.DEFINE_string('training_data', 'data/train/train', help='directory of training data')
flags.DEFINE_string('test_data', 'data/test/test', help='directory of training data')
flags.DEFINE_bool('augmentation', True, help='whether to use data augmentation during training')
flags.DEFINE_bool('use_tfrecord', True, help='whether to create and load data in tfrecord')

if __name__ == "__main__":
    tf.reset_default_graph()

    fnames = [f.name for f in os.scandir(FLAGS.training_data) if f.is_dir()]
    classes = {name: int(name) for name in fnames if name.isnumeric()}
    class_names = [c for c in sorted(classes.keys(), key=lambda x: classes[x])]
    class_count = dict.fromkeys(class_names, 0)
    for c in class_names:
        class_count[c] = float(len(glob_re('.*\.(jpg|JPG).*', os.listdir(os.path.join(FLAGS.training_data, c)))))

    print('='*50)
    print('Training dataset summaries')
    print('='*50)
    for c in class_names:
        print('{:<15s}{:<15d}'.format(c, int(class_count[c])))
    print('-'*50)
    print('{:<15s}{:<15f}'.format('Summation', sum([class_count[c] for c in class_names])))
    print('='*50)

    N = sum(list(class_count.values()))
    K = len(class_names)
    print('Number of classes: ', K)
    if K > 2:
        loss_weight = [100.*(1-class_count[c]/N)/(K-1) for c in class_names]
    else:
        loss_weight = class_count[class_names[0]]/class_count[class_names[1]]
    print('Loss weight: ', loss_weight)

    net = model.initialize({'is_training':False, 
                            'loss_weight':loss_weight, 
                            'use_focal_loss':True, 
                            'use_os_loss':True,
                            'num_class':K})

    solver = net_solver.initialize({
        'working_dir'           : FLAGS.working_dir,
        'label_to_index'        : classes,
        'max_iter'              : FLAGS.epoch * (N//FLAGS.batch_size),
        'save_iter'             : 5  * (N//FLAGS.batch_size), 
        'lr_start_decay'        : 30 * (N//FLAGS.batch_size),
        'lr_decay_every'        : 10 * (N//FLAGS.batch_size),
        'learning_rate'         : 0.0001,
        'pretrained_resnet'     : 'pretrained/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt',
        'batch_size'            : FLAGS.batch_size
    })
    solver.setup_data(
        FLAGS.training_data, 
        augmentation=FLAGS.augmentation, 
        use_tfrecord=FLAGS.use_tfrecord
    )
    images, labels = solver.get_data_source('train')
    net.set_data_source(images, labels)
    net.build_model()
    solver.setup_net(net, ckpt_id=82325)
    solver.setup_summary()
    solver.train()
    
    # test_err, others = solver.validate(src='test')
    # index_to_label = {k:v for v,k in solver.label_to_index.items()}

    # print('{:<15s}{:<23f}'.format('Accuracy', 1. - test_err))
    # for i in range(net.num_class):
    #     print('{:<15s}{:<15s}'.format('Metric', index_to_label[i]))
    #     print('{:<15s}{:<15f}'.format('Support', others[3][i]))
    #     print('{:<15s}{:<15f}'.format('Precision', others[0][i]))
    #     print('{:<15s}{:<15f}'.format('Recall', others[1][i]))
    #     print('{:<15s}{:<15f}'.format('F1', others[2][i]))
    #     print('\n==\n')

    #solver.export_to_pb(os.path.join(FLAGS.working_dir, 'pb'))


