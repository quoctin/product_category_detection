import tensorflow as tf 
import numpy as np 

if __name__ == "__main__":
    X = [1,2,3,4,5,6,7,8,9]
    y = X

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(buffer_size=len(y))
    ds = ds.repeat()
    ds = ds.batch(1)
    ds = ds.prefetch(buffer_size=2)

    iterator = tf.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(ds),\
                            tf.compat.v1.data.get_output_shapes(ds))
    initializer = iterator.make_initializer(ds)
    
    X_, y_ = iterator.get_next()
    input_X = tf.placeholder_with_default(X_, shape=(None))
    input_y = tf.placeholder_with_default(y_, shape=(None))
    output = input_X*input_y

    with tf.Session() as sess:
        sess.run(initializer)
        for i in range(20):
            print('Product: %f' % sess.run(output))
