import tensorflow as tf
import pathlib
import os
import argparse
from tqdm import tqdm


def check_and_remove(paths):
    ret = []
    path_input = tf.compat.v1.placeholder(tf.string)
    file_contents = tf.io.read_file(path_input)
    image = tf.image.decode_jpeg(file_contents, channels=3)
    with tf.Session() as sess:
        for p in tqdm(paths):
            try:
                image_data = sess.run(image, feed_dict={path_input:p})
                assert image_data.shape[2] == 3, 'Single-channel image is disallowed'
                ret.append(p)
            except:
                print('\n\nRemoving ', p)
                os.remove(p)
    return ret

parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    data_root = pathlib.Path(args.path)
    image_paths = list(data_root.glob('*.jpg')) + list(data_root.glob('*.JPG'))
    image_paths = [str(p) for p in image_paths]
    print('%d files to check' % len(image_paths))
    image_paths = check_and_remove(image_paths)
    print('%d files remain' % len(image_paths))
