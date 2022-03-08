import os
import numpy as np
import tensorflow as tf
from data.rectanglesImagesDataSet import load_rectangles_im


def _shuffle_data_set(data_images, data_labels):
    index_shuffle = np.arange(len(data_images))
    np.random.shuffle(index_shuffle)
    data_images = data_images[index_shuffle]
    data_labels = data_labels[index_shuffle]
    return data_images, data_labels


def _divide_data_set_by_class_random(data_images, data_labels, samples_size_each_class):
    data_images, data_labels = _shuffle_data_set(data_images, data_labels)
    data_dic = {}
    for images_item, labels_item in zip(data_images, data_labels):
        if labels_item in data_dic:
            data_dic[labels_item].append(images_item)
        else:
            data_dic[labels_item] = [images_item]
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for key in data_dic.keys():
        train_images.extend(data_dic[key][samples_size_each_class:])
        train_labels.extend([key for _ in range(len(data_dic[key]) - samples_size_each_class)])
        test_images.extend(data_dic[key][:samples_size_each_class])
        test_labels.extend([key for _ in range(samples_size_each_class)])
    train_images, train_labels = _shuffle_data_set(train_images, train_labels)
    test_images, test_labels = _shuffle_data_set(test_images, test_labels)
    return (train_images, train_labels), (test_images, test_labels)


def _load_rectangles_im(path=None, is_valid=False):
    # read data
    if not path:
        path = os.path.abspath(r'./rectanglesImages')
    (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = load_rectangles_im(path)
    train_images, train_labels = _shuffle_data_set(train_images, train_labels)
    # deal with valid dataset
    if is_valid:
        return (train_images, train_labels), (valid_images, valid_labels)
    else:
        train_images = np.concatenate([train_images, valid_images], 0)
        train_labels = np.concatenate([train_labels, valid_labels], 0)
        return (train_images, train_labels), (test_images, test_labels)


def _normalized_image(images_set, data_shape):
    images_set = images_set.astype(np.float32)
    for image_index in range(len(images_set)):
        mean = np.mean(images_set[image_index])
        var = np.sqrt(np.var(images_set[image_index]))
        if var > 1.0/data_shape[0]:
            tmp_var = var
        else:
            tmp_var = 1.0/data_shape[0]
        images_set[image_index] = (images_set[image_index] - mean)/tmp_var


def _parse_function_rectangles_im(data, label):
    data = tf.cast(data, dtype=tf.float32)
    data = tf.reshape(data, [28, 28, 1])
    label = tf.cast(label, tf.int64)
    return data, label


def _make_dataset_with_tf_data(data_images, data_labels, data_shape, repeat_num, batch_size, is_shuffle=True,
                               parse_fn=_parse_function_rectangles_im):
    _normalized_image(data_images, data_shape)
    data_set = tf.data.Dataset.from_tensor_slices((data_images, data_labels))
    data_set = data_set.map(parse_fn)
    data_set = data_set.repeat(repeat_num)
    if is_shuffle:
        buffer_size = int(len(data_images)*1.2)
        data_set = data_set.shuffle(buffer_size=buffer_size)
    data_set = data_set.batch(batch_size)
    data_iterator = tf.data.Iterator.from_structure(data_set.output_types, data_set.output_shapes)
    data_next_element = data_iterator.get_next()
    data_init_op = data_iterator.make_initializer(data_set)
    return data_init_op, data_next_element


# make dataset for rectangles
def get_rectangles_im(epochs, batch_size, is_valid=False):
    # load data
    (train_images, train_labels), (test_images, test_labels) = _load_rectangles_im(is_valid=is_valid)

    # training dataset
    train_init_op, train_next_element = \
        _make_dataset_with_tf_data(train_images, train_labels, [28, 28, 1], epochs*28, batch_size,
                                   parse_fn=_parse_function_rectangles_im)
    # testing dataset
    test_init_op, test_next_element = \
        _make_dataset_with_tf_data(test_images, test_labels, [28, 28, 1], epochs*28, batch_size, False,
                                   parse_fn=_parse_function_rectangles_im)

    return (train_init_op, train_next_element), (test_init_op, test_next_element)

